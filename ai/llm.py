"""LLM fallback for questions the FAQ matcher isn't confident about.

Providers are tried in config.AI_PROVIDER_ORDER: a local model through
Ollama first (free, private, works offline), then cloud APIs. Any provider
that isn't set up is skipped - no Ollama running, no API key - and a failing
one (network, quota, bad key) hands over to the next.
The LLM only ever sees the top FAQ entries as its knowledge base and is told
to reply ESCALATE when they don't cover the question, so it rephrases and
combines real policy instead of inventing new policy.
"""

import json
import sys
import urllib.request

import config
import network

SYSTEM_PROMPT = (
    f"You are {config.BOT_NAME}, a friendly customer support voice assistant for "
    f"{config.COMPANY_NAME}, an online shopping store in India.\n"
    "Answer the customer using ONLY the facts in the knowledge base included in "
    "their message. You may combine or rephrase facts, and use the earlier "
    "conversation to understand follow-up questions.\n"
    "Your reply is spoken aloud, so answer in 1 to 3 short, natural sentences, "
    "with no markdown, lists or emojis.\n"
    f"If the knowledge base does not answer the question, reply with exactly: "
    f"{config.AI_ESCALATE_TOKEN}\n"
    "Never invent policies, prices, dates or order details."
)


class ProviderError(Exception):
    pass


class Skip(Exception):
    """This provider shouldn't handle this question; let the next one try."""


def _require(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise ProviderError(f"{package} isn't installed, run: pip install {package}") from None


class OllamaProvider:
    """A small model running locally through Ollama (https://ollama.com).
    Ollama serves a plain HTTP API on localhost, so no client library is
    needed. CPU inference is Ollama's default when there's no GPU."""

    name = "ollama"
    local = True

    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.max_faqs = config.OLLAMA_MAX_FAQS
        self.key = "local" if self._model_installed() else ""

    def _request(self, path: str, payload: dict | None = None, timeout: float = 2.0) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(config.OLLAMA_URL + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _model_installed(self) -> bool:
        try:
            tags = self._request("/api/tags", timeout=1.0)
        except (OSError, ValueError):
            return False  # Ollama not installed or not running: simply not used
        names = {m.get("name", "") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def generate(self, system: str, messages: list[dict], relevant: int = 1,
                 language: str = "English") -> str | None:
        if relevant == 0:
            # No FAQ entry is even loosely related. A small model would spend
            # many seconds on a CPU only to answer ESCALATE; skip straight on.
            raise Skip()
        if language not in config.OLLAMA_LANGUAGES:
            raise Skip()  # a 1.5B model writes poor Telugu; leave it to bigger models or escalation
        r = self._request("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.3, "num_predict": 200, "num_ctx": config.OLLAMA_NUM_CTX},
        }, timeout=config.OLLAMA_TIMEOUT)
        return r.get("message", {}).get("content")


class GroqProvider:
    name = "groq"
    local = False
    max_faqs = None

    def __init__(self):
        self.key, self.model = config.GROQ_API_KEY, config.GROQ_MODEL
        self._client = None

    def generate(self, system: str, messages: list[dict]) -> str | None:
        if self._client is None:
            groq = _require("groq", "groq")
            self._client = groq.Groq(api_key=self.key, timeout=config.AI_TIMEOUT_SECONDS, max_retries=1)
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=config.AI_MAX_TOKENS,
            temperature=0.3,
        )
        return r.choices[0].message.content


class GeminiProvider:
    name = "gemini"
    local = False
    max_faqs = None

    def __init__(self):
        self.key, self.model = config.GEMINI_API_KEY, config.GEMINI_MODEL
        self._client = None

    def generate(self, system: str, messages: list[dict]) -> str | None:
        genai = _require("google.genai", "google-genai")
        types = genai.types
        if self._client is None:
            self._client = genai.Client(
                api_key=self.key,
                http_options=types.HttpOptions(timeout=config.AI_TIMEOUT_SECONDS * 1000),
            )
        contents = [
            types.Content(role="user" if m["role"] == "user" else "model",
                          parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        r = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=config.AI_MAX_TOKENS,
                temperature=0.3,
                # No tools are used; disabling this also silences an SDK warning
                # that would otherwise print mid-conversation.
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return r.text


class ClaudeProvider:
    name = "claude"
    local = False
    max_faqs = None

    def __init__(self):
        self.key, self.model = config.ANTHROPIC_API_KEY, config.CLAUDE_MODEL
        self._client = None

    def generate(self, system: str, messages: list[dict]) -> str | None:
        if self._client is None:
            anthropic = _require("anthropic", "anthropic")
            self._client = anthropic.Anthropic(api_key=self.key, timeout=config.AI_TIMEOUT_SECONDS, max_retries=1)
        r = self._client.beta.messages.create(
            model=self.model,
            max_tokens=config.AI_MAX_TOKENS,
            system=system,
            messages=messages,
            # Short FAQ-grounded replies don't need deep reasoning; low effort
            # keeps voice latency down.
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if r.stop_reason == "refusal":
            return None
        return "".join(b.text for b in r.content if b.type == "text")


PROVIDERS = {"ollama": OllamaProvider, "groq": GroqProvider, "gemini": GeminiProvider,
             "claude": ClaudeProvider}


class LLMAssistant:
    def __init__(self, providers: list | None = None, local_only: bool = False):
        if providers is None:
            providers = [PROVIDERS[n]() for n in config.AI_PROVIDER_ORDER if n in PROVIDERS]
            providers = [p for p in providers if p.key]
        if local_only:
            providers = [p for p in providers if getattr(p, "local", False)]
        self.providers = providers
        self._disabled: set[str] = set()

    def describe(self) -> str:
        return ", ".join(f"{p.name} ({p.model})" if getattr(p, "local", False) else p.name
                         for p in self.providers if p.name not in self._disabled)

    @property
    def available(self) -> list[str]:
        return [p.name for p in self.providers if p.name not in self._disabled]

    def answer(self, query: str, knowledge: list[dict], history: list[dict],
               language: str = "English", relevant: int | None = None) -> tuple[str | None, str | None]:
        """Returns (reply, provider_name). reply is None when every provider
        failed or the model decided the question needs a human.

        knowledge should be ranked most-relevant first; `relevant` says how
        many of those entries actually matched the question. Small local
        models get only the top few (reading the prompt is the slow part on
        a CPU); cloud models get everything."""
        if relevant is None:
            relevant = len(knowledge)
        reply_in = f"\n\nReply in {language}." if language != "English" else ""

        for p in self.providers:
            if p.name in self._disabled:
                continue
            local = getattr(p, "local", False)
            if not local and not network.is_online():
                continue  # don't wait on a cloud timeout when we know we're offline
            limit = getattr(p, "max_faqs", None)
            entries = knowledge[:limit] if limit else knowledge
            kb = "\n".join(f"- Q: {f['question']}\n  A: {f['answer']}" for f in entries) or "(empty)"
            messages = [*history, {"role": "user",
                                   "content": f"Knowledge base:\n{kb}\n\nCustomer: {query}{reply_in}"}]
            try:
                if local:
                    text = (p.generate(SYSTEM_PROMPT, messages, relevant=relevant, language=language) or "").strip()
                else:
                    text = (p.generate(SYSTEM_PROMPT, messages) or "").strip()
            except Skip:
                continue
            except ProviderError as e:
                # Missing SDK won't fix itself mid-session; stop retrying it.
                self._disabled.add(p.name)
                print(f"  [ai] {p.name} disabled: {e}", file=sys.stderr)
                continue
            except Exception as e:  # network, quota, auth: try the next provider
                print(f"  [ai] {p.name} failed ({type(e).__name__}): {str(e)[:120]}", file=sys.stderr)
                continue
            if not text or text.upper().startswith(config.AI_ESCALATE_TOKEN):
                return None, p.name
            return text, p.name
        return None, None
