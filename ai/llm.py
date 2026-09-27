"""Cloud LLM fallback for questions the FAQ matcher isn't confident about.

Providers are tried in config.AI_PROVIDER_ORDER; any without an API key is
skipped, and a failing one (network, quota, bad key) hands over to the next.
The LLM only ever sees the top FAQ entries as its knowledge base and is told
to reply ESCALATE when they don't cover the question, so it rephrases and
combines real policy instead of inventing new policy.
"""

import sys

import config

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


def _require(module: str, package: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError:
        raise ProviderError(f"{package} isn't installed, run: pip install {package}") from None


class GroqProvider:
    name = "groq"

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


PROVIDERS = {"groq": GroqProvider, "gemini": GeminiProvider, "claude": ClaudeProvider}


class LLMAssistant:
    def __init__(self, providers: list | None = None):
        if providers is None:
            providers = [PROVIDERS[n]() for n in config.AI_PROVIDER_ORDER if n in PROVIDERS]
            providers = [p for p in providers if p.key]
        self.providers = providers
        self._disabled: set[str] = set()

    @property
    def available(self) -> list[str]:
        return [p.name for p in self.providers if p.name not in self._disabled]

    def answer(self, query: str, knowledge: list[dict], history: list[dict]) -> tuple[str | None, str | None]:
        """Returns (reply, provider_name). reply is None when every provider
        failed or the model decided the question needs a human."""
        kb = "\n".join(f"- Q: {f['question']}\n  A: {f['answer']}" for f in knowledge) or "(empty)"
        messages = [*history, {"role": "user", "content": f"Knowledge base:\n{kb}\n\nCustomer: {query}"}]

        for p in self.providers:
            if p.name in self._disabled:
                continue
            try:
                text = (p.generate(SYSTEM_PROMPT, messages) or "").strip()
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
