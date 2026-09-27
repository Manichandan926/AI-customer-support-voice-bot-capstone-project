"""Local model via Ollama. A fake Ollama server (real HTTP on localhost) stands
in for the real one, so the requests the provider sends are checked end to
end without installing Ollama or downloading a model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import config
import network
from ai.llm import LLMAssistant, OllamaProvider
from dialogue.manager import DialogueManager


class FakeOllama(BaseHTTPRequestHandler):
    models = ["qwen2.5:1.5b"]
    reply = "You can reach us any time at 1800-123-4567."
    requests: list = []

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(body)
        self._send({"message": {"role": "assistant", "content": self.reply}, "done": True})

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "OLLAMA_URL", f"http://127.0.0.1:{server.server_port}")
    FakeOllama.requests = []
    FakeOllama.models = ["qwen2.5:1.5b"]
    yield FakeOllama
    server.shutdown()


def test_detected_when_model_is_pulled(ollama):
    assert OllamaProvider().key


def test_not_used_when_model_missing(ollama):
    ollama.models = ["llama3:8b"]
    assert not OllamaProvider().key


def test_not_used_when_ollama_not_running(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    assert not OllamaProvider().key


def test_request_is_cpu_friendly_and_non_streaming(ollama):
    reply, provider = LLMAssistant([OllamaProvider()]).answer(
        "how do I contact you", [{"question": "q", "answer": "a"}], [])
    assert reply == ollama.reply and provider == "ollama"
    req = ollama.requests[0]
    assert req["model"] == config.OLLAMA_MODEL and req["stream"] is False
    assert req["options"]["num_ctx"] == config.OLLAMA_NUM_CTX
    assert req["messages"][0]["role"] == "system"


def test_local_model_gets_only_top_faqs(ollama):
    faqs = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(40)]
    LLMAssistant([OllamaProvider()]).answer("question", faqs, [], relevant=10)
    prompt = ollama.requests[0]["messages"][-1]["content"]
    assert prompt.count("- Q:") == config.OLLAMA_MAX_FAQS and "q0" in prompt and "q39" not in prompt


def test_skipped_when_nothing_relevant(ollama):
    reply, provider = LLMAssistant([OllamaProvider()]).answer("weather?", [{"question": "q", "answer": "a"}],
                                                              [], relevant=0)
    assert (reply, provider) == (None, None) and ollama.requests == []


def test_skipped_for_telugu(ollama):
    LLMAssistant([OllamaProvider()]).answer("ప్రశ్న", [{"question": "q", "answer": "a"}], [], language="Telugu")
    assert ollama.requests == []


class CloudFake:
    name, key, local, max_faqs = "cloud", "k", False, None

    def __init__(self):
        self.calls = 0

    def generate(self, system, messages):
        self.calls += 1
        return "cloud answer"


def test_offline_skips_cloud_but_uses_local(ollama, monkeypatch):
    network.mark_offline()
    cloud = CloudFake()
    reply, provider = LLMAssistant([cloud, OllamaProvider()]).answer("q", [{"question": "q", "answer": "a"}], [])
    assert provider == "ollama" and cloud.calls == 0


def test_local_only_filters_out_cloud(ollama):
    assistant = LLMAssistant([CloudFake(), OllamaProvider()], local_only=True)
    assert assistant.available == ["ollama"]


def test_end_to_end_low_confidence_question_answered_locally(ollama):
    bot = DialogueManager(ai=LLMAssistant([OllamaProvider()]))
    # Scores just below the clarify band: no FAQ answers it, but one is related.
    t = bot.handle("can my friend collect the parcel for me")
    assert t.action == "ai" and t.source == "ai:ollama"
    # Knowledge is ranked: the loosely related FAQ comes first in what the model saw.
    prompt = ollama.requests[0]["messages"][-1]["content"]
    assert prompt.index("My order is late") < prompt.index("Customer:")
    assert prompt.count("- Q:") == config.OLLAMA_MAX_FAQS
