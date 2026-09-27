"""CLI entry point.

  python main.py                     text chat (smart rules + AI fallback if keys set)
  python main.py --mode voice        speak a question, hear the answer
  python main.py --speak             text input, spoken output
  python main.py --no-ai             rule engine only, fully offline
  python main.py --lang te           start in Telugu (te) or Hindi (hi); text mode
                                     also switches language per message automatically
"""

import argparse
import os
import sys

import config
from dialogue.manager import DialogueManager
from nlu.language import SUPPORTED, load_pack

if os.name == "nt":
    os.system("")  # enables ANSI colours in the classic Windows console
# Telugu/Hindi text must survive redirected or piped I/O, where Windows would
# otherwise fall back to a legacy code page and crash on the first character.
for stream in (sys.stdout, sys.stderr, sys.stdin):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
if not sys.stdout.isatty():  # piped/redirected output shouldn't contain escape codes
    DIM = BOLD = CYAN = GREEN = YELLOW = RED = RESET = ""
ACTION_COLOR = {"answer": GREEN, "order": GREEN, "ai": CYAN, "smalltalk": GREEN,
                "clarify": YELLOW, "escalate": RED, "exit": DIM}


def build_bot(use_ai: bool, language: str) -> DialogueManager:
    ai = None
    if use_ai:
        from ai.llm import LLMAssistant
        ai = LLMAssistant()
        if not ai.providers:
            ai = None
    return DialogueManager(ai=ai, language=language)


def voice_for(lang: str) -> str:
    # English keeps the configurable voice (TTS_VOICE in .env); others come from their pack.
    return config.TTS_VOICE if lang == "en" else load_pack(lang).tts_voice


def print_banner(bot: DialogueManager, mode: str) -> None:
    ai = ", ".join(bot.ai.available) if bot.ai else "off (rule engine only)"
    print(f"{BOLD}{config.COMPANY_NAME} Customer Support - {config.BOT_NAME}{RESET}")
    print(f"{DIM}mode: {mode} | language: {load_pack(bot.lang).name} (auto-detect: English, Telugu, Hindi)"
          f" | FAQs: {len(bot.matcher.faqs)} | AI fallback: {ai}")
    print(f"say or type 'exit' to quit{RESET}\n")


def print_turn(turn) -> None:
    color = ACTION_COLOR.get(turn.action, "")
    print(f"{BOLD}{config.BOT_NAME}:{RESET} {turn.response}")
    print(f"{DIM}  [{color}{turn.action}{RESET}{DIM} | confidence {turn.confidence:.2f}"
          f"{' | ' + turn.source if turn.source else ''}"
          f"{' | mood: ' + turn.sentiment if turn.sentiment != 'neutral' else ''}"
          f"{' | ' + turn.lang if turn.lang != 'en' else ''}"
          f" | {turn.latency_ms:.1f} ms]{RESET}\n")


def print_stats(stats: dict) -> None:
    print(f"\n{BOLD}Session stats{RESET}")
    for k, v in stats.items():
        print(f"  {k:<15} {v}")


def get_voice_query(stt) -> str | None:
    typed = input(f"{DIM}[Enter] then speak, or type:{RESET} ").strip()
    if typed:
        return typed
    print(f"{CYAN}  Listening... (I'll stop when you stop talking){RESET}")
    try:
        text = stt.listen(on_speech_start=lambda: print(f"{CYAN}  Hearing you...{RESET}"))
    except RuntimeError as e:
        print(f"{RED}  {e}{RESET}")
        return None
    if not text:
        print(f"{YELLOW}  Didn't catch that - please try again.{RESET}")
        return None
    print(f"{BOLD}You (voice):{RESET} {text}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Customer Support Voice Bot")
    parser.add_argument("--mode", choices=["text", "voice"], default="text")
    parser.add_argument("--speak", action="store_true", help="speak replies in text mode")
    parser.add_argument("--no-ai", action="store_true", help="disable the cloud AI fallback")
    parser.add_argument("--lang", choices=SUPPORTED, default=config.DEFAULT_LANGUAGE,
                        help="starting language; voice mode listens in this language")
    args = parser.parse_args()

    bot = build_bot(use_ai=not args.no_ai, language=args.lang)

    # Voice libraries are imported only when needed so text mode starts instantly.
    stt = tts = None
    if args.mode == "voice":
        from asr.speech_to_text import SpeechToText
        # Speech recognition needs the language up front; English keeps the
        # configurable accent (ASR_LANGUAGE, default en-IN).
        stt = SpeechToText(config.ASR_LANGUAGE if args.lang == "en" else load_pack(args.lang).asr_code)
    if args.mode == "voice" or args.speak:
        from tts.text_to_speech import TextToSpeech
        tts = TextToSpeech()

    print_banner(bot, args.mode + (" + speech output" if args.speak and args.mode == "text" else ""))
    greeting = bot.greeting()
    print(f"{BOLD}{config.BOT_NAME}:{RESET} {greeting}\n")

    def say(text: str, lang: str) -> None:
        if tts:
            try:
                tts.speak(text, voice=voice_for(lang), lang=lang)
            except RuntimeError as e:
                print(f"{RED}  {e}{RESET}")

    say(greeting, bot.lang)
    try:
        while True:
            if stt:
                query = get_voice_query(stt)
                if query is None:
                    continue
            else:
                query = input(f"{BOLD}You:{RESET} ").strip()
                if not query:
                    continue
            turn = bot.handle(query)
            print_turn(turn)
            say(turn.response, turn.lang)
            if turn.is_exit:
                break
    except (KeyboardInterrupt, EOFError):
        print()
    print_stats(bot.stats())


if __name__ == "__main__":
    sys.exit(main())
