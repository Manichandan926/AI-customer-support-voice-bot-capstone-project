"""CLI entry point.

  python main.py                     text chat (smart rules + AI fallback if keys set)
  python main.py --mode voice        speak a question, hear the answer
  python main.py --speak             text input, spoken output
  python main.py --no-ai             rule engine only, fully offline
"""

import argparse
import os
import sys

import config
from dialogue.manager import DialogueManager

if os.name == "nt":
    os.system("")  # enables ANSI colours in the classic Windows console

DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = (
    "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m")
if not sys.stdout.isatty():  # piped/redirected output shouldn't contain escape codes
    DIM = BOLD = CYAN = GREEN = YELLOW = RED = RESET = ""
ACTION_COLOR = {"answer": GREEN, "order": GREEN, "ai": CYAN, "smalltalk": GREEN,
                "clarify": YELLOW, "escalate": RED, "exit": DIM}


def build_bot(use_ai: bool) -> DialogueManager:
    ai = None
    if use_ai:
        from ai.llm import LLMAssistant
        ai = LLMAssistant()
        if not ai.providers:
            ai = None
    return DialogueManager(ai=ai)


def print_banner(bot: DialogueManager, mode: str) -> None:
    ai = ", ".join(bot.ai.available) if bot.ai else "off (rule engine only)"
    print(f"{BOLD}{config.COMPANY_NAME} Customer Support - {config.BOT_NAME}{RESET}")
    print(f"{DIM}mode: {mode} | FAQs: {len(bot.matcher.faqs)} | AI fallback: {ai}")
    print(f"say or type 'exit' to quit{RESET}\n")


def print_turn(turn) -> None:
    color = ACTION_COLOR.get(turn.action, "")
    print(f"{BOLD}{config.BOT_NAME}:{RESET} {turn.response}")
    print(f"{DIM}  [{color}{turn.action}{RESET}{DIM} | confidence {turn.confidence:.2f}"
          f"{' | ' + turn.source if turn.source else ''} | {turn.latency_ms:.1f} ms]{RESET}\n")


def print_stats(stats: dict) -> None:
    print(f"\n{BOLD}Session stats{RESET}")
    for k, v in stats.items():
        print(f"  {k:<15} {v}")


def get_voice_query(stt) -> str | None:
    typed = input(f"{DIM}[Enter] to speak for {config.RECORD_SECONDS}s, or type:{RESET} ").strip()
    if typed:
        return typed
    print(f"{CYAN}  Listening...{RESET}")
    try:
        text = stt.listen()
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
    args = parser.parse_args()

    bot = build_bot(use_ai=not args.no_ai)

    # Voice libraries are imported only when needed so text mode starts instantly.
    stt = tts = None
    if args.mode == "voice":
        from asr.speech_to_text import SpeechToText
        stt = SpeechToText()
    if args.mode == "voice" or args.speak:
        from tts.text_to_speech import TextToSpeech
        tts = TextToSpeech()

    print_banner(bot, args.mode + (" + speech output" if args.speak and args.mode == "text" else ""))
    greeting = f"Hi, I'm {config.BOT_NAME} from {config.COMPANY_NAME}. How can I help you today?"
    print(f"{BOLD}{config.BOT_NAME}:{RESET} {greeting}\n")

    def say(text: str) -> None:
        if tts:
            try:
                tts.speak(text)
            except RuntimeError as e:
                print(f"{RED}  {e}{RESET}")

    say(greeting)
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
            say(turn.response)
            if turn.is_exit:
                break
    except (KeyboardInterrupt, EOFError):
        print()
    print_stats(bot.stats())


if __name__ == "__main__":
    sys.exit(main())
