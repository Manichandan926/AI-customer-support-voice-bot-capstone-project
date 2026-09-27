"""Scripted demo: plays the showcase conversation turn by turn.

  python -m tools.demo              step through with Enter (rehearsal, or live if the mic fails)
  python -m tools.demo --auto       run straight through with short pauses
  python -m tools.demo --speak      also speak each reply (needs audio)

Every scene shows one capability, so the panel sees the whole system in a
couple of minutes regardless of room noise or network.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as cli  # noqa: E402  (reuses the CLI's formatting and voice selection)
from dialogue.manager import DialogueManager  # noqa: E402

SCENES = [
    ("Greeting and small talk", ["hi"]),
    ("Confident FAQ answer", ["how do I reset my password"]),
    ("Typo tolerance", ["i forgot my pasword"]),
    ("Order lookup: asks for the number, then finds it", ["track my order", "10567"]),
    ("Context memory: 'that' refers to the previous topic", ["how do I return an item", "how long does that take"]),
    ("Confidence routing: confirms before answering", ["my package was left open", "yes"]),
    ("Sentiment: empathy, then priority handoff on repeated anger",
     ["the product arrived broken", "this is the worst service, useless"]),
    ("Telugu, with an order lookup in Telugu", ["నా ఆర్డర్ ఎక్కడ ఉంది", "10234"]),
    ("Hindi, and romanized Hindi", ["मेरा रिफंड कब आएगा", "mera order kahan hai"]),
    ("Off-topic: handed to a human instead of guessing", ["what is the weather today"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto", action="store_true", help="no Enter between scenes")
    parser.add_argument("--speak", action="store_true", help="speak each reply")
    args = parser.parse_args()

    tts = None
    if args.speak:
        from tts.text_to_speech import TextToSpeech
        tts = TextToSpeech()
    bot = DialogueManager(ai=None)
    B, D, R = cli.BOLD, cli.DIM, cli.RESET

    for i, (title, lines) in enumerate(SCENES, 1):
        print(f"\n{B}[{i}/{len(SCENES)}] {title}{R}")
        for line in lines:
            print(f"{B}You:{R} {line}")
            turn = bot.handle(line)
            cli.print_turn(turn)
            if tts:
                try:
                    tts.speak(turn.response, voice=cli.voice_for(turn.lang), lang=turn.lang)
                except RuntimeError as e:
                    print(f"{D}  ({e}){R}")
        if args.auto:
            time.sleep(0.3)
        elif i < len(SCENES):
            input(f"{D}  [Enter] next scene{R}")
    cli.print_stats(bot.stats())


if __name__ == "__main__":
    main()
