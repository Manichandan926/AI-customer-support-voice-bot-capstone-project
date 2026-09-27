"""Download the offline models into models/ (one-time, needs internet).

  python -m tools.download_models                  all languages: speech models + voices (~330 MB)
  python -m tools.download_models --lang en,hi     just English and Hindi
  python -m tools.download_models --voices-only    Piper voices only
  python -m tools.download_models --status         show what's installed

Files already present are skipped, so it's safe to re-run after an
interrupted download.
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from asr.offline_asr import is_available as asr_ready, model_path  # noqa: E402
from tts.offline_tts import is_available as voice_ready, voice_path  # noqa: E402

VOSK_URL = "https://alphacephei.com/vosk/models/{name}.zip"
PIPER_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{family}/{locale}/{speaker}/{quality}/{name}{ext}"


def piper_url(name: str, ext: str) -> str:
    locale, speaker, quality = name.split("-")      # e.g. te_IN, maya, medium
    return PIPER_URL.format(family=locale.split("_")[0], locale=locale, speaker=speaker,
                            quality=quality, name=name, ext=ext)


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 16):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)  # only a complete file ever gets the real name


def get_vosk(lang: str) -> None:
    name = config.VOSK_MODELS[lang]
    if asr_ready(lang):
        print(f"[{lang}] speech model {name}: already installed")
        return
    zip_path = config.MODELS_DIR / "vosk" / f"{name}.zip"
    fetch(VOSK_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(zip_path.parent)
    zip_path.unlink()
    if not asr_ready(lang):
        shutil.rmtree(model_path(lang), ignore_errors=True)
        raise RuntimeError(f"{name}: archive didn't contain the expected model layout")
    print(f"[{lang}] speech model {name}: installed")


def get_piper(lang: str) -> None:
    name = config.PIPER_VOICES[lang]
    if voice_ready(lang):
        print(f"[{lang}] voice {name}: already installed")
        return
    onnx = voice_path(lang)
    fetch(piper_url(name, ".onnx.json"), onnx.with_suffix(".onnx.json"))
    fetch(piper_url(name, ".onnx"), onnx)
    print(f"[{lang}] voice {name}: installed")


def status() -> None:
    print(f"models folder: {config.MODELS_DIR}")
    for lang in config.PIPER_VOICES:
        asr = "yes" if asr_ready(lang) else "no "
        tts = "yes" if voice_ready(lang) else "no "
        print(f"  {lang}: offline speech recognition {asr} | offline voice {tts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default=",".join(config.PIPER_VOICES), help="comma-separated: en,hi,te")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--voices-only", action="store_true")
    group.add_argument("--speech-only", action="store_true")
    parser.add_argument("--status", action="store_true", help="show installed models and exit")
    args = parser.parse_args()

    if args.status:
        status()
        return
    langs = [l.strip() for l in args.lang.split(",") if l.strip()]
    unknown = [l for l in langs if l not in config.PIPER_VOICES]
    if unknown:
        parser.error(f"unknown language(s): {', '.join(unknown)}")
    failed = []
    for lang in langs:
        steps = ([] if args.voices_only else [get_vosk]) + ([] if args.speech_only else [get_piper])
        for step in steps:
            try:
                step(lang)
            except Exception as e:  # keep going: one bad download shouldn't block the rest
                failed.append(f"{lang}/{step.__name__}")
                print(f"[{lang}] FAILED: {e}")
    print()
    status()
    if failed:
        sys.exit(f"\nSome downloads failed ({', '.join(failed)}). Re-run the same command to retry.")


if __name__ == "__main__":
    main()
