"""Speech-to-text via the SpeechRecognition library's free Google recognizer.

Audio is captured with sounddevice rather than SpeechRecognition's own
Microphone class, because that needs PyAudio, which often has no prebuilt
wheel for new Python versions on Windows.

Recording stops when the speaker does (energy-based endpointing), instead of
after a fixed number of seconds: short questions get answered sooner, and
long ones aren't cut off.
"""

from collections import deque

import config

FRAME_SECONDS = config.VAD_FRAME_MS / 1000


def noise_level(levels: list[float]) -> float:
    """Room noise from calibration frames. Zero frames (mic still waking up)
    are dropped, and the 75th percentile is used rather than the median
    because real rooms are bursty - fans, typing, distant voices."""
    live = sorted(l for l in levels if l > 5) or [0.0]
    return live[round(0.75 * (len(live) - 1))]


class Endpointer:
    """Decides, frame by frame, when an utterance starts and ends, from the
    frames' RMS levels alone. Kept free of audio I/O so it can be unit-tested
    with synthetic levels.

    feed() returns "waiting", "speaking", "done" (utterance complete) or
    "timeout" (nobody spoke).
    """

    def __init__(self, noise_floor: float):
        self.floor = noise_floor
        self._frames = 0
        self._loud_run = 0
        self._silent_run = 0
        self.started = False

    @property
    def threshold(self) -> float:
        # Relative to the room: a fan or traffic raises the floor, so
        # background noise alone never counts as speech.
        return max(config.SILENCE_RMS, self.floor * config.VAD_NOISE_FACTOR)

    def feed(self, rms: float) -> str:
        self._frames += 1
        loud = rms >= self.threshold
        if not self.started:
            if not loud:
                # Keep tracking the room while waiting: laptop mics often
                # ramp their gain up for a second after the stream opens.
                self.floor = 0.9 * self.floor + 0.1 * rms
            self._loud_run = self._loud_run + 1 if loud else 0
            # A few consecutive loud frames, so a single click or bump isn't speech.
            if self._loud_run * FRAME_SECONDS >= config.VAD_MIN_SPEECH:
                self.started = True
                self._frames = self._loud_run
                return "speaking"
            if self._frames * FRAME_SECONDS >= config.VAD_START_TIMEOUT:
                return "timeout"
            return "waiting"
        self._silent_run = 0 if loud else self._silent_run + 1
        if self._silent_run * FRAME_SECONDS >= config.VAD_END_SILENCE:
            return "done"
        if self._frames * FRAME_SECONDS >= config.VAD_MAX_SECONDS:
            return "done"
        return "speaking"


class SpeechToText:
    def __init__(self, language: str = config.ASR_LANGUAGE):
        self.language = language
        self._sr = None
        self._recognizer = None

    def _load(self):
        if self._recognizer is None:
            try:
                import speech_recognition as sr
            except ImportError:
                raise RuntimeError("SpeechRecognition isn't installed, run: pip install SpeechRecognition") from None
            self._sr = sr
            self._recognizer = sr.Recognizer()
        return self._sr

    def record_utterance(self, on_speech_start=None):
        """Records from the default mic until the speaker stops. Returns int16
        samples, or None if nobody spoke before the start timeout."""
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError:
            raise RuntimeError("Microphone support isn't installed, run: pip install sounddevice numpy") from None

        block = int(config.SAMPLE_RATE * FRAME_SECONDS)
        rms = lambda x: float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))
        # Keep a little audio from before speech was detected, otherwise the
        # first syllable ("Where...") is clipped and recognition suffers.
        preroll = deque(maxlen=int(config.VAD_PREROLL / FRAME_SECONDS))
        frames = []
        try:
            with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="int16", blocksize=block) as stream:
                for _ in range(int(config.VAD_WARMUP / FRAME_SECONDS)):
                    stream.read(block)  # mic warm-up: often pure zeros, which would fake a silent room
                calib = [rms(stream.read(block)[0]) for _ in range(int(config.VAD_CALIBRATE / FRAME_SECONDS))]
                ep = Endpointer(noise_floor=noise_level(calib))
                self.last_noise_floor = ep.floor
                while True:
                    data = stream.read(block)[0].copy()
                    state = ep.feed(rms(data))
                    if state == "timeout":
                        return None
                    if state == "waiting":
                        preroll.append(data)
                        continue
                    if not frames:
                        frames.extend(preroll)
                        if on_speech_start:
                            on_speech_start()
                    frames.append(data)
                    if state == "done":
                        break
        except Exception as e:
            raise RuntimeError(f"Couldn't record from the microphone ({e}). Check that a mic is connected "
                               "and that apps are allowed to use it.") from None
        return np.concatenate(frames)

    def transcribe_samples(self, samples) -> str:
        sr = self._load()
        data = sr.AudioData(samples.tobytes(), config.SAMPLE_RATE, 2)
        return self._recognize(data)

    def transcribe_file(self, path: str) -> str:
        sr = self._load()
        with sr.AudioFile(path) as source:
            data = self._recognizer.record(source)
        return self._recognize(data)

    def listen(self, on_speech_start=None) -> str:
        samples = self.record_utterance(on_speech_start)
        return "" if samples is None else self.transcribe_samples(samples)

    def _recognize(self, data) -> str:
        try:
            return self._recognizer.recognize_google(data, language=self.language)
        except self._sr.UnknownValueError:
            return ""  # speech was unintelligible
        except self._sr.RequestError as e:
            raise RuntimeError(f"Google speech service unreachable ({e}). Check your internet connection, "
                               "or type your question instead.") from None


def _mic_check() -> None:
    """python -m asr.speech_to_text  - shows mic levels and one transcript,
    for tuning VAD_* settings on a new machine (e.g. the Raspberry Pi)."""
    import time
    stt = SpeechToText()
    print("Measuring room noise, then say something like 'where is my order'...")
    t0 = time.perf_counter()
    samples = stt.record_utterance(on_speech_start=lambda: print(f"  speech started at {time.perf_counter()-t0:.1f}s"))
    floor = stt.last_noise_floor
    print(f"  noise floor {floor:.0f} RMS -> speech threshold "
          f"{max(config.SILENCE_RMS, floor * config.VAD_NOISE_FACTOR):.0f} RMS")
    if samples is None:
        print("  no speech detected. Speak closer to the mic, or lower VAD_NOISE_FACTOR in config.py")
        return
    import numpy as np
    level = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    print(f"  recorded {len(samples)/config.SAMPLE_RATE:.1f}s, stopped at {time.perf_counter()-t0:.1f}s, "
          f"average speech level {level:.0f} RMS")
    print(f"  transcript: {stt.transcribe_samples(samples)!r}")


def _level_meter(seconds: float = 8.0) -> None:
    """python -m asr.speech_to_text --levels  - records a fixed window, draws
    the loudness over time, and recommends VAD_NOISE_FACTOR for this mic."""
    import numpy as np
    import sounddevice as sd
    print(f"Stay quiet for 2 seconds, then say 'where is my order' at normal volume. Recording {seconds:.0f}s...")
    x = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    x = x[:, 0].astype(np.float32)
    block = int(config.SAMPLE_RATE * FRAME_SECONDS)
    levels = [float(np.sqrt(np.mean(x[i:i + block] ** 2))) for i in range(0, len(x) - block, block)]
    skip = int(config.VAD_WARMUP / FRAME_SECONDS)
    quiet = levels[skip:int(2.0 / FRAME_SECONDS)]
    noise = noise_level(quiet)
    speech = float(np.percentile(levels[int(2.0 / FRAME_SECONDS):], 90))
    step = int(0.25 / FRAME_SECONDS)
    peak = max(levels) or 1
    for i in range(0, len(levels), step):
        lvl = max(levels[i:i + step])
        print(f"  {i * FRAME_SECONDS:4.1f}s {lvl:6.0f} {'#' * int(40 * lvl / peak)}")
    print(f"\n  room noise ~{noise:.0f}, your speech ~{speech:.0f} (ratio {speech / max(noise, 1):.1f}x)")
    ratio = speech / max(noise, 1)
    if ratio < 1.6:
        print("  Speech is barely louder than the room. Move closer to the mic, or raise the mic volume:\n"
              "  Windows: Settings > System > Sound > Input > Microphone > Input volume (and Microphone Boost).")
    else:
        # Threshold halfway (geometrically) between room noise and speech.
        print(f"  Recommended: VAD_NOISE_FACTOR = {max(1.3, min(3.5, ratio ** 0.5)):.1f} in config.py "
              f"(currently {config.VAD_NOISE_FACTOR})")


if __name__ == "__main__":
    import sys
    _level_meter() if "--levels" in sys.argv else _mic_check()
