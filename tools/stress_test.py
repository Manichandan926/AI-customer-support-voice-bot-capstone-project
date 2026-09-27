"""Stress test under constrained hardware (memory cap + single slow core).

  python -m tools.stress_test                          1 GB cap, 1 core, full speed
  python -m tools.stress_test --cpu-fraction 0.4       also throttle the core (~Raspberry Pi 5 on a fast laptop)
  python -m tools.stress_test --cpu-fraction 0.17      (~Raspberry Pi 4)

On a real Raspberry Pi, run it with no --cpu-fraction: the hardware is the test.
Limits are enforced by the OS - a Windows Job Object, or setrlimit/affinity
on Linux - so exceeding the memory cap is a real failure, not an estimate.
"""

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------- constraints
def apply_limits(mem_mb: int, cpu_fraction: float | None) -> str:
    if os.name == "nt":
        return _limit_windows(mem_mb, cpu_fraction)
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
    note = "" if cpu_fraction is None else " (CPU throttling is Windows-only; on Linux the real CPU is the test)"
    return f"address space capped at {mem_mb} MB, pinned to 1 core{note}"


def _limit_windows(mem_mb: int, cpu_fraction: float | None) -> str:
    import ctypes
    from ctypes import wintypes as wt
    k32 = ctypes.windll.kernel32
    k32.CreateJobObjectW.restype = wt.HANDLE
    k32.GetCurrentProcess.restype = wt.HANDLE
    k32.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    k32.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wt.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wt.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wt.DWORD), ("SchedulingClass", wt.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class CPURATE(ctypes.Structure):
        _fields_ = [("ControlFlags", wt.DWORD), ("CpuRate", wt.DWORD)]

    job = k32.CreateJobObjectW(None, None)
    ext = EXT()
    ext.Basic.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    ext.ProcessMemoryLimit = mem_mb * 2**20
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(ext), ctypes.sizeof(ext)):
        raise OSError("could not set the memory limit")
    msg = f"committed memory capped at {mem_mb} MB, pinned to 1 core"
    if cpu_fraction:
        # CpuRate is in 1/100 % of the whole machine; one core is 1/ncpu of it.
        rate = max(1, int(cpu_fraction / os.cpu_count() * 10000))
        cpu = CPURATE(0x1 | 0x4, rate)  # ENABLE | HARD_CAP
        if not k32.SetInformationJobObject(job, 15, ctypes.byref(cpu), ctypes.sizeof(cpu)):
            raise OSError("could not set the CPU cap")
        msg += f", throttled to {cpu_fraction:.0%} of that core"
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise OSError("could not join the job object")
    k32.SetProcessAffinityMask(k32.GetCurrentProcess(), 1)
    return msg


def memory_mb() -> tuple[float, float]:
    """(current, peak) resident memory in MB."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                        ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
        pmc = PMC(cb=ctypes.sizeof(PMC))
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        cur = peak
    return cur, peak


# ------------------------------------------------------------------- phases
def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p * (len(s) - 1)))]


def phase_startup(results):
    t = time.perf_counter()
    from dialogue.manager import DialogueManager
    results["import_ms"] = (time.perf_counter() - t) * 1000
    bot = DialogueManager(ai=None)
    for lang, q in (("en", "where is my order"), ("te", "నా ఆర్డర్ ఎక్కడ ఉంది"), ("hi", "मेरा ऑर्डर कहाँ है")):
        t = time.perf_counter()
        bot.handle(q)
        results[f"first_reply_{lang}_ms"] = (time.perf_counter() - t) * 1000
    results["mem_after_startup_mb"] = memory_mb()[0]
    return DialogueManager


def question_pool():
    cases = json.load(open(Path(__file__).resolve().parent.parent / "data/eval/test_set.json", encoding="utf-8"))
    return [c["q"] for c in cases["cases"]] + ["10234", "yes", "no", "thanks", "hi", "bye bye now"]


def phase_throughput(DM, results, n):
    pool, bot, lat = question_pool(), DM(ai=None), []
    random.seed(7)
    start = time.perf_counter()
    for _ in range(n):
        q = random.choice(pool)
        t = time.perf_counter()
        bot.handle(q)
        lat.append((time.perf_counter() - t) * 1000)
    total = time.perf_counter() - start
    results.update(throughput_qps=n / total, lat_p50_ms=pct(lat, .5), lat_p95_ms=pct(lat, .95),
                   lat_p99_ms=pct(lat, .99), lat_max_ms=max(lat))


def phase_sessions(DM, results, sessions, turns):
    """A day of customers: each gets a fresh conversation, then leaves."""
    pool = question_pool()
    gc.collect()
    before = memory_mb()[0]
    t = time.perf_counter()
    for _ in range(sessions):
        bot = DM(ai=None)
        for _ in range(turns):
            bot.handle(random.choice(pool))
    results["sessions_per_s"] = sessions / (time.perf_counter() - t)
    gc.collect()
    results["mem_growth_after_sessions_mb"] = memory_mb()[0] - before


def phase_long_session(DM, results, turns):
    """One conversation that never ends (e.g. a kiosk that's never restarted)."""
    pool, bot = question_pool(), DM(ai=None)
    gc.collect()
    before = memory_mb()[0]
    for _ in range(turns):
        bot.handle(random.choice(pool))
    gc.collect()
    results["long_session_turns"] = turns
    results["mem_growth_long_session_mb"] = memory_mb()[0] - before
    results["bytes_per_turn_retained"] = (memory_mb()[0] - before) * 2**20 / turns


def phase_adversarial(DM, results):
    bot = DM(ai=None)
    cases = {
        "empty": "",
        "emoji only": "😡😡😡🙏",
        "10k chars of words": " ".join(random.choice(["refund", "order", "xyzzy", "qwerty", "plz"]) for _ in range(1700)),
        "100k chars one token": "a" * 100_000,
        "10k chars mixed scripts": ("నా ఆర్డర్ मेरा order " * 500)[:10_000],
        "5k unknown words": " ".join(f"zq{i}x" for i in range(5000)),
        "control chars": "\x00\x01\x02 order \x7f refund",
        "SQL-looking": "'; DROP TABLE orders; -- where is my order",
    }
    out = {}
    for name, text in cases.items():
        t = time.perf_counter()
        try:
            turn = bot.handle(text)
            out[name] = f"{(time.perf_counter() - t) * 1000:.0f} ms -> {turn.action}"
        except Exception as e:  # a crash here is a finding, not a test failure
            out[name] = f"CRASH {type(e).__name__}: {e}"
    results["adversarial"] = out


def phase_voice_imports(results):
    """Memory cost of the voice/offline libraries themselves (no models loaded)."""
    out = {}
    for mod in ("numpy", "sounddevice", "speech_recognition", "edge_tts", "pyttsx3", "vosk", "onnxruntime", "piper"):
        before = memory_mb()[0]
        t = time.perf_counter()
        try:
            __import__(mod)
            out[mod] = f"+{memory_mb()[0] - before:.0f} MB, {(time.perf_counter() - t) * 1000:.0f} ms"
        except Exception as e:
            out[mod] = f"not importable here ({type(e).__name__})"
    results["voice_imports"] = out
    results["mem_with_voice_stack_mb"] = memory_mb()[0]


def phase_cap_check(results, mem_mb):
    """Prove the cap is real: allocating past it must fail."""
    try:
        blob = bytearray(int(mem_mb * 1.2) * 2**20)
        del blob
        results["cap_enforced"] = "NO - allocation beyond the cap succeeded"
    except MemoryError:
        results["cap_enforced"] = "yes (allocating 1.2x the cap raised MemoryError)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mem-mb", type=int, default=1024)
    parser.add_argument("--cpu-fraction", type=float, default=None, help="Windows: throttle the core to this share")
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--sessions", type=int, default=1000)
    parser.add_argument("--long-turns", type=int, default=20000)
    parser.add_argument("--json", metavar="PATH")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    results = {"limits": apply_limits(args.mem_mb, args.cpu_fraction)}
    DM = phase_startup(results)
    phase_throughput(DM, results, args.queries)
    phase_sessions(DM, results, args.sessions, turns=6)
    phase_long_session(DM, results, args.long_turns)
    phase_adversarial(DM, results)
    phase_voice_imports(results)
    results["peak_mem_mb"] = memory_mb()[1]
    phase_cap_check(results, args.mem_mb)

    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk:26} {vv}")
        else:
            print(f"{k:32} {v:.2f}" if isinstance(v, float) else f"{k:32} {v}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
