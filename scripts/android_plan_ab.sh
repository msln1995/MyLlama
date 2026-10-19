#!/usr/bin/env bash
# On-device A/B harness for the plans the Android tuner produces (dev tool, not
# part of CI): starts a standalone llama-server with the app's own CPU-only arg
# shape, drives a fixed chat-shaped workload through it and samples RSS +
# MemAvailable while it runs, so two cache-ram/context plans can be compared on
# the same phone in the same machine state.
#
#   scripts/android_plan_ab.sh <label> <ctx> <cache-ram> [model] [binary-dir]
#
# The binary dir must be app-executable (SELinux denies exec on external
# storage); /data/local/tmp/llama is a directory the shell can push into and
# execute from. Results land in $OUT (default /tmp/plan-ab).
#
# Single reads are worthless on this class of device — the same arm re-measured
# 446s vs 590s wall clock — so run each arm at least twice and compare medians.
#
# The sampler also records each cluster's cpufreq, because it dominates the
# numbers: this vivo caps all three clusters to ~1.15 GHz whenever the display is
# off (2002/2600/2808 MHz with it on, 1170/1144/1144 with it off, at 31C — a
# screen-state power policy, not heat), which is a 2.45x clock cut on the big
# cores. Arms run with the screen off therefore measure a different machine from
# arms run with it on, and the app's real condition is screen-on. Keep the screen
# awake for comparable arms (adb shell svc power stayon true).

set -u
export PATH="$PATH:$HOME/Library/Android/sdk/platform-tools"

LABEL=${1:?usage: android_plan_ab.sh <label> <ctx> <cache-ram> [model] [bin]}
CTX=${2:?}
CACHERAM=${3:?}
MODEL=${4:-/storage/emulated/0/Download/Qwen3-4B-Q4_K_M.gguf}
BIN=${5:-/data/local/tmp/llama}
OUT=${OUT:-/tmp/plan-ab}
LOG=/data/local/tmp/srv-$LABEL.log
mkdir -p "$OUT"

# Stop the server by pid, not by pkill -f: the pkill pattern matches the invoking
# `adb shell` command line itself, so the remote shell kills itself and the caller
# gets empty output plus a lingering server.
stop_server() { adb shell "p=\$(pidof llama-server); [ -n \"\$p\" ] && kill \$p; true" 2>/dev/null; }

# The app's direct-mode CPU plan for a 4B Q4 on an Exynos 1080 (see
# core/preset.go modelDirectArgs: GPU offload stripped, threads capped at the
# performance cores). Only --ctx-size and --cache-ram vary per arm.
ARGS="--host 127.0.0.1 --port 8080 -m $MODEL --alias Qwen3-4B --ctx-size $CTX \
--batch-size 2048 --ubatch-size 512 --threads 4 --ctx-checkpoints 0 --gpu-layers 0 \
--cont-batching --no-webui --cache-ram $CACHERAM"

stop_server
sleep 3
adb shell "rm -f $LOG"

# Optional core pinning (CPUSET=<hex mask>, f0 = cpu4-7 the 2.6/2.8 GHz cluster,
# 0f = cpu0-3 the 2.0 GHz cluster): the tuner emits --threads equal to the
# performance-core count, but that is a thread count, not a placement promise -
# the scheduler can still land those threads on the little cluster. Pinning is
# how to tell the two effects apart. Android's toybox taskset takes a hex MASK,
# not the util-linux "-c 4-7" list form.
PREFIX=""
if [ -n "${CPUSET:-}" ]; then PREFIX="taskset $CPUSET "; fi

# adb shell never returns while the daemon holds the socket, so launch detached
# locally and poll the server log instead of waiting on the command.
( adb shell "cd $BIN && LD_LIBRARY_PATH=$BIN setsid nohup ${PREFIX}./llama-server $ARGS > $LOG 2>&1 < /dev/null &" \
  </dev/null >/dev/null 2>&1 & )

for _ in $(seq 1 90); do
  adb shell "grep -q 'listening on' $LOG" 2>/dev/null && break
  sleep 2
done
if ! adb shell "grep -q 'listening on' $LOG"; then
  echo "$LABEL FAILED TO START"
  adb shell "tail -20 $LOG"
  exit 1
fi
echo "=== $LABEL ctx=$CTX cache-ram=$CACHERAM started"

# The forward can drop when adbd restarts; re-assert it per arm.
adb forward tcp:18080 tcp:8080 >/dev/null

(
  while true; do
    adb forward tcp:18080 tcp:8080 >/dev/null 2>&1
    adb shell "ps -A -o RSS,NAME | grep llama-server | head -1; grep MemAvailable /proc/meminfo;
      printf 'cur0=%s cur4=%s cur7=%s max4=%s' \$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) \$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq) \$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq) \$(cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_max_freq);
      printf ' temp=%s' \$(dumpsys battery | grep temperature | tr -cd '0-9'); echo" \
      | tr -s ' \n' ' '
    echo
    # One line per tick: "<tid> <utime+stime jiffies> <last-run cpu>;" for every
    # thread. Read together across ticks this gives CPU-time-weighted core
    # placement: a thread's runtime delta lands on the core it reports, so
    # threads that are simply parked contribute ~nothing. (Field 39 of stat is
    # the last CPU it ran on; a single sample of it says nothing, because an
    # idle worker still reports the core it fell asleep on - see the 2026-10-18
    # note in docs/design/autotune.md.)
    adb shell "p=\$(pidof llama-server); [ -n \"\$p\" ] || exit 0;
      for t in /proc/\$p/task/*; do tid=\${t##*/};
        awk -F') ' -v t=\$tid '{split(\$2,a,\" \"); printf t\" \"a[12]+a[13]\" \"a[37]\";\"}' \$t/stat 2>/dev/null;
      done; echo" >> "$OUT/$LABEL-threads.txt"
    sleep 4
  done > "$OUT/$LABEL-sample.txt"
) &
SAMP=$!

sleep 5
# The placement analyzer pairs samples within this run only, so start clean.
rm -f "$OUT/$LABEL-threads.txt"
python3 - "$LABEL" "$OUT/$LABEL-work.txt" <<'PY'
import json, random, re, string, subprocess, sys, time, urllib.error, urllib.request

BASE = "http://127.0.0.1:18080"
label, outfile = sys.argv[1], sys.argv[2]
out = open(outfile, "w", buffering=1)


def probe():
    # Server RSS and battery temperature at each round boundary, so memory and
    # heat can be compared at the same request index. Comparing them at the same
    # wall-clock fraction instead biases the median: a throttled arm simply
    # spends longer in the later, larger-RSS part of the run.
    try:
        p = subprocess.run(["adb", "shell",
                            "grep VmRSS /proc/$(pidof llama-server)/status; "
                            "dumpsys battery | grep temperature; "
                            "cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq "
                            "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    rss = re.search(r"VmRSS:\s+(\d+)", p.stdout)
    temp = re.search(r"temperature:\s*(\d+)", p.stdout)
    if not rss:
        return ""
    bit = f" rss={int(rss.group(1)) // 1024}MiB"
    if temp:
        bit += f" temp={int(temp.group(1)) / 10:.1f}C"
    # The clock the *next* round starts in, which is what the sag actually does
    # to a round; the whole-run median of it hides the collapse (see 2026-10-19).
    big = re.findall(r"^\s*(\d{6,8})\s*$", p.stdout, re.M)
    if big:
        bit += f" cpu7={int(big[0]) // 1000}MHz(max {int(big[1]) // 1000})"
    return bit


def filler(n_words, seed):
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india julia kilo lima mike "
             "novel oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu "
             "system memory kernel tensor layer gradient vector matrix router cache thread").split()
    rnd = random.Random(seed)
    return " ".join(rnd.choice(words) for _ in range(n_words))


def chat(messages, max_tokens):
    # The adb forward can drop mid-request (it closed two arms' first request
    # without a response before this existed); the sampler re-asserts it every
    # tick, so one retry after a gap covers the window. The retry is recorded in
    # the work file so a retried rep is never mistaken for a clean timing read.
    for attempt in range(3):
        payload = {"model": "Qwen3-4B", "messages": messages, "max_tokens": max_tokens,
                   "temperature": 0.0, "stream": False}
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                body = json.loads(r.read())
        except (urllib.error.URLError, OSError) as exc:
            out.write(f"retry {attempt + 1}: {type(exc).__name__}\n")
            time.sleep(8)
            continue
        u = body.get("usage", {})
        return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0
    raise RuntimeError("three consecutive connection failures")


out.write(f"# {label}\n")
# Phase 1: distinct prompts with no shared prefix - worst case for the cache,
# this is where an over-large --cache-ram shows up as RAM pressure.
for i in range(3):
    nonce = "".join(random.choices(string.ascii_lowercase, k=10))
    msgs = [{"role": "user", "content": f"[{nonce}] {filler(900, 1000 + i * 977)}\n\nThree short bullets."}]
    pp, tg, dt = chat(msgs, 128)
    out.write(f"distinct {i}: pp={pp} tg={tg} t={dt:.1f}s{probe()}\n")

# Phase 2: one growing conversation - the same prefix is re-sent every turn,
# which is what a chat app does and what the prompt cache is for.
msgs = [{"role": "user", "content": filler(900, 4242) + "\n\nGive me a one-line summary."}]
for turn in range(4):
    pp, tg, dt = chat(msgs, 64)
    out.write(f"grow {turn}: pp={pp} tg={tg} t={dt:.1f}s{probe()}\n")
    msgs.append({"role": "assistant", "content": "ok " * 40})
    msgs.append({"role": "user", "content": f"turn {turn}: name three details.\n\nfiller: " + filler(120, turn)})
out.write("# done\n")
PY
kill $SAMP 2>/dev/null

sleep 10
stop_server
sleep 3
adb shell "cat $LOG" > "$OUT/$LABEL-srv.log" 2>/dev/null

echo "--- $LABEL RSS samples (MiB) / min avail / cpufreq ceiling / core placement"
python3 - "$OUT/$LABEL-sample.txt" "$OUT/$LABEL-threads.txt" <<'PY'
import os
import re
import sys

rss, avail, cur0, cur4, cur7, max4, temp = [], [], [], [], [], [], []
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) < 4 or not parts[0].isdigit():
        continue
    rss.append(int(parts[0]) // 1024)
    for i, word in enumerate(parts):
        if word.startswith("MemAvailable") and i + 1 < len(parts):
            avail.append(int(parts[i + 1]) // 1024)
    # cpufreq readings, labelled from 2026-10-19 on; before that the same four
    # values were the trailing bare numbers (cur per cluster, then big-cluster
    # max). Temps only exist in the labelled form.
    lab = dict(re.findall(r"(cur0|cur4|cur7|max4|temp)=(\d+)", line))
    if lab:
        for key, sink in (("cur0", cur0), ("cur4", cur4), ("cur7", cur7), ("max4", max4)):
            if key in lab:
                sink.append(int(lab[key]) // 1000)
        if "temp" in lab:
            temp.append(int(lab["temp"]) / 10.0)
    else:
        tail = [int(w) for w in parts[-4:] if w.isdigit() and int(w) > 100000]
        if len(tail) == 4:
            cur0.append(tail[0] // 1000)
            cur4.append(tail[1] // 1000)
            cur7.append(tail[2] // 1000)
            max4.append(tail[3] // 1000)
if rss:
    line = (f"peak_rss_MiB={max(rss)} last_rss_MiB={rss[-1]} "
            f"min_avail_MiB={min(avail) if avail else -1} n={len(rss)}")
    if cur7:
        med = lambda xs: sorted(xs)[len(xs) // 2]
        half = len(cur7) // 2
        line += (f" cpuMHz med(cur0/cur4/cur7)={med(cur0)}/{med(cur4)}/{med(cur7)}"
                 f" cur7 1st/2nd half={med(cur7[:half])}/{med(cur7[half:])}"
                 f" ceiling_big={med(max4)}")
        # The whole-run medians matched exactly for the pair whose decode speeds
        # differed by 1.4x, so the split-half numbers are the ones to compare.
        if temp:
            line += f" temp={min(temp):.1f}-{max(temp):.1f}C med={med(temp):.1f}"
    print(line)

# CPU-time-weighted placement: each thread's runtime delta between two ticks is
# charged to the core it reports on the later tick, so parked threads drop out
# and the result answers "where did the compute actually happen".
if os.path.exists(sys.argv[2]):
    ticks = []
    for line in open(sys.argv[2]):
        cur = {}
        for item in line.strip().split(";"):
            f = item.split()
            if len(f) == 3 and f[1].isdigit() and f[2].isdigit():
                cur[f[0]] = (int(f[1]), int(f[2]))
        if cur:
            ticks.append(cur)
    core_ms = {}
    for prev, nxt in zip(ticks, ticks[1:]):
        for tid, (run, core) in nxt.items():
            if tid in prev:
                d = run - prev[tid][0]
                if 0 < d < 500:  # jiffies per tick; >=500 would be >5 cores busy
                    core_ms[core] = core_ms.get(core, 0) + d
    total = sum(core_ms.values())
    if total:
        share = sorted(((c, 100.0 * ms / total) for c, ms in core_ms.items()),
                       key=lambda x: -x[1])
        big = sum(p for c, p in share if int(c) >= 4)
        print("cores(jiffies-weighted): " + " ".join(f"cpu{c}={p:.0f}%" for c, p in share)
              + f" | perf_cluster={big:.0f}% ticks={len(ticks)}")
PY
grep -E "prompt eval time|eval time =|graphs reused|slot.*n_ctx|model size|KV.*buffer" \
  "$OUT/$LABEL-srv.log" | tail -30
echo
