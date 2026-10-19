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

# The app's direct-mode CPU plan for a 4B Q4 on an Exynos 1080 (see
# core/preset.go modelDirectArgs: GPU offload stripped, threads capped at the
# performance cores). Only --ctx-size and --cache-ram vary per arm.
ARGS="--host 127.0.0.1 --port 8080 -m $MODEL --alias Qwen3-4B --ctx-size $CTX \
--batch-size 2048 --ubatch-size 512 --threads 4 --ctx-checkpoints 0 --gpu-layers 0 \
--cont-batching --no-webui --cache-ram $CACHERAM"

adb shell "pkill -f llama-server" 2>/dev/null
sleep 3
adb shell "rm -f $LOG"

# adb shell never returns while the daemon holds the socket, so launch detached
# locally and poll the server log instead of waiting on the command.
( adb shell "cd $BIN && LD_LIBRARY_PATH=$BIN setsid nohup ./llama-server $ARGS > $LOG 2>&1 < /dev/null &" \
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
    adb shell "ps -A -o RSS,NAME | grep llama-server | head -1; grep MemAvailable /proc/meminfo" \
      | tr -s ' \n' ' '
    echo
    sleep 4
  done > "$OUT/$LABEL-sample.txt"
) &
SAMP=$!

sleep 5
python3 - "$LABEL" "$OUT/$LABEL-work.txt" <<'PY'
import json, random, string, sys, time, urllib.request

BASE = "http://127.0.0.1:18080"
label, outfile = sys.argv[1], sys.argv[2]
out = open(outfile, "w", buffering=1)


def filler(n_words, seed):
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india julia kilo lima mike "
             "novel oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu "
             "system memory kernel tensor layer gradient vector matrix router cache thread").split()
    rnd = random.Random(seed)
    return " ".join(rnd.choice(words) for _ in range(n_words))


def chat(messages, max_tokens):
    payload = {"model": "Qwen3-4B", "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        body = json.loads(r.read())
    u = body.get("usage", {})
    return u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0


out.write(f"# {label}\n")
# Phase 1: distinct prompts with no shared prefix - worst case for the cache,
# this is where an over-large --cache-ram shows up as RAM pressure.
for i in range(3):
    nonce = "".join(random.choices(string.ascii_lowercase, k=10))
    msgs = [{"role": "user", "content": f"[{nonce}] {filler(900, 1000 + i * 977)}\n\nThree short bullets."}]
    pp, tg, dt = chat(msgs, 128)
    out.write(f"distinct {i}: pp={pp} tg={tg} t={dt:.1f}s\n")

# Phase 2: one growing conversation - the same prefix is re-sent every turn,
# which is what a chat app does and what the prompt cache is for.
msgs = [{"role": "user", "content": filler(900, 4242) + "\n\nGive me a one-line summary."}]
for turn in range(4):
    pp, tg, dt = chat(msgs, 64)
    out.write(f"grow {turn}: pp={pp} tg={tg} t={dt:.1f}s\n")
    msgs.append({"role": "assistant", "content": "ok " * 40})
    msgs.append({"role": "user", "content": f"turn {turn}: name three details.\n\nfiller: " + filler(120, turn)})
out.write("# done\n")
PY
kill $SAMP 2>/dev/null

sleep 10
adb shell "pkill -f llama-server" 2>/dev/null
sleep 3
adb shell "cat $LOG" > "$OUT/$LABEL-srv.log" 2>/dev/null

echo "--- $LABEL RSS samples (MiB) / min avail"
python3 - "$OUT/$LABEL-sample.txt" <<'PY'
import sys

rss, avail = [], []
for line in open(sys.argv[1]):
    parts = line.split()
    if len(parts) >= 4 and parts[0].isdigit():
        rss.append(int(parts[0]) // 1024)
        for i, word in enumerate(parts):
            if word.startswith("MemAvailable"):
                avail.append(int(parts[i + 1]) // 1024)
if rss:
    print(f"peak_rss_MiB={max(rss)} last_rss_MiB={rss[-1]} "
          f"min_avail_MiB={min(avail) if avail else -1} n={len(rss)}")
PY
grep -E "prompt eval time|eval time =|graphs reused|slot.*n_ctx|model size|KV.*buffer" \
  "$OUT/$LABEL-srv.log" | tail -30
echo
