#!/bin/bash
# FINAL DEMO CANDIDATES. 25 sources x 12 yaw x 5 pitch x 2 checkpoints, 1024x1024 PNG.
#
# Card selection (Vicson, 2026-09-25): share cards, just do not cause anyone an OOM.
# Gate on FREE memory (measured peak 27.05 GiB at 1024 -> require 34 GiB, ~7 GiB headroom),
# confirmed on two polls 10s apart.
#
# CLAIM LOCK: free-memory polling alone is not enough between sibling shards. A shard needs
# several minutes to load the 20GB backbone, during which its card still LOOKS free, so all
# three shards picked GPU3 on the first attempt and would have OOMed each other. Each shard
# now claims a card with `mkdir` (atomic) before using it, releases on exit, and treats a
# claim whose pid is dead as stale. Shards also start scanning at different offsets.
ROOT=${GAZE_ROOT:-/project/vicson_gaze}; WORK=${GAZE_WORK:-/home/vicson/gaze_diag}
PY=${GAZE_PYTHON:-$ROOT/miniconda3/envs/gaze_net_train_env/bin/python}
CODE=${GAZE_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
O=${GAZE_OUT:-$ROOT/outputs}/demo_final; mkdir -p $O
CLAIM=$WORK/.gpu_claims; mkdir -p $CLAIM
CKS="stage2_40k=$WORK/_cn_stage2/cn_step040000.safetensors,stage4_08k=$WORK/_cn_stage4/cn_step008000.safetensors"
SR=${GAZE_OUT:-$ROOT/outputs}/demo_srcs.json
NEED=34816
SHARD=$1; NSHARD=$2; LAST=""; MYGPU=""
freemib () { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $1 2>/dev/null \
             | awk -F', ' '{print $2-$1}'; }
release () { [ -n "$MYGPU" ] && rm -rf "$CLAIM/$MYGPU" 2>/dev/null; MYGPU=""; }
trap 'release; exit 130' INT TERM
claim () {   # $1 = gpu index; 0 on success
  local d="$CLAIM/$1"
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; return 0; fi
  local op; op=$(cat "$d/pid" 2>/dev/null)
  if [ -n "$op" ] && ! kill -0 "$op" 2>/dev/null; then      # stale: owner is gone
    rm -rf "$d" 2>/dev/null
    if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; return 0; fi
  fi
  return 1
}
for attempt in $(seq 1 80); do
  MYGPU=""
  while [ -z "$MYGPU" ]; do
    for k in 0 1 2 3 4 5 6 7 8 9; do
      i=$(( (k + SHARD*3) % 10 ))                            # each shard scans from its own offset
      [ "$i" = "$LAST" ] && continue
      f=$(freemib $i); [ -n "$f" ] && [ "$f" -ge "$NEED" ] || continue
      claim $i || continue
      sleep 10
      f2=$(freemib $i)
      if [ -n "$f2" ] && [ "$f2" -ge "$NEED" ]; then MYGPU=$i; break; fi
      rm -rf "$CLAIM/$i" 2>/dev/null                         # someone took it in the meantime
    done
    [ -z "$MYGPU" ] && { echo "$(date +%H:%M:%S) shard$SHARD attempt$attempt: no unclaimed card with ${NEED}MiB free"; sleep 150; LAST=""; }
  done
  echo "=== shard $SHARD attempt $attempt on GPU$MYGPU (free $(freemib $MYGPU) MiB) $(date) ==="
  cd $CODE/eval
  CUDA_VISIBLE_DEVICES=$MYGPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u demo_sweep.py \
    --ckpts "$CKS" --srcs $SR --outdir $O --size 1024 --cn_scale 1.0 --seed 0 \
    --shard $SHARD --nshard $NSHARD \
    && { echo "SHARD${SHARD}_DONE"; release; exit 0; }
  echo "--- shard $SHARD attempt $attempt failed on GPU$MYGPU ---"; LAST=$MYGPU; release; sleep 60
done
release; echo "SHARD${SHARD}_GAVE_UP"
