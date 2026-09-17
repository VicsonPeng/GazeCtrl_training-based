#!/bin/bash
# From-scratch RGB-condition ControlNet on HITL-only data (183 ids, 1864 frames).
# Waits for a free GPU (mem<1000MiB), auto-resumes from latest checkpoint if killed.
ROOT=${GAZE_ROOT:-/project/vicson_gaze}; WORK=${GAZE_WORK:-/home/vicson/gaze_diag}
PY=${GAZE_PYTHON:-$ROOT/miniconda3/envs/gaze_net_train_env/bin/python}
CODE=${GAZE_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=$WORK/_cn_hitl
CSV=$OUT/hitl.csv
BOXES=$OUT/head_boxes.json
EYE_BOXES=$OUT/eye_boxes_used.json
EYE_BOOST=30.0
STEPS=90000
mkdir -p $OUT; cd $CODE/train
for attempt in $(seq 1 60); do
  latest=$(ls $OUT/cn_step0*.safetensors 2>/dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    st=$(basename "$latest" | grep -oE '[0-9]+'); resume_args="--resume $latest --start_step $st"
  else
    st=0; resume_args=""
  fi
  [ "$st" -ge "$STEPS" ] && { echo "REACHED_$STEPS"; break; }
  # pick a free GPU (mem<1000MiB) — require it to be free on TWO polls 8s apart
  # to avoid grabbing a card another job is simultaneously claiming (race → OOM).
  G=""; while [ -z "$G" ]; do
    for i in 0 1 2 3 4 5 6 7 8 9; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null)
      [ -n "$m" ] && [ "$m" -lt 1000 ] || continue
      sleep 8
      m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2>/dev/null)
      [ -n "$m2" ] && [ "$m2" -lt 1000 ] && { G=$i; break; }
    done
    [ -z "$G" ] && { echo "$(date +%H:%M:%S) no free GPU, waiting..."; sleep 60; }
  done
  echo "=== attempt $attempt: start@$st GPU$G (confirmed free twice) ==="
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u train_gaze_controlnet.py \
    --csv $CSV --boxes $BOXES --eye_boxes $EYE_BOXES --eye_boost $EYE_BOOST --out $OUT --cond rgb --fixed_src 0 \
    $resume_args --steps $STEPS --lr 1e-4 --head_boost 8.0 --gaze_drop 0.0 \
    --cn_scale 1.0 --save_every 5000 --eval_every 2500 --gpu 0
  echo "--- exited attempt $attempt (step was $st) ---"; sleep 10
done
echo "HITL_TRAIN_DONE"
