#!/bin/bash
# CURRICULUM STAGE 1 — SHHQ full-body Wan-turnaround pairs with a head-preserving random
# crop, mixed 80/20 with HITL pairs. From the InstantX pretrained ControlNet, NOT from the
# HITL 90k run: the point is to learn the headpose<->fullbody relation before stage 2
# sharpens eye gaze on the human-labelled close-up data.
# Waits for a free GPU (mem<1000MiB, confirmed twice), auto-resumes from the latest ckpt.
ROOT=${GAZE_ROOT:-/project/vicson_gaze}; WORK=${GAZE_WORK:-/home/vicson/gaze_diag}
PY=${GAZE_PYTHON:-$ROOT/miniconda3/envs/gaze_net_train_env/bin/python}
CODE=${GAZE_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=$WORK/_cn_stage1
CSV=$WORK/_cn_50id/50id_stage1.csv
BOXES=${GAZE_DATASET:-$ROOT/gaze_dataset}/head_boxes_stage1.json
CSV2=$WORK/_cn_hitl/hitl.csv
BOXES2=$WORK/_cn_hitl/head_boxes.json
STEPS=35000
mkdir -p $OUT; cd $CODE/train
for attempt in $(seq 1 60); do
  latest=$(ls $OUT/cn_step0*.safetensors 2>/dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    st=$(basename "$latest" | grep -oE '[0-9]+'); resume_args="--resume $latest --start_step $st"
  else
    st=0; resume_args=""
  fi
  [ "$st" -ge "$STEPS" ] && { echo "REACHED_$STEPS"; break; }
  # refuse to start an attempt that cannot store its next checkpoint (3.3G each)
  free=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
  [ "$free" -lt 12 ] && { echo "ABORT: only ${free}G free on /home"; break; }
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
  echo "=== attempt $attempt: start@$st GPU$G free=${free}G $(date) ==="
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u train_stage1.py \
    --csv $CSV --boxes $BOXES --csv2 $CSV2 --boxes2 $BOXES2 --mix2 0.2 \
    --crop_p 0.65 --crop_k 2.0 --crop_minfrac 0.45 --crop_margin 0.15 \
    --out $OUT --cond rgb --fixed_src 0 \
    $resume_args --steps $STEPS --lr 1e-4 --head_boost 8.0 --eye_boost 0.0 --gaze_drop 0.0 \
    --cn_scale 1.0 --save_every 5000 --eval_every 2500 --gpu 0
  echo "--- exited attempt $attempt (step was $st) ---"; sleep 10
done
echo "STAGE1_TRAIN_DONE"
