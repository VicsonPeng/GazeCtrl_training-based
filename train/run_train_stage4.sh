#!/bin/bash
# STAGE 4 — add a perceptual term in PIXEL space, and change nothing else.
# Stage 3 tried to fix the facial artifacts by rebalancing the latent-MSE weights and shifting
# the timestep distribution; it made both control and face quality worse. What that bought is
# the knowledge that latent MSE cannot express naturalness at any weighting, so the remedy has
# to live in pixel space. Here the ONLY change against stage 2 is:
#     loss += w(sigma) * LPIPS( decode(clean_hat)[head] , target[head] )
#     w(sigma) = 1.0 * max(0, 1 - sigma/0.5)
# head_boost, eye_boost, the uniform sigma schedule, the data mix, the crop and the lr are all
# held at the stage-2 values, so the attribution is clean whichever way this goes.
# The sigma ramp is mandatory: the term's measured gradient norm RISES with sigma (0.11 -> 0.33)
# because the one-step estimate of the clean image is meaningless at high noise.
ROOT=${GAZE_ROOT:-/project/vicson_gaze}; WORK=${GAZE_WORK:-/home/vicson/gaze_diag}
PY=${GAZE_PYTHON:-$ROOT/miniconda3/envs/gaze_net_train_env/bin/python}
CODE=${GAZE_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=$WORK/_cn_stage4
INIT=$WORK/_cn_stage2/cn_step040000.safetensors
CSV=$WORK/_cn_50id/50id_stage1.csv
BOXES=${GAZE_DATASET:-$ROOT/gaze_dataset}/head_boxes_stage1.json
CSV2=$WORK/_cn_hitl/hitl.csv
BOXES2=$WORK/_cn_hitl/head_boxes.json
EYE2=$WORK/_cn_hitl/eye_boxes_used.json
STEPS=12000
mkdir -p $OUT; cd $CODE/train
for attempt in $(seq 1 60); do
  latest=$(ls $OUT/cn_step0*.safetensors 2>/dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    st=$(basename "$latest" | grep -oE '[0-9]+'); resume_args="--resume $latest --start_step $st"
  else
    st=0; resume_args="--resume $INIT --start_step 0"
  fi
  [ "$st" -ge "$STEPS" ] && { echo "REACHED_$STEPS"; break; }
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
  CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -u train_stage4.py \
    --csv $CSV --boxes $BOXES --csv2 $CSV2 --boxes2 $BOXES2 --eye_boxes2 $EYE2 --mix2 0.7 \
    --crop_p 0.65 --crop_k 2.0 --crop_minfrac 0.45 --crop_margin 0.15 \
    --out $OUT --cond rgb --fixed_src 0 \
    $resume_args --steps $STEPS --lr 1e-4 --head_boost 8.0 --eye_boost 30.0 --gaze_drop 0.0 \
    --lpips_w 1.0 --lpips_sigma_max 0.5 \
    --cn_scale 1.0 --save_every 2000 --eval_every 2000 --gpu 0
  echo "--- exited attempt $attempt (step was $st) ---"; sleep 10
done
echo "STAGE4_TRAIN_DONE"
