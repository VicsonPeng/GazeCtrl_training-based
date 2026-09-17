#!/bin/bash
# Generate front2back (back) + look_around (look) for each sample, distributed
# across free GPUs in waves. Writes DONE_DATASET when all jobs finish.
cd ${GAZE_WAN_DIR:-${GAZE_ROOT:-/project/vicson_gaze}/Wan2.2}
C=${GAZE_RAW:-${GAZE_ROOT:-/project/vicson_gaze}/condition_dataset_v2}/crops
GPUS=(0 1 2 4 5 6 7 8 9)
SAMPLES=(035313 036484 037614 038829 032398 039707 032094 033945 037389 034136)

# build job list: "sample mode"
JOBS=()
for s in "${SAMPLES[@]}"; do
  JOBS+=("shhq_image_${s}__p0 back")
  JOBS+=("shhq_image_${s}__p0 look")
done

n=${#JOBS[@]}; g=${#GPUS[@]}; i=0
while [ $i -lt $n ]; do
  pids=()
  for ((k=0; k<g && i<n; k++, i++)); do
    set -- ${JOBS[$i]}; sid=$1; mode=$2; gpu=${GPUS[$k]}
    nohup bash gen_motion.sh $gpu "$sid" "$C/$sid.jpg" "$mode" \
      > "turnaround_out/_gen_${sid}_${mode}.log" 2>&1 &
    pids+=($!)
    echo "wave: GPU$gpu <- $sid $mode (pid $!)"
  done
  echo "waiting on ${#pids[@]} jobs ..."
  wait "${pids[@]}"
done
echo "DONE_DATASET"
