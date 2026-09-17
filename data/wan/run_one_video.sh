#!/bin/bash
# End-to-end for ONE (sample, mode): generate -> 6DRepNet -> head-crop combine
# -> sample 25 diverse frames -> extract (512x768) -> DELETE video+viz+intermediate.
# Resumable: skips if the per-video label already exists.
# Usage: run_one_video.sh <GPU> <sample_id> <mode>
set -e
GPU="$1"; SID="$2"; MODE="$3"
ROOT=${GAZE_ROOT:-/project/vicson_gaze}
O=$ROOT/Wan2.2/turnaround_out
C=$ROOT/condition_dataset_v2/crops
DST=$ROOT/gaze_dataset
FX=$ROOT/miniconda3/envs/flux_gaze_env/bin/python
GZ=$ROOT/miniconda3/envs/3DGazeNet/bin/python
tag=${SID}__${MODE}             # file stem, e.g. shhq_image_035313__p0__look
lbltag=${SID}_${MODE}           # label tag uses single underscore before mode
mkdir -p $O $DST/labels

if [ -f "$DST/labels/${lbltag}.csv" ]; then echo "SKIP $tag (already done)"; exit 0; fi

# 1) generate
bash $ROOT/Wan2.2/gen_motion.sh $GPU "$SID" "$C/$SID.jpg" "$MODE" > "$O/_w_${tag}.log" 2>&1
V=$O/${tag}.mp4
[ -s "$V" ] || { echo "GEN_FAIL $tag"; exit 1; }

# 2) 6DRepNet head pose
( cd $ROOT/6DRepNet/sixdrepnet && CUDA_VISIBLE_DEVICES=$GPU $FX run_video_yolohead.py \
    --video "$V" --snapshot "6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth" --gpu 0 --out_dir "$O" ) >> "$O/_w_${tag}.log" 2>&1

# 3) head-crop combined viz
( cd $ROOT/3DGazeNet/demo && CUDA_VISIBLE_DEVICES=$GPU $GZ headcrop_combine_viz.py \
    --cfg configs/infer_res18_x128_all_vfhq_vert.yaml --video "$V" \
    --head_txt "$O/${tag}_headpose_v2.txt" --out "$O/viz_hc_${lbltag}.mp4" --gpu_id 0 ) >> "$O/_w_${tag}.log" 2>&1

# 4) sample 25 diverse frames
$FX $ROOT/sample_frames.py --viz "$O/viz_hc_${lbltag}.mp4" --vectors "$O/viz_hc_${lbltag}_vectors.txt" \
    --out_prefix "$O/_sample/${lbltag}" --n_candidates 25 --cols 5 >> "$O/_w_${tag}.log" 2>&1

# 5) extract frames (512x768) + per-video label
$FX $ROOT/extract_one.py --clean "$V" --viz "$O/viz_hc_${lbltag}.mp4" \
    --cands "$O/_sample/${lbltag}_cands.txt" --tag "$lbltag" --dst "$DST" >> "$O/_w_${tag}.log" 2>&1

# 6) cleanup big intermediates (keep only extracted frames + label)
rm -f "$V" "$O/viz_hc_${lbltag}.mp4" "$O/${tag}_headpose_v2.mp4" "$O/${tag}_headpose_v2.txt" \
      "$O/viz_hc_${lbltag}_vectors.txt" "$O/_sample/${lbltag}_cands.png" "$O/_sample/${lbltag}_cands.txt" \
      "$O/_w_${tag}.log" 2>/dev/null

echo "DONE_VIDEO $lbltag"
