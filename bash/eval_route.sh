

export $(grep -v '^#' .env | xargs)

python script/evaluation/baseline_routellm_getscore.py \
      --dataset-name olympiadbench \
      --out output/evaluation/routellm \
      --router mf \

DATASET_NAME=olympiadbench
STRONG_SIZE=4.0
WEAK_SIZE=0.6
TARGET_AVG=1.7

python ./script/evaluation/compute_routellm_threshold_accuracy.py \
    --scores output/evaluation/routellm/mf/${DATASET_NAME}.csv \
    --strong-detailed /path/to/strong/detailed_results.csv \
    --weak-detailed /path/to/weak/detailed_results.csv \
    --strong-size ${STRONG_SIZE} --weak-size ${WEAK_SIZE} --target-avg ${TARGET_AVG} --rounding round\
    --save-csv output/evaluation/routellm/thresholded_${STRONG_SIZE}_${WEAK_SIZE}_${TARGET_AVG}_${DATASET_NAME}_detail.csv