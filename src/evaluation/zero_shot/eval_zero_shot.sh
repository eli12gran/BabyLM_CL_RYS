#!/bin/bash

# NOTE: Replace in strict/scripts
MODEL_PATH=$1
BACKEND=$2
EVAL_DIR=${3:-"evaluation_data/full_eval"}
REVISION=${4:-main}

if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo $BACKEND_READ

python -m evaluation_pipeline.sentence_zero_shot.run_modified --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/blimp_filtered" --save_predictions --revision_name $REVISION
python -m evaluation_pipeline.sentence_zero_shot.run_modified --model_path_or_name $MODEL_PATH --backend $BACKEND --task blimp --data_path "${EVAL_DIR}/supplement_filtered" --save_predictions --revision_name $REVISION
python -m evaluation_pipeline.sentence_zero_shot.run_modified --model_path_or_name $MODEL_PATH --backend $BACKEND --task ewok --data_path "${EVAL_DIR}/ewok_filtered" --save_predictions --revision_name $REVISION
python -m evaluation_pipeline.sentence_zero_shot.run_modified --model_path_or_name $MODEL_PATH --backend $BACKEND --task entity_tracking --data_path "${EVAL_DIR}/entity_tracking" --save_predictions --revision_name $REVISION
python -m evaluation_pipeline.sentence_zero_shot.run_modified --model_path_or_name $MODEL_PATH --backend $BACKEND --task comps --data_path "${EVAL_DIR}/comps" --save_predictions --revision_name $REVISION
python -m evaluation_pipeline.reading.run --model_path_or_name $MODEL_PATH --backend $BACKEND_READ --data_path "${EVAL_DIR}/reading/reading_data.csv" --revision_name $REVISION