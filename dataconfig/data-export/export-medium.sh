# TO BE RUN in src folder, with bash <this script>

# MEDIUM 2K
EXP_ID=data-medium-2k-no-rf  # this is a left-over due to batch processing, the dataset is medium!!!
INPUT_DIR=/scratch/milp/out/medium/dataset-medium-2k-no-rf/dataset-medium-2k-no-rf-e32-l2
OUTPUT_DIR=/scratch/milp/exported/medium
mkdir -p $OUTPUT_DIR

python g4satb_export.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID


