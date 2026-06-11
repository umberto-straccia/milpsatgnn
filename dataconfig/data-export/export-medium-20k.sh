# TO BE RUN in src folder, with bash <this script>

# MEDIUM 20K
EXP_ID=data-easy-2k-8  # this is a left-over due to batch processing, the dataset is medium!!!
INPUT_DIR=/scratch/milp/out/medium/dataset-medium-20k-0.5-rf/dataset-medium-20k-0.5-rf-e32-l2
OUTPUT_DIR=/scratch/milp/exported/medium-20k
mkdir -p $OUTPUT_DIR

python g4satb_export.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID


