# TO BE RUN in src folder, with bash <this script>

# EASY 2K
EXP_ID=data-easy-2k-common
INPUT_DIR=/scratch/milp/out/easy/dataset-easy-2k-05-rf-TAB/dataset-easy-2k-05-rf-e64-l2
OUTPUT_DIR=/scratch/milp/exported/easy
mkdir -p $OUTPUT_DIR

python g4satb_export.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID


