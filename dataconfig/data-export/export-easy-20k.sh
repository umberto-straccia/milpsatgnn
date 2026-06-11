# TO BE RUN in src folder, with bash <this script>

# EASY 20K
EXP_ID=data-easy-20k-8
INPUT_DIR=/scratch/milp/out/easy/dataset-easy-20k-1-rf/dataset-easy-20k-1-rf-e128-l2
OUTPUT_DIR=/scratch/milp/exported/easy-20k
mkdir -p $OUTPUT_DIR

python g4satb_export.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --exp_id $EXP_ID


