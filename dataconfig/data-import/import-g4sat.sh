RENAME=yes
# APPEND_SAT=yes

# EASY
set -x
INPUT_DIR=/scratch/milp/g4sat-dataset-ecai/easy/3-sat
OUTPUT_DIR=/scratch/milp/imported/easy/

python g4satb_import.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --rename $RENAME \
#    --append_sat $APPEND_SAT


# EASY 20K
INPUT_DIR=/scratch/milp/g4sat-dataset-arxiv/easy-20k/3-sat
OUTPUT_DIR=/scratch/milp/imported/easy-20k/
python g4satb_import.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --rename $RENAME

# MEDIUM
INPUT_DIR=/scratch/milp/g4sat-dataset-arxiv/medium/3-sat
OUTPUT_DIR=/scratch/milp/imported/medium/
python g4satb_import.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --rename $RENAME

# MEDIUM 20K
INPUT_DIR=/scratch/milp/g4sat-dataset-arxiv/medium-20k/3-sat
OUTPUT_DIR=/scratch/milp/imported/medium-20k/
python g4satb_import.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --rename $RENAME