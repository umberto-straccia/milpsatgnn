# INPUT_DIR=../../G4SATBench/dataset/easy/3-sat
INPUT_DIR=../output/exps/dev/export_to_g4sat
OUTPUT_DIR=~/workspace/G4SATBench/exported-dataset
EXP_ID=0

python g4satb_export.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --exp_id $EXP_ID


