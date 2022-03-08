# extract intent & snippet lines from all dataset files 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

SCRIPT=${ROOT_DIR}/"baseline/mbart/preprocess"
SRC_DATA="dataset" 
TGT_DATA="baseline/mbart/dataset/lines"

TRAIN_DIR="train"
TEST_DIR="test"

cd $SCRIPT



# train sets 
mkdir -p $ROOT_DIR/$TGT_DATA/$TRAIN_DIR

python extract_lines.py \
--input_dir $ROOT_DIR/$SRC_DATA/$TRAIN_DIR \
--output_dir $ROOT_DIR/$TGT_DATA/$TRAIN_DIR


declare -a lang_list=("es" "ja" "ru")
for lang in ${lang_list[@]}; do
    mkdir -p $ROOT_DIR/$TGT_DATA/$TRAIN_DIR/"to_"${lang}

    python extract_lines.py \
    --input_dir $ROOT_DIR/$SRC_DATA/$TRAIN_DIR/"to_"${lang} \
    --output_dir $ROOT_DIR/$TGT_DATA/$TRAIN_DIR/"to_"${lang}
done 



# test sets 
mkdir -p $ROOT_DIR/$TGT_DATA/$TEST_DIR

python extract_lines.py \
--input_dir $ROOT_DIR/$SRC_DATA/$TEST_DIR \
--output_dir $ROOT_DIR/$TGT_DATA/$TEST_DIR \
--split_dev_test 


declare -a mt_list=("flores101" "marianmt" "m2m")
for mt in ${mt_list[@]}; do
    mkdir -p $ROOT_DIR/$TGT_DATA/$TEST_DIR/${mt}

    python extract_lines.py \
    --input_dir $ROOT_DIR/$SRC_DATA/$TEST_DIR/${mt} \
    --output_dir $ROOT_DIR/$TGT_DATA/$TEST_DIR/${mt} \
    --split_dev_test 
done 
