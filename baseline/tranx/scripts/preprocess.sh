# preprocess-train-test pipeline of trans-test setting  

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi


declare -a lang_list=("es" "ja" "ru")

TRAIN_DATA=${ROOT_DIR}/"dataset"/"train"
TEST_DATA=${ROOT_DIR}/"dataset"/"test"
MT="flores101"     # ("flores101" "marianmt" "m2m")

TRANX_DIR="baseline/tranx/external-knowledge-codegen"

cd ${ROOT_DIR}/${TRANX_DIR}



# trans-train 
mkdir -p "data/trans_train"

for lang in ${lang_list[@]}; do
  mkdir -p "data/trans_train"/${lang} 

  python -m datasets.conala.dataset \
    --train_file ${TRAIN_DATA}/"to_"${lang}/"train_to_"${lang}".json" \
    --test_file ${TEST_DATA}/${lang}"_test.json" \
    --pretrain ${TRAIN_DATA}/"to_"${lang}/"mined_to_"${lang}".jsonl" \
    --topk 100000 \
    --include_api ${TRAIN_DATA}/"to_"${lang}/"api_to_"${lang}".jsonl" \
    --out_dir "data/trans_train"/${lang}
done 



# trans-test 
mkdir -p "data/trans_test"

for lang in ${lang_list[@]}; do
  mkdir -p "data/trans_test"/${lang}

  python -m datasets.conala.dataset \
    --train_file ${TRAIN_DATA}/"train.json" \
    --test_file ${TEST_DATA}/${MT}/${lang}"_test_to_en.json" \
    --pretrain ${TRAIN_DATA}/"mined.jsonl" \
    --topk 100000 \
    --include_api ${TRAIN_DATA}/"api.jsonl" \
    --out_dir "data/trans_test"/${lang}
done 
