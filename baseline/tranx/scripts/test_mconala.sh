# test TranX model

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi


LANG="es"
SETTING="trans_test"     # ("trans_train" "trans_test")
DATA_DIR="data"/${SETTING}/${LANG}

echo "evaluation on lang: "${LANG}", setting: "${SETTING}

TRANX_DIR="baseline/tranx/external-knowledge-codegen"
test_file=${DATA_DIR}/"test.bin"

MODEL_NAME="mconala"/${LANG}"_"${SETTING}".bin"
# MODEL_NAME="finetune.mined.retapi.distsmpl.dr0.3.lr0.001.lr_de0.5.lr_da15.beam15.seed0.mined_100000.intent_count100k_topk1_temp5.bin"
MODEL_PATH=${ROOT_DIR}/${TRANX_DIR}/"best_pretrained_models"/${MODEL_NAME}


DECODE_DIR="decodes"/${SETTING}/${LANG}
mkdir -p ${DECODE_DIR}

cd ${ROOT_DIR}/${TRANX_DIR}

python exp.py \
    --cuda \
    --mode test \
    --load_model ${MODEL_PATH} \
    --beam_size 15 \
    --test_file ${ROOT_DIR}/${TRANX_DIR}/${test_file} \
    --evaluator conala_evaluator \
    --save_decode_to ${DECODE_DIR}/${LANG}"_"${SETTING}".test.decode" \
    --decode_max_time_step 100
