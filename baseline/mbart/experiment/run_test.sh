# evaluate pre-trained/fine-tuned mBART model 
# # for trans_train and trans_test settings 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

langs_27=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,intent,snippet
# langs_25=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

SETTING="trans_test"   # ("trans_train" "trans_test")
LANG="es"               # ("es" "ja" "ru")

SRC="intent"
TGT="snippet"

MBART_DIR=${ROOT_DIR}/"baseline/mbart"
BINARIZED_DATA=${MBART_DIR}/"dataset"/"binarized"
EVAL_DATA=${BINARIZED_DATA}/${SETTING}/${LANG}

SPM_MODEL=${MBART_DIR}/"checkpoint"/"mbart.cc25.v2"/"sentence.bpe.model"
MODEL=${MBART_DIR}/"checkpoint"/"mbart.cc25.v2"/"model.pt"
SAVE_DIR=${MBART_DIR}/"checkpoint"/${SETTING}/${LANG}

PRED_DIR=${MBART_DIR}/"evaluation"/${SETTING}/${LANG}
OUT_DIR=${PRED_DIR}/"output"
HYP_DIR=${PRED_DIR}/"hyp"
REF_DIR=${PRED_DIR}/"ref"

mkdir -p ${PRED_DIR}

fairseq-generate ${EVAL_DATA} \
    --langs ${langs_27} \
    --source-lang ${SRC} --target-lang ${TGT} \
    --path ${SAVE_DIR}/"checkpoint_best.pt" \
    --sentencepiece-model ${SPM_MODEL} \
    --bpe 'sentencepiece' --remove-bpe 'sentencepiece' \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    --sacrebleu --scoring sacrebleu \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 32 \
    > ${OUT_DIR}

cat ${OUT_DIR} | grep -P "^H" |sort -V |cut -f 3- > ${HYP_DIR}
cat ${OUT_DIR} | grep -P "^T" |sort -V |cut -f 2-  > ${REF_DIR}
sacrebleu -w 4 $REF_DIR < $HYP_DIR
