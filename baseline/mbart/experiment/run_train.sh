#!/bin/sh

# train (tune) the mBART model on specified setting 
# # no aditional training for the zero_shot setting 

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
TRAIN_DATA=${BINARIZED_DATA}/${SETTING}/${LANG}

MODEL=${MBART_DIR}/"checkpoint"/"mbart.cc25.v2"/"model.pt"
SAVE_DIR=${MBART_DIR}/"checkpoint"/${SETTING}/${LANG}

fairseq-train ${TRAIN_DATA} \
    --langs ${langs_27} \
    --source-lang ${SRC} --target-lang ${TGT} \
    --restore-file ${MODEL} \
    --save-dir ${SAVE_DIR} \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_from_pretrained_bart \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay \
    --warmup-updates 2500 --total-num-update 40000 \
    --lr 3e-05 --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 512 --update-freq 2 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 
    --no-epoch-checkpoints --seed 222 --log-format simple --log-interval 2 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --ddp-backend c10d --max-update 50000 \
    --batch-size 16  --batch-size-valid 2  --max-epoch 10
