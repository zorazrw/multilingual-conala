set -e

LANG="es"
SETTING="trans-test"

seed=0
mined_num=$1
ret_method=$2
pretrained_model_name=$3

freq=3

OUT_DIR="data"/${SETTING}/${LANG}
vocab=${OUT_DIR}/"vocab.src_freq${freq}.code_freq${freq}.mined_${mined_num}.goldmine_${ret_method}.bin"
finetune_file=${OUT_DIR}/"train.var_str_sep.bin"
dev_file=${OUT_DIR}/"dev.bin"

dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=finetune.mined.retapi.distsmpl.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.seed${seed}.mined_${mined_num}.${ret_method}

SAVE_DIR=saved_models/${LANG}"-conala_"${SETTING}/${model_name}

LOG_DIR="logs"/${LANG}"-conala_"${SETTING}
LOG_PATH=${LOG_DIR}/${model_name}".log"
echo "**** Writing results to logs/es-conala-trans-test/${model_name}.log ****"

mkdir -p ${LOG_DIR}
echo commit hash: `git rev-parse HEAD` > ${LOG_PATH}

python -u exp.py \
    --cuda \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${finetune_file} \
    --dev_file ${dev_file} \
    --pretrain ${pretrained_model_name} \
    --vocab ${vocab} \
    --lstm ${lstm} \
    --no_parent_field_type_embed \
    --no_parent_production_embed \
    --hidden_size ${hidden_size} \
    --embed_size ${embed_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --patience 5 \
    --max_num_trial 5 \
    --glorot_init \
    --lr ${lr} \
    --lr_decay ${lr_decay} \
    --lr_decay_after_epoch ${lr_decay_after_epoch} \
    --max_epoch 80 \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to ${SAVE_DIR} 2>&1 | tee ${LOG_PATH}

. test_mconala.sh ${SAVE_DIR}".bin" 2>&1 | tee -a ${LOG_PATH}
