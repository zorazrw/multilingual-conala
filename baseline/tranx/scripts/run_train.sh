
set -e

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi


LANG="es"
SETTING="trans-test"
OUT_DIR=${ROOT_DIR}/"data"/${SETTING}/${LANG}

seed=0
mined_num=$1
ret_method=$2
freq=3

vocab=${OUT_DIR}/"vocab.src_freq${freq}.code_freq${freq}.mined_${mined_num}.goldmine_${ret_method}.bin"
train_file=${OUT_DIR}/"pre_${mined_num}_goldmine_${ret_method}.bin"
dev_file=${OUT_DIR}/"dev.bin"

dropout=0.3
hidden_size=256
embed_size=128
action_embed_size=128
field_embed_size=64
type_embed_size=64
lr=0.001
lr_decay=0.5
batch_size=64
max_epoch=80
beam_size=15
lstm='lstm'  # lstm
lr_decay_after_epoch=15
model_name=retdistsmpl.dr${dropout}.lr${lr}.lr_de${lr_decay}.lr_da${lr_decay_after_epoch}.beam${beam_size}.$(basename ${vocab}).$(basename ${train_file}).seed${seed}

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
    --batch_size ${batch_size} \
    --evaluator conala_evaluator \
    --asdl_file asdl/lang/py3/py3_asdl.simplified.txt \
    --transition_system python3 \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
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
    --max_epoch ${max_epoch} \
    --beam_size ${beam_size} \
    --log_every 50 \
    --save_to ${SAVE_DIR} 2>&1 | tee ${LOG_PATH}

. test_mconala.sh 2>&1 | tee -a ${LOG_PATH}
