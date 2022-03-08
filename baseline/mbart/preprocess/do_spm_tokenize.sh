# (spm) tokenization of all dataset files 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

MBART="baseline/mbart"
LINES_DATA=${MBART}/"dataset"/"lines"
TOKENIZED_DATA=${MBART}/"dataset"/"tokenized"
TRAIN_DIR="train"
TEST_DIR="test"

FAIRSEQ_DIR=${ROOT_DIR}/${MBART}/"fairseq"
BPE_MODEL=${ROOT_DIR}/${MBART}/"checkpoint"/"mbart.cc25.v2"/"sentence.bpe.model"

FAIRSEQ_SCRIPT=${FAIRSEQ_DIR}/"scripts"
SPM_ENCODE=${FAIRSEQ_SCRIPT}/"spm_encode.py"

SCRIPT=${ROOT_DIR}/"baseline/mbart/preprocess"
cd $SCRIPT

declare -a lang_list=("es" "ja" "ru")
declare -a suffix_list=("intent" "snippet")



# train sets 

mkdir -p ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}

for sfx in ${suffix_list[@]}; do
    python "${SPM_ENCODE}" --model=${BPE_MODEL} \
    < ${ROOT_DIR}/${LINES_DATA}/${TRAIN_DIR}/"train."${sfx} \
    > ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"train."${sfx} 
done 

for lang in ${lang_list[@]}; do
    mkdir -p ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"to_"${lang}

    for sfx in ${suffix_list[@]}; do
        python "${SPM_ENCODE}" --model=${BPE_MODEL} \
        < ${ROOT_DIR}/${LINES_DATA}/${TRAIN_DIR}/"to_"${lang}/"train_to_"${lang}.${sfx} \
        > ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"to_"${lang}/"train_to_"${lang}.${sfx}
    done 
done 



# test sets 
mkdir -p ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}

for sfx in ${suffix_list[@]}; do
    for lang in ${lang_list[@]}; do
        # test set 
        python "${SPM_ENCODE}" --model=${BPE_MODEL} \
        < ${ROOT_DIR}/${LINES_DATA}/${TEST_DIR}/${lang}"_test".${sfx} \
        > ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${lang}"_test".${sfx}

        # dev set 
        python "${SPM_ENCODE}" --model=${BPE_MODEL} \
        < ${ROOT_DIR}/${LINES_DATA}/${TEST_DIR}/"dev_"${lang}"_test".${sfx} \
        > ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/"dev_"${lang}"_test".${sfx}
    done 
done 


declare -a mt_list=("flores101" "marianmt" "m2m")
for mt in ${mt_list[@]}; do
    mkdir -p ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${mt}

    for sfx in ${suffix_list[@]}; do
        for lang in ${lang_list[@]}; do
            # test set 
            python "${SPM_ENCODE}" --model=${BPE_MODEL} \
            < ${ROOT_DIR}/${LINES_DATA}/${TEST_DIR}/${mt}/${lang}"_test_to_en".${sfx} \
            > ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${mt}/${lang}"_test_to_en".${sfx}

            # dev set 
            python "${SPM_ENCODE}" --model=${BPE_MODEL} \
            < ${ROOT_DIR}/${LINES_DATA}/${TEST_DIR}/${mt}/"dev_"${lang}"_test_to_en".${sfx} \
            > ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${mt}/"dev_"${lang}"_test_to_en".${sfx}
        done 
    done 
done
