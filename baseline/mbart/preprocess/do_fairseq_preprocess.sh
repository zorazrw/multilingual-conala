# do fairseq pre-processing to binarize the dataset 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi


MBART="baseline/mbart"
TOKENIZED_DATA=${MBART}/"dataset"/"tokenized"
BINARIZED_DATA=${MBART}/"dataset"/"binarized"

DICT=${ROOT_DIR}/${MBART}/"checkpoint"/"mbart.cc25.v2"/"dict.txt"


TRAIN_DIR="train"
TEST_DIR="test"

SRC="intent"
TGT="snippet"

declare -a lang_list=("es" "ja" "ru")

# trans-train 
mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"trans_train"

for lang in ${lang_list[@]}; do
	mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"trans_train"/${lang} 

	fairseq-preprocess --workers 50 \
	-s ${SRC} -t ${TGT} \
	--srcdict ${DICT} --tgtdict ${DICT} \
	--thresholdsrc 0 --thresholdtgt 0 \
	--trainpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"to_"${lang}/"train_to_"${lang} \
	--validpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/"dev_"${lang}"_test" \
	--testpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${lang}"_test" \
	--destdir ${ROOT_DIR}/${BINARIZED_DATA}/"trans_train"/${lang}
done 


# trans-test 
MT="flores101"
mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"trans_test"

for lang in ${lang_list[@]}; do
	mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"trans_test"/${lang} 

	fairseq-preprocess --workers 50 \
	-s ${SRC} -t ${TGT} \
	--srcdict ${DICT} --tgtdict ${DICT} \
	--thresholdsrc 0 --thresholdtgt 0 \
	--trainpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"train" \
	--validpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${MT}/"dev_"${lang}"_test_to_en" \
	--testpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${MT}/${lang}"_test_to_en" \
	--destdir ${ROOT_DIR}/${BINARIZED_DATA}/"trans_test"/${lang}
done 


# zero-shot 
# mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"zero_shot"
# 
# for lang in ${lang_list[@]}; do
# 	mkdir -p ${ROOT_DIR}/${BINARIZED_DATA}/"zero_shot"/${lang} 
# 	
# 	fairseq-preprocess --workers 50 \
# 	-s ${SRC} -t ${TGT} \
# 	--srcdict ${DICT} --tgtdict ${DICT} \
# 	--thresholdsrc 0 --thresholdtgt 0 \
# 	--trainpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TRAIN_DIR}/"train" \
# 	--validpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/"dev_"${lang}"_test" \
# 	--testpref ${ROOT_DIR}/${TOKENIZED_DATA}/${TEST_DIR}/${lang}"_test" \
#	--destdir ${ROOT_DIR}/${BINARIZED_DATA}/"zero_shot"/${lang}
# done 
