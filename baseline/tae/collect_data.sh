# collect test data for tae trans-test experiment 

set -euo pipefail

if [[ -z $ROOT_DIR ]]; then
  echo "\$ROOT_DIR enviromental variable needs to be set"
  exit 1
fi

declare -a lang_list=("es" "ja" "ru")

SRC_DATA=${ROOT_DIR}/"dataset"/"test"
TGT_DATA=${TAE_DIR}/"code-gen-TAE"/"data"

MT_SRC="flores101"    # ("flores101" "marianmt" "m2m")
MT_TGT="101"          # ("101" "mmt" "m2m")

TAE_DIR=${ROOT_DIR}/"baseline"/"tae"
cd ${TAE_DIR}

for lang in ${lang_list[@]}; do
  mkdir -p ${TGT_DATA}/${lang}"-"${MT_TGT}/"source"

  cp -r ${SRC_DATA}/${MT_SRC}/${lang}"_test_to_en.json" \
        ${TGT_DATA}/${lang}"-"${MT_TGT}/"source"/"test.json"
done 
