# MCoNaLa 

<p align="left">
    <a href="http://creativecommons.org/licenses/by-sa/4.0/">
        <img alt="CC BY-SA 4.0" src="https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg">
    </a>
    <br/>
</p>

This repository contains the MCoNaLa dataset and the code implementation of baseline models in the following paper: 

[MCoNaLa: A Benchmark for Code Generation
from Multiple Natural Languages](https://arxiv.org/pdf/2203.08388.pdf)

### 🤗 Huggingface Hub 
MCoNaLa is available on Huggingface Hub [here](https://huggingface.co/datasets/neulab/mconala) 

### 📊 ExplainaBoard Leaderboard
MCoNaLa has its leaderboard powered by [ExplainaBoard](https://explainaboard.inspiredco.ai/leaderboards?dataset=mconala), where you can upload and analyze your own system results with just a few simple clicks. Follow the detailed instructions below to submit your results to the leaderboard. 



## Benchmark Dataset 

### 1. Multilingual Samples: Spanish, Japanese, Russian 
The **Multilingual CoNaLa** dataset contains intent-snippet pairs collected from three different language versions of StackOverflow forums. 

These samples are located in the `dataset/test` directory, where `es_test.json`/`ja_test.json`/`ru_test.json` are original annotated samples. 

For the **trans-test** setting in baseline experiments, we also provide the translated version under the `flores101` directory: `es_test_to_en.json`/`ja_test_to_en.json`/`ru_test_to_en.json`, where the Spanish/Japanses/Russian intents are translated into English using the [FLORES-101](https://github.com/facebookresearch/flores) model. 

To study the influence of translation quality, we also experiment with two other widely used Machine Translation (MT) systems: [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) and [M2M](https://github.com/pytorch/fairseq/tree/main/examples/m2m_100). The intents in Spanish/Japanese/Russian samples are translated into English using the respective MT systems and put into the `marianmt` and `m2m` directories. 


### 2. English Samples for Training 
Due to the limited sample of multiple languages, we use English [CoNaLa](https://conala-corpus.github.io/) samples for training, where the intents are originally written in English. 
In the `dataset/train` directory contains the annotated `train.json`, the automatically mined samples from the StackOverflow webpages (`mined.jsonl`) and the API documents (`api.jsonl`). 

However, due to the uploading file size limitation of GitHub, we alternatively provide the training data via [zenodo](https://zenodo.org/record/6359692#.YjFN9BDMJFM).

In the **trans-train** experiment setting, we also translate the English intents into the three target languages of interest using [FLORES-101](https://github.com/facebookresearch/flores), under the `to-es` / `to-ja` / `to-ru` directories.  


### 3. Data Usage

Spanish, Japanese, and Russian are of the Target Language (TL), whose samples are always (only) used for testing purpose due to the limited amount. 

English is the High-Resource Language (HRL) for which the samples can be leveraged for model training. 

To give an illustration, the directory is organized as: 
```
.
├── README.md
├── datasets
│   ├── test 
│   │   ├── flores101
│   │   │   ├── es_test_to_en.json
│   │   │   ├── ja_test_to_en.jsonl
│   │   │   └── ru_test_to_en.jsonl
│   │   ├── marianmt
│   │   │   ├── es_test_to_en.json
│   │   │   ├── ja_test_to_en.jsonl
│   │   │   └── ru_test_to_en.jsonl
│   │   ├── m2m
│   │   │   ├── es_test_to_en.json
│   │   │   ├── ja_test_to_en.jsonl
│   │   │   └── ru_test_to_en.jsonl
│   │   ├── es_test.json
│   │   ├── ja_test.json
│   │   └── ru_test.json
│   ├── train 
│   │   ├── to-es
│   │   │   ├── train_to_es.json
│   │   │   ├── mined_to_es.jsonl
│   │   │   └── api_to_es.jsonl
│   │   ├── to-ja
│   │   │   ├── train_to_ja.json
│   │   │   ├── mined_to_ja.jsonl
│   │   │   └── api_to_ja.jsonl
│   │   ├── to-ru
│   │   │   ├── train_to_ru.json
│   │   │   ├── mined_to_ru.jsonl
│   │   │   └── api_to_ru.jsonl
│   │   ├── train.json
│   │   ├── mined.jsonl
└── └── └── api.jsonl
```

- translate-train 

The **trans-train** setting evaluates samples in different langauges as independent tasks. Take Spanish (es) as an example, we use the translated CoNaLa samples in `train/to-es` (`train_to_es.json`, `mined_to_es.jsonl`, and `api_to_es.jsonl`) for training, then test on `test/es_test.json`. 
Japanese (ja) and Russian (ru) samples work in similar mechanisms.

- translate-test 

The **trans-test** setting evaluates samples in three target languages using the same model. Specifically, we use the original English CoNaLa samples `train/train.json`, `train/mined.jsonl`, and `train/api.jsonl` in joint for training. The resulting model are evaluated on the translated version `es_test_to_en_xxx.json`, `ja_test_to_en_xxx.json`, `ru_test_to_en_xxx.json`. `xxx` stands for the MT model used (flores101, marianmt, m2m). Our experiments test on the `flores101`-translated samples by default. 

- zero-shot 

The **zero-shot** setting trains the model using English samples (`train/train.json`, `train/mined.jsonl`, `train/api.jsonl`) and directly tests on multilingual samples (`test/es_test.json`, `test/ja_test.json`, `test/ru_test.json`). 
Intuitively, this require the model being able to encode natural langauge intents in multiple language without intentional training. 



## Submitting Results to the Leaderboard

Go to the submission site [here](https://explainaboard.inspiredco.ai/leaderboards?dataset=mconala) and click the __New__ button on the top-right to start a new submission, then fill out a few blanks in the pop-up window: 
- *System Name*: give an informative name for your system
- *Task*: select 'machine-translation' from the drop-down list
- *Dataset*: select 'mconala' with the target language (es/ja/ru) from the drop-down list, and for *Split* select 'test' 
- *System Output*: click on 'Text' and submit your results in TXT format. Please make sure that your results file has the same number of lines as the corresponding testset. If a predicted code snippet contains `\n` that could spread one prediction into multiple lines. One trick to fix this is doing `a_multi_line_string.replace('\n', '\\n')` before writing into the file. 
- *Metrics*: select 'bleu', which computes the code-specific BLEU (-4) score.
- check that the *Input Lang* is automatically filled with your target NL (es/ja/ru) and the *Output Lang* is python. 

Click the __Submit__ button on the bottom, then your results are ready in a few seconds! 

You can also click the __Analysis__ button on the right to view more fine-grained analyses with cool figures 📊



## Baseline Models
To present the baseline performance on the Multilingual CoNaLa dataset, we use three state-of-the-art models that are proficient at multilingual learning or code generation. 

Set the root directory using the following command, as this would be required by most experimental bash scripts. 
```bash
export ROOT_DIR=`pwd`
```


### 1. mBART
[mBART](https://github.com/pytorch/fairseq/blob/main/examples/mbart/README.md) is a multilingual denoising auto-encoder trained for machine translation tasks. 

To reproduce the baseline result of mBART, following: 
#### Installation [fairseq](https://github.com/pytorch/fairseq)
Clone and install the repository. 
```bash 
git clone git@github.com:pytorch/fairseq.git
cd fairseq
# pip install .
pip install fairseq=0.10.2
# pip install fairseq=1.0.0a0+53bf2b1
cd ..
```
warning: may require earlier versions to solve some instantiation error (e.g., ```fairseq==0.10.2```).

Also download the pre-trained mBART model checkpoint. 
```bash 
mkdir checkpoint && cd checkpoint 
wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz 
tar -xzvf mbart.cc25.v2.tar.gz
cd ..
```


#### Data Pre-processing 
Data pre-processing are conducted on both nl-intent and code-snippet, and in three consecutive steps: 1) sentence-piece tokenization, 2) fairseq preprocessing, and 3) data binarization. 

Before the pre-processing, make sure to install SPM [here](https://github.com/google/sentencepiece), or run: 
```bash
pip install sentencepiece
```

First, we need to extract the intent and snippets into a line-by-line text file. 
To process all samples in the provided `dataset`, use the script 
```bash 
bash extract_lines.sh 
``` 
which will create a `dataset/lines` directory with all processed training and testing files. 

One can also process json/jsonl files in a specific folder by: 
```bash 
python extract_lines.py --input_dir source_dir --output_dir target_dir 
```

Next, to perform the spm tokenization, run 
```bash
bash do_spm_tokenization.sh
```

Lastly, do the fairseq pre-processing to binarize the data files
```bash
bash do_fairseq_preprocess.sh
```
By default, this step will use the FLORES-101 translation for `trans-test` evaluation. 


#### Training and Evaluation 
Head into the `baseline/mbart/experiment` directory. 

To fine-tune a pre-trained mBART model: 
```bash
bash run_train.sh
```
Note that we only need to train the model for `trans_train` and `trans_test` settings. Evaluation on `zero_shot` setting can directly load the saved checkpoint from the `trans_test` experiment. 

To evaluate on `trans_train` or `trans_test` setting: 
```bash 
bash run_test.sh
```
`run_test_zero_shot.sh` should be easier to use for evaluation in the `zero_shot` setting. 

You can change the `SETTING` (trans_train, trans_test) and `LANG` (es, ja, ru) in both scripts to run different experiments. 


### 2. TranX 
[TranX](https://arxiv.org/abs/2004.09015) is a pre-trained natural language to code generation model by leveraging external knowledge. 
Our experiments uses its [code](https://github.com/neulab/external-knowledge-codegen) implementation to perform training and testing on the Multilingual CoNaLa dataset.  

To reproduce the TranX results: 
#### Installation 
Clone the repository and install required libraries. 
```bash
cd baseline/tranx
# git clone https://github.com/neulab/external-knowledge-codegen.git 

pip install python==3.7
pip install pytorch==1.1.0
pip install astor==0.7.1       # this is very important
```

#### Data Pre-processing 
```bash
bash baseline/tranx/scripts/preprocess.sh
```
This will organize and process the train-test files for both the `trans-train` and `trans-test` settings for three languages. 

Note: be sure to download the necessary resource via
```bash
import nltk
nltk.download('punkt')
```


#### Model Training and Evaluation 
Head into the TranX directory using 
```bash
cd baseline/tranx
```

To pre-train with additional mined data and api documents under a specific `SETTING` for a specific `LANG`uage, run 
```bash
bash scripts/run_train.sh
```

To further fine-tune with the annotated training set, run 
```bash
bash scripts/run_tune.sh 
```

Use the `scripts/test_mconala.sh` for evaluation. 

We provide the best pre-trained model checkpoint for all three languages and both settings, under the `best_pretrained_models/mconala`. Alter the language and setting arguments in the bash script to run individual experiments. 
```bash
bash scripts/test_mconala.sh 
```



### 3. TAE 
[TAE](https://aclanthology.org/2021.acl-short.98/) is a seq2seq model, augmented with a target auto-encoding objective, for code generation from English intents.

The `tae` code implementation is built upon its original [repository]((https://github.com/BorealisAI/code-gen-TAE)). To reproduce the baseline performance of TAE, following: 
#### Installation
Clone the repository and install necessary libraries. 
```bash
cd baseline/tae/code-gen-TAE/
pip install -r requirements.txt
```

Download the pre-trained TAE model from [here](https://github.com/BorealisAI/code-gen-TAE/raw/main/pretrained_weights/conala/conala_model_combined_training%3DFalse_seed%3D4_trns_back%3DFalse_use_backtr%3DFalse_lmd%3D1_cp_bt%3DTrue_add_no%3DFalse_no_en_upd%3DTrue_ratio%3D0.5_ext_li%3DTrue_ext_cp_li%3DTrue_cp_att%3DTrue_EMA%3DT_rnd_enc%3DF_de_lr%3D7.5e-05_mmp%3D0.1_saug%3DF_dums%3DF_dumQ%3DF_rsr%3DF_fc%3DF_ccr%3DF.pth). 


#### Data Pre-processing
Copy the test samples (with intents translated into English). 
```bash
bash ../collect_data.sh 
```
uses the FLORES-101 translation by default. 


#### Evaluation 

To reproduce the evaluation result on Spanish CoNaLa samples, run
```bash
python3 test_mconala.py \
  --dataset_name "es-101" \
  --save_dir "pretrained_weights/conala" \
  --copy_bt --no_encoder_update --seed 4 \
  --monolingual_ratio 0.5 --epochs 80 \
  --use_conala_model
```
Change `es-101` to `ja-101`/`ru-101` to test the Japanese/Russian samples. 
Change `xx-101` to `xx-mmt` or `xx-m2m` to test with different machine translation models. 


Also, to evaluate on the English CoNaLa samples 
```bash
python3 train.py \
  --dataset_name "conala" \
  --save_dir "pretrained_weights/conala" \
  --copy_bt --no_encoder_update --seed 4 \
  --monolingual_ratio 0.5 --epochs 80 \
  --just_evaluate
```


## Reference 

```
@article{wang2022mconala,
  title={MCoNaLa: A Benchmark for Code Generation from Multiple Natural Languages},
  author={Zhiruo Wang, Grace Cuenca, Shuyan Zhou, Frank F. Xu, Graham Neubig},
  journal={arXiv preprint arXiv:2203.08388},
  year={2022}
}
```
