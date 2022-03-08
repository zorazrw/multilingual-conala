# Code generation from natural language with less prior and more monolingual data (TAE)

Paper published in [ACL 2021](https://aclanthology.org/2021.acl-short.98/)


install the requirments:
```
pip install -r requirements.txt
```

To train model on Django
```
python3 train.py --dataset_name django --save_dir CHECKPOINT_DIR --copy_bt --no_encoder_update --monolingual_ratio 1.0 --early_stopping
``` 
To evaluate the provided Django checkpoint:
```
python3 train.py --dataset_name django --save_dir pretrained_weights/django --copy_bt --no_encoder_update --monolingual_ratio 1.0 --early_stopping --just_evaluate --seed 2
``` 
To train model on CoNaLa
```
python3 train.py --dataset_name conala --save_dir CHECKPOINT_DIR --copy_bt --no_encoder_update --monolingual_ratio 0.5 --epochs 80
``` 
To evaluate the provided CoNaLa chceckpoint:
```
python3 train.py --dataset_name conala --save_dir pretrained_weights/conala --copy_bt --no_encoder_update --monolingual_ratio 0.5 --epochs 80 --just_evaluate --seed 4
```

### Evaluation Results
Here are the evaluation numbers for the provided checkpoints:

| Dataset | Results      | Metric             |
| ------- | ------------ | ------------------ |
| Django  | 81.77        | Exact Match Acc.   |
| CoNaLa  | 33.41        | Corpus BLEU        |
