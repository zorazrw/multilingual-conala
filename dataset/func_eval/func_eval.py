"""Run Functional Evaluation on (M)CoNaLa Samples. """

import os 
import re 
import json 
import argparse 
import multiprocessing 

from typing import Any, Dict, List


# %% evaluation 
from datasets import load_metric

code_eval_metric = load_metric("code_eval")
bleu_eval_metric = load_metric("bleu")


# functional 
def func_eval(predictions: List[List[str]], annotations: List[Dict]): 
    code_preds_list, code_golds_list = [], [] 
    for p_list, a in zip (predictions, annotations): 
        prompt = a['prompt']
        suffix = a.get('suffix', '')

        wrapped_pred_list = [] 
        for p in p_list: 
            if p.startswith('print(') and p.endswith(')'): 
                p = p[len('print('):-1]
            if 'return' in prompt:
                if p.startswith('return '): 
                    p = p.lstrip('return').lstrip()

            wrapped_pred = f"{prompt}{p}{suffix}"
            wrapped_pred_list.append(wrapped_pred)
        
        code_preds_list.append(wrapped_pred_list)
        wrapped_test = f"\n{a['test']}\ncheck({a['entry_point']})"
        code_golds_list.append(wrapped_test)
    
    code_results = code_eval_metric.compute(
        predictions=code_preds_list, 
        references=code_golds_list, 
        k=args.num_samples, 
        num_workers=1,  # multiprocessing.cpu_count()
    )
    print(f"Code Eval: \n{code_results}") 


# bleu
def tokenize_python(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


def bleu_eval(predictions: List[List[str]], annotations: List[Dict]): 
    bleu_preds_list, bleu_golds_list = [], [] 
    for p_list, a in zip (predictions, annotations): 
        p = p_list[0]

        bleu_preds_list.append(tokenize_python(p))
        if args.is_test_file: 
            bleu_golds_list.append([tokenize_python(a['snippet'])])
        else:
            bleu_golds_list.append([tokenize_python(a['canonical_solution'])])

    bleu_results = bleu_eval_metric.compute(
        predictions=bleu_preds_list, 
        references=bleu_golds_list, 
    )
    print(f"Bleu Eval: \n{bleu_results}")


# %% file i/o 

def load_data(path: str, file_type: str) -> List[Any]:
    with open(path, 'r') as fr: 
        if file_type == 'txt': 
            data = [[l.strip()] for l in fr]
        elif file_type == 'json': 
            data = json.load(fr)
        elif file_type == 'jsonl':
            data = [json.loads(l.strip()) for l in fr]
        else: 
            data = None
    return data


# %% main pipeline 

def main(): 
    # load predictions & annotation
    p_file_suffix = args.prediction_file.split('.')[-1]
    predictions = load_data(args.prediction_file, p_file_suffix)
    print(f"load predictions #{len(predictions)}")

    a_file_suffix = args.annotation_file.split('.')[-1]
    annotations = load_data(args.annotation_file, a_file_suffix)
    print(f"load annotations #{len(annotations)}")
    
    def get_index(task_id: str) -> int: 
        return int(task_id.split('-')[-1])
    if not args.is_test_file: 
        predictions = [predictions[get_index(a['task_id'])] for a in annotations]
    assert len(predictions) == len(annotations)
    
    if 'task_id' in predictions[0]: 
        predictions = [p['predictions'] for p in predictions]

    if args.do_func_eval: 
        # enables code execution in code_eval metric
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

        func_eval(predictions, annotations)
    
    if args.do_bleu_eval: 
        bleu_eval(predictions, annotations)
    
    if args.do_per_case: 
        for idx, (p, a) in enumerate(zip(predictions, annotations)): 
            print(f"\n\n#{idx}")
            func_eval([p], [a])
            bleu_eval([p], [a])
    



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="(M)CoNaLa functional evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prediction-file",
        type=str,
        help="Predictions in `.txt/json/jsonl` file",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        help="Annotations in `.jsonl` file",
    )
    parser.add_argument(
        "--is-test-file", 
        action='store_true', 
        help="If loaded annotation is not unit test.", 
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        nargs='+',
        default=[1],
        help="Number of samples generated per prompt",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only for the first k examples",
    )
    parser.add_argument(
        "--do-func-eval", 
        action='store_true', 
        help="If evaluating function correctness", 
    )
    parser.add_argument(
        "--do-bleu-eval", 
        action='store_true', 
        help="If evaluating function correctness", 
    )
    parser.add_argument(
        "--do-per-case", 
        action='store_true', 
        help="If evaluate specified metrics per case. ", 
    )
    args = parser.parse_args() 

    main() 