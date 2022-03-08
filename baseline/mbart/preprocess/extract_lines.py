""" Extract intent and snippet instances into separate text files. 

Input: 
    directory containing the training/testing json/jsonl files 
    for training: 
        `dataset/train` (en)
        `dataset/train/to-es` (es)
        `dataset/train/to-ja` (ja)
        `dataset/train/to-ru` (ru)
    for testing: 
        `dataset/test` (es, ja, ru)
        `dataset/test/flores101` (en, flores-101)
        `dataset/test/marianmt` (en, marian-mt)
        `dataset/test/m2m` (en, m2m)
    
    note: `.json` and `.jsonl` files will be respectively tackled by:
        `xxx.json`        >>> `json_to_parallel`
        `xxx.jsonl`       >>> `jsonl_to_parallel` 
        
Output: 
    directory to output the processed files 
    e.g., `baseline/mbart/myproc`

    will write `.intent` and `.snippet` ending files, 
    having the same filename as the input ones. 
""" 


import os 
import argparse 
import json

from typing import Dict, List 

import logging 
logger = logging.getLogger()


# %% cleansing function 
def clean_intent(intent_str: str) -> str: 
    """Clean the raw (rewritten-) intent string. """ 
    return intent_str.replace("\n", "\\n").replace("\r", "\\r") 

def clean_snippet(snippet_str: str) -> str: 
    """Clean the raw snippet (code) string. """ 
    return snippet_str.replace("\n", "\\n") 



# %% read source samples 
def read_json_samples(path: str) -> List[Dict]: 
    with open(path, 'r') as fr: 
        dataset = json.load(fr) 
    logger.info(f"loaded {len(dataset)} samples from {path}") 
    return dataset 

def read_jsonl_samples(path: str) -> List[Dict]: 
    with open(path, 'r') as fr: 
        lines = [l.strip() for l in fr.readlines()]
        dataset = [json.loads(line) for line in lines] 
    return dataset     

READ_DICT = {
    'json': read_json_samples, 
    'jsonl': read_jsonl_samples, 
} 

# write extracted lines 
def write_lines(text_list: List[str], output_path: str): 
    with open(output_path, 'w') as fw: 
        for txt in text_list: 
            fw.write(f"{txt}\n") 
    


# %% extract parallel data: intent and snippet 
def extract_parallel(info_dict: Dict, split_test_by_half: bool = False): 
    """Extract intent/snippet and write as lines. """
    dataset = [] 
    for input_dict in info_dict['input_path_list']: 
        dataset.extend(
            READ_DICT[input_dict['file_type']](path=input_dict['input_path'])
        )

    intent_list, snippet_list = [], [] 
    for sample in dataset: 
        # get and clean the intent 
        s_intent = sample.get('rewritten_intent', "")
        if not s_intent: 
            s_intent = sample.get('intent', "")
        s_intent = clean_intent(s_intent)

        # get and clean the snippet 
        s_snippet = sample.get('snippet', "")
        s_snippet = clean_snippet(s_snippet)

        if not (s_intent and s_snippet): # check if case valid 
            continue 

        intent_list.append(s_intent)
        snippet_list.append(s_snippet)

    assert len(intent_list) == len(snippet_list)

    if ('output_path_intent_dev' in info_dict) and ('output_path_snippet_dev' in info_dict): 
        if split_test_by_half: 
            nhalf = len(intent_list) // 2 

            intent_list_dev = intent_list[: nhalf] 
            write_lines(intent_list_dev, info_dict['output_path_intent_dev'])
            intent_list = intent_list[nhalf: ]

            snippet_list_dev = snippet_list[: nhalf]
            write_lines(snippet_list_dev, info_dict['output_path_snippet_dev'])
            snippet_list = snippet_list[nhalf: ]
        else: 
            write_lines(intent_list, info_dict['output_path_intent_dev'])
            write_lines(snippet_list, info_dict['output_path_snippet_dev'])
        
    write_lines(intent_list, info_dict['output_path_intent'])
    write_lines(snippet_list, info_dict['output_path_snippet'])


def get_inputs_outputs_names_test(
    input_dir: str, 
    output_dir: str, 
    do_split: bool = False, 
): 
    """Collect the input-output file path pairs. """
    info_dict_list = []

    files = os.listdir(input_dir) 
    json_files = [f for f in files if f.endswith('.json')]
    logger.info(f"input json files: {json_files}")
    for jf in json_files: 
        jf_name = jf[: -len("json")]
        info_dict = {
            'input_path_list': [{
                'input_path': os.path.join(input_dir, jf), 
                'file_type': 'json', 
            }], 
            'output_path_intent': os.path.join(output_dir, f"{jf_name}intent"), 
            'output_path_snippet': os.path.join(output_dir, f"{jf_name}snippet"), 
        }
        if do_split: 
            jf_name_dev = f"dev_{jf_name}"
            info_dict.update({
                'output_path_intent_dev': os.path.join(output_dir, f"{jf_name_dev}intent"), 
                'output_path_snippet_dev': os.path.join(output_dir, f"{jf_name_dev}snippet"), 
            })
        info_dict_list.append(info_dict)

    return info_dict_list 


def get_inputs_outputs_names_train(
    input_dir: str, 
    output_dir: str, 
    use_json_only: bool = False, 
): 
    """Collect the input-output file path pairs. 
    Identify json/jsonl files in the input directory. 
    Determine the output file names accordingly. 
    """
    files = os.listdir(input_dir) 
    json_files = [f for f in files if f.endswith('.json')]
    path_list = [
        {
            'input_path': os.path.join(input_dir, jf), 
            'file_type': "json", 
        } 
        for jf in json_files
    ]

    if not use_json_only: 
        jsonlines_files = [f for f in files if f.endswith('.jsonl')]
        for jlf in jsonlines_files: 
            path_list.append({
                'input_path': os.path.join(input_dir, jlf), 
                'file_type': "jsonl", 
            })
    
    suffix = "."
    if '_' in json_files[0]: 
        suffix_index = json_files[0].index('_')
        suffix = json_files[0][suffix_index: -len("json")]
    info_dict = {
        'input_path_list': path_list, 
        'output_path_intent': os.path.join(output_dir, f"train{suffix}intent"), 
        'output_path_snippet': os.path.join(output_dir, f"train{suffix}snippet"), 
    }
    return [info_dict]
 



def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--input_dir', type=str, required=True, 
        help="Directory of the raw source data files. ") 
    parser.add_argument('--output_dir', type=str, required=True, 
        help="Directory to write the extracted intent and snippet. ") 
    parser.add_argument('--split_dev_test', action='store_true', 
        help="If split the files by half, into dev/test sets. ") 
    parser.add_argument('--use_json_only', action='store_true', 
        help="If only use the json files for training sample collection. ")
    args = parser.parse_args() 

    if not args.split_dev_test:   # train 
        info_dict_list = get_inputs_outputs_names_train(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            use_json_only=args.use_json_only, 
        )
    else: 
        info_dict_list = get_inputs_outputs_names_test(
            input_dir=args.input_dir, 
            output_dir=args.output_dir, 
            do_split=args.split_dev_test, 
        )

    for info_dict in info_dict_list: 
        extract_parallel(info_dict)  


if __name__ == "__main__":
    main()
 
