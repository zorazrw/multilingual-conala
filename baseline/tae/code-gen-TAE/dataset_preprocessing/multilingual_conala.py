"""Data Pre-processing Module for the Multilingual CoNaLa Dataset. """


import os, json 
from typing import Dict, List 
from .monolingual_python import MonolingualPython 
from .conala import Conala 



class MConalaFlores101(MonolingualPython): 
    def __init__(self, name, split, tokenizer, args, monolingual=False): 
        self.threshold = {'train': 10000, 'dev': 10000, 'test': 10000} 
        super(MConalaFlores101, self).__init__(
            name, # ['es-101', 'ja-101', 'ru-101']
            split, tokenizer, args, monolingual) 
  
    def preprocess_dataset(self, file_path: str) -> List[Dict]: 
        try: 
            dataset = json.load(open(file_path)) 
        except: 
            dataset = [json.loads(jline) for jline in open(file_path).readlines()] 
        
        examples = [] 
        for i, example_json in enumerate(dataset): 
            try: 
                example_dict = Conala.preprocess_example(example_json)
            except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
                continue
            examples.append(example_dict)
        return examples

    def _preprocess(self) -> List[Dict]: 
        json_file = os.path.join(self.dir_name, f"{self.split}.json")   # data/es-101/train.json
        if not os.path.exists(json_file): 
            orig_json_file = os.path.join(self.dir_name, "source", f"{self.split}.json")   # data/es-101/source/train.json
            examples = self.preprocess_dataset(orig_json_file)
            with open(json_file, 'w') as fw: 
                json.dump(examples, fw) 
        else: 
            with open(json_file, 'r') as fr: 
                examples = json.load(fr) 
        return examples 
    
    def _download_dataset(self): 
        return 



class MConalaMarianMT(MonolingualPython): 
    def __init__(self, name, split, tokenizer, args, monolingual=False): 
        self.threshold = {'train': 10000, 'dev': 10000, 'test': 10000} 
        super(MConalaMarianMT, self).__init__(
            name, # ['es-mmt', 'ja-mmt', 'ru-mmt']
            split, tokenizer, args, monolingual) 
  
    def preprocess_dataset(self, file_path: str) -> List[Dict]: 
        try: 
            dataset = json.load(open(file_path)) 
        except: 
            dataset = [json.loads(jline) for jline in open(file_path).readlines()] 
        
        examples = [] 
        for i, example_json in enumerate(dataset): 
            try: 
                example_dict = Conala.preprocess_example(example_json)
            except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
                continue
            examples.append(example_dict)
        return examples

    def _preprocess(self) -> List[Dict]: 
        json_file = os.path.join(self.dir_name, f"{self.split}.json")  # data/es-mmt
        if not os.path.exists(json_file): 
            orig_json_file = os.path.join(self.dir_name, "source", f"{self.split}.json")
            examples = self.preprocess_dataset(orig_json_file)
            with open(json_file, 'w') as fw: 
                json.dump(examples, fw) 
        else: 
            with open(json_file, 'r') as fr: 
                examples = json.load(fr) 
        return examples 
    
    def _download_dataset(self): 
        return 


class MConalaM2M(MonolingualPython): 
    def __init__(self, name, split, tokenizer, args, monolingual=False): 
        self.threshold = {'train': 10000, 'dev': 10000, 'test': 10000} 
        super(MConalaM2M, self).__init__(
            name, # ['es-m2m', 'ja-m2m', 'ru-m2m']
            split, tokenizer, args, monolingual) 
  
    def preprocess_dataset(self, file_path: str) -> List[Dict]: 
        try: 
            dataset = json.load(open(file_path)) 
        except: 
            dataset = [json.loads(jline) for jline in open(file_path).readlines()] 
        
        examples = [] 
        for i, example_json in enumerate(dataset): 
            try: 
                example_dict = Conala.preprocess_example(example_json)
            except (AssertionError, SyntaxError, ValueError, OverflowError) as e:
                continue
            examples.append(example_dict)
        return examples

    def _preprocess(self) -> List[Dict]: 
        json_file = os.path.join(self.dir_name, f"{self.split}.json") 
        if not os.path.exists(json_file): 
            orig_json_file = os.path.join(self.dir_name, "source", f"{self.split}.json")
            examples = self.preprocess_dataset(orig_json_file)
            with open(json_file, 'w') as fw: 
                json.dump(examples, fw) 
        else: 
            with open(json_file, 'r') as fr: 
                examples = json.load(fr) 
        return examples 
    
    def _download_dataset(self): 
        return 
