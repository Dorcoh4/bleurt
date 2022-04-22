
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, DefaultFlowCallback, PrinterCallback
from transformers import Trainer
import torch
from torch import nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import random
import json



def make(guess_file, gold_file):
  references = []
  candidates = []
  lengths = []
  with open(guess_file, 'r') as the_file:
    guess_lines = the_file.readlines()
  with open(gold_file, 'r') as the_file:
    gold_lines = the_file.readlines()
    
  for i in range(len(gold_lines)):
    gold_obj = json.loads(gold_lines[i])
    guess_obj = json.loads(guess_lines[i])
    question = "question: " + guess_obj["input"]
    candidates.append(question + " answer: " + guess_obj["output"][0]["answer"])
    lenghts.append(len(gold_obj["output"]))
    for ref in gold_obj["output"]:
      references.append(question + " answer: " + ref["answer"])
    
    
    
  with open(f"sentence_pairs_{guess_file[:guess_file.find(".")]}.jsonl", 'a') as the_file:
    for i in range(len(references)):
    reference = references[i]
    candidate = candidates[i]
    the_file.write(f'{{"candidate": {json.dumps(candidate)}, "reference": {json.dumps(reference)}}}\n')
    
    
make("copy_input.jsonl", "gold_copy_input.jsonl")
make("random_train_ans.jsonl", "gold_copy_input.jsonl")
make("longest_gold_ans.jsonl", "gold_longest_gold_gold.jsonl")
