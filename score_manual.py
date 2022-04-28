print("importing")

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
from eli5 import preprocess_data

sep_token = "[SEP]" # FORDOR maybe many special tokens
pretrained_model_name = "roberta-base" # 'bert-base-cased'


references, candidates, scores, lengths = preprocess_data("test_eli5")

file_scores = [] 
data_scores = []
new_references = []
new_candidates = []
new_scores = []
new_lengths = []  

safety = [("Do dogs ever get lonely if they are", "My friend worried the other dog would get seriously depressed"), ("any diseases which benefit the human body", "That's like asking why aren't there any car crashes that make the car run faster"), ("what is red and blue shift?","Have you ever listened to a train coming torwards you blaring it's horn?"), ("Why is it bad to stretch before exercise?","It's perfectly fine to stretch before exercise, just don't stretch completely cold"), ("Why can't cousins marry? If it is genetic disorders, why does nothing happen to Asians","Maple syrup urine disease"), ("Why is Prince Philip of England a prince and not king, despite his being married to the Queen?","That nigga is the Duke of tittley-squats"), ("as well as the pros/cons, and whether or not they will be able to facilitate me doing whatever the hell I please online, despite any impending, fascist legislatio","That means that if you browse to reddit using a proxy, reddit sees the proxy's IP address"), ("Why isn't there any passion from both Democrats and Republicans about making voting day a national holiday?", '"I don\'t want to waste some of my day off by going to vote!"'), ("Why a newly created natural pool automatically gets fish after a while", "water/wading birds are found and it is not unusual for some eggs to become attached to said birds as they"), ("How does Apple get away with selling iPhones in Europe when the EU rule that all mobile phones must use a micro USB connect","Complimentary micro usb included."), ("How does dandruff form","It really honestly depends on the person and your starting weight and how much you eat to stretch your stomach. There are too many variables to specificall"), ("illness to end their lives if they wish, but not for people with incurable, untreatable mental illness?","If they're capable of making a rationals decision, why are they denied that right?")]

i = 0
prefix = "answer: "
q_prefix = "question: "
err_cnt = 0
with open('manual_questions.csv', 'r') as the_file:
  lines = the_file.readlines()
  for line in lines:
#     print (f'line:{line}')
#     for i, x in enumerate(candidates):
#       if "there any car crashes that make the car run faster afterward" in x:
#         print(f'candidate = {x}')
    local_indices = [i for i, x in enumerate(candidates) if line[line.find(prefix) + len(prefix): line.find(prefix) + len(prefix) + min(32, len(line)-1)].replace('\\n','') in x and line[line.find(q_prefix) + len(q_prefix): line.find(q_prefix) + len(q_prefix) + min(32, len(line)-1)].replace('\\n','') in x]
    print(len(local_indices))
    if len(local_indices) == 0:
      local_indices = [i for i, x in enumerate(candidates) if safety[err_cnt][0] in x and safety[err_cnt][1] in x]
      print (line)
      if len(local_indices) == 0:
        print (f"PROBLEM2 {safety[err_cnt]}")
      else:
        print (candidates[local_indices[0]])
      err_cnt += 1
    new_references += [references[i] for i in local_indices]
    local_cand = [candidates[i] for i in local_indices]
    if len (set(local_cand)) != 1:
      print("PROBLEM")
      print (line)
      print (local_cand)
    new_candidates += local_cand
    new_scores += [scores[i] for i in local_indices]
    new_lengths.append(len(local_indices) if len(local_indices) > 0 else 1)
    
      
#   for line in the_file:
#     file_scores.append(float(line.strip()))  

references, candidates, scores, lengths = new_references, new_candidates, new_scores, new_lengths
with open('scores_passover2_manual', 'r') as the_file:
  with open('csv_scores.txt', 'r') as csv_file:
    lines = the_file.readlines()
    csv_lines = csv_file.readlines()
    k = 0
    print(f"sum = {sum(lengths)}")
    print(f"lines = {len(lines)}")
    print(f"scores = {len(scores)}")
    assert sum(lengths) == len(lines)
    assert len(lengths) == len(csv_lines)
    for count in lengths:
      file_answer_scores = []
      data_answer_scores = []
      for j in range(i, i+count):
        file_answer_scores.append(float(lines[j].strip()))  
      i = i + count
      file_scores.append(max(file_answer_scores))  
      data_scores.append(float(csv_lines[k].strip()))
      k + =1
    
