import sys
import json
from datasets import load_metric


file_name = sys.argv[1]
file_name2 = sys.argv[2]
file_name3 = sys.argv[3]

scores = []
corr_scores = []
with open(file_name, 'r') as in_file:  
  with open(file_name2, 'r') as out_file:
    in_lines = in_file.readlines()
    out_lines = out_file.readlines()
    assert len(out_lines) == len(in_lines)
    i = 0
    while i < len(out_lines):
      local_score = [float(out_lines[i].strip())]
      candidate = json.loads(in_lines[i])["candidate"]
      i += 1
      while i < len(out_lines) and candidate == json.loads(in_lines[i])["candidate"]:
        local_score.append(float(out_lines[i].strip()))
        i += 1
      scores.append(max(local_score))
    
with open(file_name3, 'r') as corr_file:  
  for line in corr_file:
    corr_score.append(float(line.strip()))

    
metric = load_metric("spearmanr")
print (f"FORDOR result: {metric.compute(predictions=scores, references=corr_scores)}")
metric = load_metric("pearsonr")
print (f"FORDOR result: {metric.compute(predictions=scores, references=corr_scores)}")
