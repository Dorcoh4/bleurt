import sys
from datasets import load_metric
import json
file_name = sys.argv[1]

bertscore = load_metric("bertscore")
# bleurt = load_metric("bleurt")
rouge = load_metric("rouge")
# bleu = load_metric("bleu")


candidates = []
references = []
with open(file_name, 'r') as the_file:  
#   with open("bleurt_"+file_name, 'w') as bleurt_file:  
  with open("rouge_" + file_name, 'w') as rouge_file:  
    with open("bertscore_" + file_name, 'w') as bertscore_file: 
#       with open("bleu_" + file_name, 'w') as bleu_file: 
      for line in the_file:
        obj = json.loads(line)
        candidates.append(obj["candidate"])
        references.append(obj["reference"])
      bertscore_scores = bertscore.compute(predictions=candidates, references=references, lang="en") ['f1']
      rouge_scores = rouge.compute(predictions=candidates, references=references)['rougeL']
#         bleu_scores = bleu.compute(predictions=candidates, references=references)[]
      rouge_scores = [score["fmeasure"] for score in rouge_scores]
      if len(rouge_scores) < 10:
        print(rouge_scores)
      print(len(rouge_scores))
      print(len(bertscore_scores))
      assert len(rouge_scores) == len(bertscore_scores)
      for i in range(len(rouge_scores)):
        rouge_file.write(str(rouge_scores[i]) + "\n" )
        bertscore_file.write(str(bertscore_scores[i]) + "\n" )
    
print (f"{len(bertscore_scores)}  -> {sum(scores)/len(bertscore_scores)}")
print("that was bertscore now rougeL")
print (f"{len(rouge_scores)}  -> {sum(scores)/len(rouge_scores)}")
