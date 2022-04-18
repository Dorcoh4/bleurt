import json
scores = []
with open('train_eli5.json') as f_in:
  for line in f_in:
    obj = json.loads(line)
    scores.append(float(obj['score']))
    
print (max(scores))
print(min(scores))
print(sum(scores)/len(scores))
print(len(scores))
print(sorted(scores)[len(scores)//2])
