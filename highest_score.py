import json
scores = []
with open('train_eli5.json') as f_in:
  for line in f_in:
    obj = json.loads(line)
    scores.append(float(obj['score']))
    
scores = sorted(scores)
print (max(scores))
print(min(scores))
print(sum(scores)/len(scores))
print(len(scores))
print(scores[len(scores)//2])
print(scores[len(scores)-1])
print(scores[len(scores)-2])
print(scores[len(scores)-3])
print(scores[len(scores)-4])

