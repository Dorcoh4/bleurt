import sys

file_name = sys.argv[1]
score = []
with open(file_name, 'r') as the_file:  
  for line in the_file:
    scores.append(float(line.strip()))
    
    
print (f"{len(scores)}  -> {sum(scores)/len(scores)}")
