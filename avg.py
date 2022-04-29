import sys
import json

file_name = sys.argv[1]
file_name2 = sys.argv[2]
scores = []
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
    
print (f"{len(scores)}  -> {sum(scores)/len(scores)}")
