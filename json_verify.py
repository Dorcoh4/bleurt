import json
with open("sentence_pairs_manual.jsonl", 'r') as the_file:
  with open("tmp.jsonl", 'w') as to_file:
    for line in the_file:
      try:
        json.loads(line)
        to_file.write(line)
      except:
        to_file.write(line[1:])
        print("An exception occurred: " + line)
