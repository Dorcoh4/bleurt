import json
with open("sentence_pairs_manual.jsonl", 'r') as the_file:
  for line in the_file:
    try:
      json.loads(line)
    except:
      print("An exception occurred: " + line)
