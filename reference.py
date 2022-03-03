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

sep_token = "[SEP]" # FORDOR maybe many special tokens
pretrained_model_name = "roberta-base" # 'bert-base-cased'



class my_Bert(nn.Module):
  def __init__(self, bert):
    super().__init__()
    self.bert = bert



  def forward(self,input_ids,attention_mask=None,labels=None,**kwargs):
      res = self.bert.forward(input_ids,attention_mask,labels=labels,**kwargs)
      print(f"FORDOR-input_ids {input_ids}")
      print(f"FORDOR-inputss {tokenizer.decode(input_ids[0])}")
      print(f"FORDOR-inputss {tokenizer.decode(input_ids[1])}")
      print(f"FORDOR-labels {labels}")
      print(f"FORDOR-res {res}")
      return res



print("starting load")


# for i in range(len(dataset["train_eli5"])):
#     print(f'train= {dataset["train_eli5"][i]["answers"]}')
#     print(f'valid= {dataset["validation_eli5"][i]["answers"]}')
#     print(f'test= {dataset["test_eli5"][i]["answers"]}')



class ELI5MetricDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
  
  
def changeArr(input1):
 
    # Copy input array into newArray
    newArray = input1.copy()
     
    # Sort newArray[] in ascending order
    newArray.sort()
     
    # Dictionary to store the rank of
    # the array element
    ranks = {}
     
    rank = 1
     
    for index in range(len(newArray)):
        element = newArray[index];
     
        # Update rank of element
        if element not in ranks:
            ranks[element] = rank
            rank += 1
         
    # Assign ranks to elements
    for index in range(len(input1)):
        element = input1[index]
        input1[index] = float(ranks[input1[index]])

my_dataset = {}




if False:# try:
    with open("my_dataset.pickle", "rb" ) as f:
        my_dataset = pickle.load(f)
else: # except IOError:
    print("could not load my_dataset - preprocessing")
    raw_datasets = load_dataset("eli5")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def preprocess_data(split_name):
        with open('references', 'a') as the_file:
    
          inputs = []
          labels = []
          cnt = 0
          for example in raw_datasets[split_name]:

              question = example["title"]+ example["selftext"] #FORDOR add special sep token?
              for i in range (1, len (example["answers"]["a_id"])):
                  answer = example["answers"]["text"][i]
#                   question = question.replace('"','\\"')
#                   answer = answer.replace('"','\\"')
                  candidate = f'question: {question} answer: {answer}'
                  reference = f'question: {question} answer: {example["answers"]["text"][0]}'
                  nl = '\n'
                  the_file.write(f"{reference.replace(nl,'\t')}\n")
#                   the_file.write(f'{{"candidate": {json.dumps(candidate)}, "reference": {json.dumps(reference)}, "score": {example["answers"]["score"][i]} }}\n')
#                   inputs.append(question + sep_token + answer)
  #                 print (f'FORDOR float - {float(example["answers"]["score"][i])} {example["answers"]["score"][i]}')
#                   labels.append(float(example["answers"]["score"][i]))
                  cnt = cnt+1
#                   if cnt > 200000:
#                     break

        # tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        
        #shuffle data
#         c = list(zip(inputs, labels))
#         random.seed(42)
#         random.shuffle(c)
#         inputs, labels = zip(*c)
#         inputs = list(inputs)
#         labels = list(labels)

        
#         encodings = tokenizer(inputs, padding="max_length", truncation=True)
#         encodings2 = tokenizer(inputs, padding="max_length", truncation=False)
#         for i in range(len(encodings)):
#             if len(encodings[i]) != len( encodings2[i]):
#                 print (print(f"encoding and length {encodings[i]}, {len(encodings[i])} no truncation = {encodings2[i]},  {len(encodings2[i])}"))
#         
        
        
#         tensor_labels = torch.as_tensor(labels).reshape(-1,1)
#         scaler = StandardScaler()
#         scaler.fit(tensor_labels)
#         scaled_labels = scaler.transform(tensor_labels).astype(np.float32)
#         changeArr(labels)

      
#         my_dataset[split_name] = ELI5MetricDataset(encodings, scaled_labels)
#         print (f"FORDOR lens {len(encodings)}=={len(labels)}")
    #     assert len(encodings) == len(labels)
    
    preprocess_data("train_eli5")
    preprocess_data("validation_eli5")
#     pickle.dump( my_dataset, open( "my_dataset.pickle", "wb" ) )

# metric = load_metric("spearmanr")
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     print(f'logits- {max(logits)}, {min(logits)}')
#     print(f'labels- {max(labels)}, {min(labels)}')
#     return metric.compute(predictions=logits, references=labels)

# model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=1)
# # freezing bert parameters leaving only regression layer
# # for param in model.bert.parameters():
# #     param.requires_grad = False
# # model = my_Bert(model)
# # print (f"FORDOR model = {str(model)}")
# # print (f'FORDOR debug {raw_datasets["train_eli5"][0]["answers"]} =:= {model(input_ids=my_dataset["train_eli5"][0]["input_ids"].unsqueeze(0), attention_mask=my_dataset["train_eli5"][0]["attention_mask"].unsqueeze(0), token_type_ids=my_dataset["train_eli5"][0]["token_type_ids"].unsqueeze(0))}')
# training_args = TrainingArguments("test_trainer", evaluation_strategy="steps", eval_steps=10000, save_steps=10000, per_device_train_batch_size=8, per_device_eval_batch_size=8)
# trainer = Trainer(model=model, args=training_args, train_dataset=my_dataset["train_eli5"], eval_dataset=my_dataset["validation_eli5"], compute_metrics=compute_metrics,
#                  callbacks = [
#                 DefaultFlowCallback(),
#                  PrinterCallback()
#     ],
#                  )
# #, max_steps=3000 
# trainer.train()

# # model.eval()
# # print (f'FORDOR2 debug {raw_datasets["train_eli5"][0]["answers"]} =:= {model(input_ids=my_dataset["train_eli5"][0]["input_ids"].unsqueeze(0).cuda(), attention_mask=my_dataset["train_eli5"][0]["attention_mask"].unsqueeze(0).cuda(), token_type_ids=my_dataset["train_eli5"][0]["token_type_ids"].unsqueeze(0).cuda())}')
# # print (f'FORDOR3 debug {raw_datasets["train_eli5"][0]["answers"]} =:= {model(input_ids=my_dataset["train_eli5"][1]["input_ids"].unsqueeze(0).cuda(), attention_mask=my_dataset["train_eli5"][1]["attention_mask"].unsqueeze(0).cuda(), token_type_ids=my_dataset["train_eli5"][1]["token_type_ids"].unsqueeze(0).cuda())}')
# # print (f'FORDOR4 debug {raw_datasets["train_eli5"][1]["answers"]} =:= {model(input_ids=my_dataset["train_eli5"][4]["input_ids"].unsqueeze(0).cuda(), attention_mask=my_dataset["train_eli5"][4]["attention_mask"].unsqueeze(0).cuda(), token_type_ids=my_dataset["train_eli5"][4]["token_type_ids"].unsqueeze(0).cuda())}')


# print ("evaluation starting")
# print (trainer.evaluate())
