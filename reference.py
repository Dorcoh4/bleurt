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

references, candidates, scores = preprocess_data("test_eli5")
with open('references', 'a') as the_file:
  for reference in references:
    the_file.write(f"{reference}\n")
