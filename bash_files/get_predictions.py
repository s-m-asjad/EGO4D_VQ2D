import gzip
import os
import random
import math
import json
import sys

content = "/home/asjad.s/EGO4D"


experiment_file = sys.argv[1]


with gzip.open(experiment_file, 'r') as f:
  predictions = json.loads(f.read().decode('utf-8'))
with gzip.open(content+'/episodic-memory/VQ2D/data/vq_splits/val_annot.json.gz', 'r') as f:
  split_data = json.loads(f.read().decode('utf-8'))

print(predictions)
print(split_data)