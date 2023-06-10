import json
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
import torch
import faiss
import os
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



coco_data_path = 'datastore/index3.json' # path to Karpathy splits downloaded from Kaggle
image_path = ''

#read in part of the data
# Read the file
with open(coco_data_path, 'r') as f:
    data = json.load(f)
#first 10000 and last 4665
data['images'] = data['images'][:10000] + data['images'][-4665:]
print(len(data['images']))

# Truncate "raw", "realcap", and "tokens" fields to at most 50 tokens
count = 0
count1 = 0
for image in list(data['images']):   
    count += 1
    #if image contains nan remove it
    if image['raw'] == 'nan':
        data['images'].remove(image)
    else:
        count1 += 1
        image['raw'] = ' '.join(image['tokens'][:40])
        image['realcap'] = ' '.join(image['tokens'][:40])
        image['tokens'] = image['tokens'][:40]

print(count)
print(count1)
#print new data to a json file
with open('datastore/index2.json', 'w') as f:
        json.dump(data, f)