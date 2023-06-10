import json
import nltk
import random

nltk.download('punkt')

# Define paths
json_path = 'datastore/pairs.json' # Path to input JSON file
output_path = 'datastore/index.json' # Path to output JSON file

# Load JSON file
with open(json_path, 'r') as f:
    data = json.load(f)

# Tokenize captions and create a list of image-caption pairs
pairs = []
count = 0
for key, caption in data.items():
    count += 1
    image_path = key 
    captioncopy = caption[:]
    print(caption[-13:])
    if caption[-13:] == 'validationset':
        caption = caption[:-13]
    tokens = nltk.word_tokenize(caption)
    pairs.append({
        'file_path': image_path,
        'raw': caption,
        'tokens': tokens,
        'filename': key,
        'cocoid': image_path,
        'realcap': captioncopy,
        'split': ''
    })
    if count % 10000 == 0:
        print(count)

for i in range(len(pairs)):
    if pairs[i]['realcap'][-13:] == 'validationset':
        pairs[i]['split'] = 'val'
    else:
        pairs[i]['split'] = 'train'
'''
split_idx = int(len(pairs) * 0.7)
for i in range(len(pairs)):
    if i < split_idx:
        pairs[i]['split'] = 'train'
    else:
        pairs[i]['split'] = 'val'
'''

# Create the output dictionary
output_dict = {
    'images': pairs
}

# Write the output file
with open(output_path, 'w') as f:
    json.dump(output_dict, f, indent=4)