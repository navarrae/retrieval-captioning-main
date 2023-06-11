import os
import pandas as pd
import numpy as np
import pydicom
import json
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import pydicom

# Read in the processed csv file
df = pd.read_csv('/home/ec2-user/justin-git/retrieval-captioning/smallcap/fine-tuning/new_reports.csv')

# Define the path to the root folder containing subject ID folders
root_folder_path = '/home/ec2-user/justin-git/retrieval-captioning/Image/TRAIN'

# Define the path to the datastore directory
datastore_dir = 'datastore'

# Create an empty list to store the image-caption pairs
captions = []

# Create an empty dictionary to store the index
index = {}

# Loop through the dataframe and add the image-caption pairs to the list and index dictionary

for i, row in df.iloc[:6000].iterrows():

    id = str(row['FileName'])
    target = str(row['Simluated Report'])
    img_file_path = os.path.join(root_folder_path, f'{id}.jpg')
    captions.append(target)
    index[img_file_path] = target
        
    if i % 1000 == 0:
        print(i)

for i, row in df.iloc[6000:7000].iterrows():

    id = str(row['FileName'])
    target = str(row['Simluated Report'])
    img_file_path = os.path.join(root_folder_path, f'{id}.jpg')
    captions.append(target)
    index[img_file_path] = target
        
    captions.append(target + 'validationset') 
    index[img_file_path] = target + 'validationset'
    
    if i % 1000 == 0:
        print(i)
        
# Save the captions to a text file in the datastore directory
with open(os.path.join(datastore_dir, 'captions.txt'), 'w') as f:
    for caption in captions:
        f.write(caption + '\n')

# Save the index dictionary to a JSON file in the datastore directory
with open(os.path.join(datastore_dir, 'pairs.json'), 'w') as f:
    json.dump(index, f)
