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
df = pd.read_csv('datastore/mimic_train_reports.csv')
dh = pd.read_csv('datastore/mimic_test_reports.csv')

# Define the path to the root folder containing subject ID folders
root_folder_path = '/home/ec2-user/justin-git/efs/MIMIC-CXR-JPG/files'

# Define the path to the datastore directory
datastore_dir = 'datastore'

# Create an empty list to store the image-caption pairs
captions = []

# Create an empty dictionary to store the index
index = {}

# Loop through the dataframe and add the image-caption pairs to the list and index dictionary
for i, row in df.iloc[:].iterrows():
    dicom_id = str(row['dicom_id'])
    study_id = 's' + str(row['study_id'])
    target = str(row['impression'])
    subject_id = str(row['subject_id'])
    
    # Get the subject ID folder name from the first two digits of the subject ID
    subject_folder_name = 'p' + subject_id[:2]
    subject_id = 'p' + subject_id

    img_file_path = os.path.join(root_folder_path, subject_folder_name)
    img_file_path = os.path.join(img_file_path, subject_id)
    img_file_path = os.path.join(img_file_path, study_id)
    img_file_path = os.path.join(img_file_path, f'{dicom_id}.jpg')

    captions.append(target)
    index[img_file_path] = target

    if i % 1000 == 0:
        print(i)

for i, row in dh.iloc[:].iterrows():
    dicom_id = str(row['dicom_id'])
    study_id = 's' + str(row['study_id'])
    target = str(row['impression'])
    subject_id = str(row['subject_id'])
    
    # Get the subject ID folder name from the first two digits of the subject ID
    subject_folder_name = 'p' + subject_id[:2]
    subject_id = 'p' + subject_id

    img_file_path = os.path.join(root_folder_path, subject_folder_name)
    img_file_path = os.path.join(img_file_path, subject_id)
    img_file_path = os.path.join(img_file_path, study_id)
    img_file_path = os.path.join(img_file_path, f'{dicom_id}.jpg')

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
