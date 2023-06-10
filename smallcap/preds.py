import csv
import json

# Open the JSON file for reading
with open('/home/ec2-user/justin-git/retrieval-captioning/smallcap/experiments/norag_7M_facebook/opt-350m/checkpoint-104085/val_preds.json') as f:
    data = json.load(f)

# Open a new CSV file for writing
with open('preds.csv', 'w', newline='') as f:
    # Create a CSV writer object and write the header row
    writer = csv.writer(f)
    writer.writerow(['subject_id', 'study_id', 'dicom_id', 'impression'])
    
    # Loop through each item in the JSON data
    for item in data:
        # Extract the image ID from the item's image ID field
        image_id = item['image_id'].split('/p')[-1]
        subject_id = image_id.split('/')[0]
        study_id = image_id.split('/')[1]
        dicom_id = str(image_id.split('/')[2])[:-4]
        # Extract the caption from the item's caption field
        caption = item['caption']
        # Write a new row to the CSV file with the image ID and caption
        writer.writerow([subject_id, study_id, dicom_id, caption])