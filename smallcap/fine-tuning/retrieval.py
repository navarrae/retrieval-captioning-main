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

def load_coco_data(coco_data_path):
    """We load in all images and only the train captions."""

    annotations = json.load(open(coco_data_path))['images']
    images = []
    captions = []
    for item in annotations:
        if item['split'] == 'train':
            for sentence in item['raw']:
                captions.append({'image_id': item['cocoid'],  'caption': ' '.join(item['tokens'])})
        images.append({'image_id': item['cocoid'], 'file_name': item['filename'].split('_')[-1]})
 
    return images, captions


def filter_captions(data):
    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    bs = 512
    max_length = 50

    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    filtered_caps = []

    # Truncate captions to 77 tokens
    for cap in caps:
        #if len(cap.split()) < max_length:
            #filtered_caps.append(cap)
        filtered_caps.append(" ".join(cap.split()[:55]))

    encodings = []

    for idx in range(0, len(filtered_caps), bs):
        batch_encodings = tokenizer.batch_encode_plus(
            filtered_caps[idx:idx+bs],
            return_tensors='np', 
            padding=True, 
            truncation=True,
        )['input_ids'].tolist()
        encodings += batch_encodings

    filtered_image_ids, filtered_captions = [], []
    assert len(image_ids) == len(caps) and len(caps) == len(encodings)
    for image_id in zip(image_ids, caps, encodings):

        filtered_image_ids.append(image_id)
    return filtered_image_ids, filtered_caps

'''
def filter_captions(data):
    decoder_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(decoder_name)
    bs = 512
    
    image_ids = [d['image_id'] for d in data]
    caps = [d['caption'] for d in data]
    encodings = []
    for idx in range(0, len(data), bs):
        batch_encodings = tokenizer.batch_encode_plus(caps[idx:idx+bs], return_tensors='np')['input_ids']
        for encoding in batch_encodings:
            if len(encoding) <= 25:
                encodings.append(encoding)
    
    filtered_image_ids, filtered_captions = [], []

    assert len(image_ids) == len(caps)
    for image_id, cap, encoding in zip(image_ids, caps, encodings):
        if len(encoding) <= 25:
            filtered_image_ids.append(image_id)
            filtered_captions.append(cap)

    return filtered_image_ids, filtered_captions
'''
def encode_captions(captions, model, device):

    bs = 256
    encoded_captions = []
    print(f"Number of captions: {len(captions)}")

    for idx in tqdm(range(0, len(captions), bs)):
        print(f"Processing batch {idx//bs}")
        with torch.no_grad():
            input_ids = clip.tokenize(captions[idx:idx+bs]).to(device)
            encoded_captions.append(model.encode_text(input_ids).cpu().numpy())


    encoded_captions = np.concatenate(encoded_captions)

    return encoded_captions

def encode_images(images, image_path, model, feature_extractor, device):

    image_ids = [i['image_id'] for i in images]
    
    bs = 64	
    image_features = []
    count = 0
    for idx in tqdm(range(0, len(images), bs)):
        #image_input = [feature_extractor(Image.open(os.path.join(image_path, i['file_name'])))
                                                                    #for i in images[idx:idx+bs]]
        for i in images[idx:idx+bs]:
            if os.path.exists(os.path.join(image_path, i['file_name'])) is False:
                print(os.path.join(image_path, i['file_name']))
                count += 1

        image_input = [feature_extractor(Image.open(os.path.join(image_path, i['file_name'])))
               for i in images[idx:idx+bs] if os.path.exists(os.path.join(image_path, i['file_name']))]
        with torch.no_grad():
            image_features.append(model.encode_image(torch.tensor(np.stack(image_input)).to(device)).cpu().numpy())

    image_features = np.concatenate(image_features)

    return image_ids, image_features, count

'''
def get_nns(captions, images, k=15):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 

    return index, I
'''
def get_nns(captions, images, k=5, batch_size=10):
    xq = images.astype(np.float32)
    xb = captions.astype(np.float32)
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    
    # add in smaller batches
    for i in range(0, xb.shape[0], batch_size):
        index.add(xb[i : min(i + batch_size, xb.shape[0])])
        
    faiss.normalize_L2(xq)
    D, I = index.search(xq, k) 

    return index, I

def filter_nns(nns, xb_image_ids, captions, xq_image_ids):
    """ We filter out nearest neighbors which are actual captions for the query image, keeping 7 neighbors per image."""
    retrieved_captions = {}
    for nns_list, image_id in zip(nns, xq_image_ids):
        good_nns = []
        for nn in nns_list:
            if xb_image_ids[nn] == image_id:
                continue
            good_nns.append(captions[nn])
            if len(good_nns) == 7:
                break
        #assert len(good_nns) == 7
        retrieved_captions[image_id] = good_nns
    return retrieved_captions
 
def main(): 

    coco_data_path = 'datastore/index.json' # path to Karpathy splits downloaded from Kaggle
    image_path = ''

    #read in part of the data
    # Read the file
    with open(coco_data_path, 'r') as f:
        data = json.load(f)
        #data2 = json.load(f)

    # Truncate "raw", "realcap", and "tokens" fields to at most 50 tokens
    # if the raw is 'nan', get rid of that image in data

    '''
    count  =0
    for image in list(data['images']):
        if image['raw'] == 'nan':
            data['images'].remove(image)
            count +=1
            continue
        image['raw'] = ' '.join(image['tokens'][:40])
        image['realcap'] = ' '.join(image['tokens'][:40])
        image['tokens'] = image['tokens'][:40]
    print(count)

    #print new data to a json file
    with open('datastore/index.json', 'w') as f:
        json.dump(data, f)
    
    '''
    coco_data_path = 'datastore/index.json'
    print('Loading data')
    
    # Apply load_coco_data() function to the first 1000 images
    images, captions = load_coco_data(coco_data_path)

    #images, captions = load_coco_data(coco_data_path)
    print(f"Number of images: {len(images)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, feature_extractor = clip.load("RN50x64", device=device)

    print('Filtering captions')    
    xb_image_ids, captions = filter_captions(captions)

    print('Encoding captions')
    encoded_captions = encode_captions(captions, clip_model, device)
    
    print('Encoding images')
    xq_image_ids, encoded_images, count = encode_images(images, image_path, clip_model, feature_extractor, device)
    print(count)
    print('Retrieving neighbors')
    index, nns = get_nns(encoded_captions, encoded_images)
    retrieved_caps = filter_nns(nns, xb_image_ids, captions, xq_image_ids)

    print('Writing files')
    faiss.write_index(index, "datastore/coco_index")
    json.dump(captions, open('datastore/coco_index_captions.json', 'w'))

    json.dump(retrieved_caps, open('data/retrieved_caps_resnet50x64.json', 'w'))

if __name__ == '__main__':
    main()




    

