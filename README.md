# SmallCap

## Dependencies

The code was developed in Python 3.9.

```
conda create -n smallcap python=3.9
conda activate smallcap
pip install -r requirements.txt
```

## Training SmallCap

<details>
<summary>Click to expand</summary>

### Preprocessing

At the moment CLIP models based on ResNet are not available through HuggingFace so it is necessary to also install the original CLIP implementation from [here](https://github.com/openai/CLIP):

```
pip install git+https://github.com/openai/CLIP.git
```

Get Data into right format:

```
Mount image files onto ~/Image
cd retrieval-captioning-main/smallcap/fine-tuning
python3 processing.py
python3 makejson.py
```

Extract train and val features: 

```
mkdir features
python3 extract.py
```

Retrieve captions

```python3 retrieval.py```

### Model training

```Run fine-tune.ipynb to completion```

Models are saved under name <rag/norag>_<num params>M, e.g. `rag_7M` for a model trained with retrieval augmentation and 7M trainable parameters.

### Inference

```python infer.py --model_path <MODEL_PATH>```

If you also specify `--checkpoint_path` inference runs with only that checkpoint. Else, all checkpoints in `--model_path` are used. 

If you specify `--infer_test` inference uses test data, else val data is used.

E.g. to run inference on the test split with model `rag_7M`, checkpoint `17712`, run

```python infer.py --model_path experiments/rag_7M --checkpoint_path checkpoint-17712 --infer_test```

The model predictions are stored as ```<val/test>_preds.json``` in each respective checkpoint subdirectory.

Note: You can safely ignore the warning `Some weights of ThisGPT2LMHeadModel were not initialized from the model checkpoint at gpt2 and are newly initialized...` It occurs because a new model is first built and then the pre-trained parameters are loaded into it. 











