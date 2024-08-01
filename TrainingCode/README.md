# TrainDeploy project

## Introduction


## Installation
* Training:
```bash
cd <PATH_TO_TrainDeploy>/TrainingCode
mkvirtualenv TrainDeploy -p python3.10
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.:submodules

python -m spacy download en_core_web_sm
```

## Data collection and generation
Steps:
1. Download `LibriSpeech`
2. Select sentences
```bash
python 1-select_sentences.py PATH_TO_LibriSpeech PATH_TO_OUTPUTS
```
3. Collect voices
```bash
python 2-join_samples_to_clone.py PATH_TO_LibriSpeech PATH_TO_OUTPUTS
```
4. 
```bash
python 3-gen_voices.py
```
5. 
```bash
python 4-codecs_to_clone_voices.py
```
6. 
```bash
python 5-clone_voices.py
```
