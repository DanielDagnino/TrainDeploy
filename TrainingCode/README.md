# TrainDeploy project

## Introduction


## Installation
* Data collection/generation environment:
```bash
cd <PATH_TO_TrainDeploy>/TrainingCode
mkvirtualenv DataGeneration -p python3.8.19
python -m pip install --upgrade pip
pip install --no-deps -r requirements_data.txt
export PYTHONPATH=.:submodules/bark-with-voice-clone

python -m spacy download en_core_web_sm
```

* Training environment:
```bash
cd <PATH_TO_TrainDeploy>/TrainingCode
mkvirtualenv TrainDeploy -p python3.10
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=.:submodules/bark-with-voice-clone
```

## Data collection and generation
Download `LibriSpeech`:

Create a list of sentences with a minimum complexity and length.
```bash
python 1-select_sentences.py
```

Synthetic voice generator based on Bark. It uses the default Bark's voices.
```bash
python 3-gen_voices.py
```

Generator of the Bark codecs needed to generate custom voices.
```bash
python 4-codecs_to_clone_voices.py
```

Generate custom voices.
```bash
python 5-clone_voices.py
```

## Train a model with the created dataset
```bash
python cli/train_clf.py "cfg/train_clf.yaml"
```
