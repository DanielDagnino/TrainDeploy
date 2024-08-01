import argparse
import datetime
import hashlib
import json
import random
import sys
from copy import deepcopy

import numpy as np
import torch
import torchaudio
from path import Path

from bark import generate_audio, preload_models, SAMPLE_RATE
from bark.api import semantic_to_waveform
from bark.generation import generate_text_semantic


def run(fn_sentences, bark_model_dir, base_dir_out):
    # Load models.
    use_small = False
    preload_models(text_use_small=use_small, coarse_use_small=use_small, fine_use_small=use_small, path=bark_model_dir)

    # Get sentences.
    sentences = json.load(open(fn_sentences))
    sentences = [_["utterance"] for _ in sentences]
    print(f'len(sentences) = {len(sentences)}')

    # Get voice ids.
    dir_voices = Path("../../../submodules/bark-with-voice-clone/bark/assets/prompts_custom/")
    voice_ids = list(Path(_).stem for _ in dir_voices.files("*.npz"))
    print(f"len(voice_ids) = {len(voice_ids)}")

    # Generate audio with the clone voices.
    model_id = "bark-preload_models-use_small=False-20231221"

    base_history_prompt = Path('../../../submodules/bark-with-voice-clone/bark/assets/prompts_custom')

    n_samples_per_voice = 2
    base_dir_out = Path(base_dir_out)
    min_eos_p = None
    cp_sentences = []
    for _ in range(n_samples_per_voice):
        random.shuffle(sentences)
        for voice_id in voice_ids:
            if not cp_sentences:
                cp_sentences = deepcopy(sentences)
            sentence = cp_sentences.pop()
            sentence = sentence.strip() + "."

            history_prompt = base_history_prompt / voice_id
            if min_eos_p is None:
                audio_array = generate_audio(sentence, history_prompt=history_prompt, text_temp=0.7, waveform_temp=0.7,
                                             silent=False)
            else:
                semantic_token = generate_text_semantic(sentence, history_prompt=history_prompt, temp=0.7,
                                                        min_eos_p=0.2)
                audio_array = semantic_to_waveform(semantic_token, history_prompt=history_prompt, temp=0.7)

            audio_array = np.concatenate([audio_array, np.zeros(int(0.25 * SAMPLE_RATE), dtype=audio_array.dtype)])
            audio_array = torch.tensor(audio_array).reshape(1, -1)
            duration = audio_array.shape[1] / SAMPLE_RATE

            dir_out = base_dir_out / voice_id
            dir_out.makedirs_p()
            idd = hashlib.sha256((voice_id + "---" + model_id + "---" + sentence).encode()).hexdigest()
            fn_out = dir_out / f"{idd}.mp3"
            torchaudio.save(fn_out, audio_array, sample_rate=SAMPLE_RATE)
            data = {"utterance": sentence, "dataset": "LibriSpeech", "voice_id": voice_id, "model": model_id,
                    "audio_saver": "torchaudio.save", "sr": SAMPLE_RATE, "duration_sec": duration,
                    "date": f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"}
            json.dump(data, open(fn_out.splitext()[0] + ".json", 'w'))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("bark_model_dir", type=Path, help="Directory with the bark model")
    parser.add_argument("fn_sentences", type=Path, help="File containing the sentences")
    parser.add_argument("base_dir_out", type=Path, help="Directory outputs")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
