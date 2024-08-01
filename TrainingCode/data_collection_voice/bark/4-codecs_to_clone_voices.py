import argparse
import sys

import numpy as np
import torch
import torchaudio
from encodec.utils import convert_audio
from path import Path
from tqdm import tqdm

from bark.generation import load_codec_model
from hubert.hubert_manager import HuBERTManager


def run(dir_voice_cloners):
    dir_clone_voices = Path("../../../submodules/bark-with-voice-clone/bark/assets/prompts/")
    dir_clone_voices = Path(dir_clone_voices)
    dir_clone_voices.makedirs_p()

    # Load model.
    model = load_codec_model(use_gpu=True)

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()

    # From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer
    # Load HuBERT for semantic tokens
    from hubert.pre_kmeans_hubert import CustomHubert
    from hubert.customtokenizer import CustomTokenizer

    # Load the HuBERT model
    hubert_model = CustomHubert(checkpoint_path='data/models/hubert/hubert.pt').cuda()

    # Load the CustomTokenizer model
    tokenizer = CustomTokenizer.load_from_checkpoint('data/models/hubert/tokenizer.pth').cuda()

    # Create codecs to clone voices.
    ds = "LibriSpeech"
    fns_voice_samples = list(Path(f"{dir_voice_cloners}/samples_from_{ds}/sample_to_clone").walkfiles("*.mp3"))

    for fn_voice_samples in tqdm(fns_voice_samples, total=len(fns_voice_samples)):
        # Load and pre-process the audio waveform
        wav, sr = torchaudio.load(fn_voice_samples)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.cuda()

        semantic_vectors = hubert_model.forward(wav, input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)
        semantic_tokens = semantic_tokens.cpu().numpy()

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = model.encode(wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        codes = codes.cpu().numpy()

        voice_id = fn_voice_samples.stem
        output_path = dir_clone_voices / f'ds={ds}-voice_id={voice_id}.npz'
        np.savez(output_path, fine_prompt=codes, coarse_prompt=codes[:2, :], semantic_prompt=semantic_tokens)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dir_voice_cloners", type=Path, help="Output directory")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
