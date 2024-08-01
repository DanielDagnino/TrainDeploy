import argparse
import json
import sys

import torch
import torchaudio
from path import Path


def run(dir_librispeech, dir_voice_cloners, subfolder):
    base_dir_out_data = Path(f"{dir_voice_cloners}")
    duration_sec_min = 120
    sr_target = 16_000

    # Check how many ids are inside
    fn_in = f"{dir_librispeech}/SPEAKERS.TXT"
    lines = open(fn_in, 'r').readlines()
    lines = [line.split('|') for line in lines if line[0] != ";"]
    ids = [[line[0].strip(), line[-1].strip()] for line in lines]
    print(f'len(ids) = {len(ids)}')

    # Get previously generated samples.
    fn_out = base_dir_out_data / f'samples_to_clone_voice-{subfolder.replace("-", "_")}.json'
    if not fn_out.exists():
        base_libri = Path(f"{dir_librispeech}/{subfolder}")
        samples = dict()
        for ddir in sorted(base_libri.dirs()):
            idd = Path(ddir).stem
            samples.setdefault(idd, dict(duration=0, fns=[]))
            for fn in sorted(ddir.walkfiles("*.flac")):
                wf, sr = torchaudio.load(fn)
                samples[idd]["duration"] += wf.shape[1] / sr
                samples[idd]["fns"].append(fn)
                if samples[idd]["duration"] > duration_sec_min:
                    print(idd, samples[idd]["duration"], len(samples[idd]["fns"]))
                    break
        json.dump(samples, open(fn_out, "w"), indent=4)

    # Join samples in a single audio file.
    samples = json.load(open(fn_out))
    dir_sample_to_clone_out = base_dir_out_data / f'sample_to_clone-{subfolder.replace("-", "_")}'
    dir_sample_to_clone_out.makedirs_p()
    n_voices = len(samples.items())
    print(f"number of voices = {n_voices}")
    for idx, (idd, sample) in enumerate(samples.items()):
        fn_out = dir_sample_to_clone_out / f"{idd}.mp3"
        if not fn_out.exists():
            fns = sample["fns"]
            wf_joined = []
            for fn in fns:
                wf, sr = torchaudio.load(fn)
                wf = wf[0].reshape(1, wf.shape[1])
                sox_effects = [["rate", str(sr_target)]]
                wf, _ = torchaudio.sox_effects.apply_effects_tensor(wf, sr, sox_effects)
                wf_joined.append(wf)
            wf_joined = torch.concat(wf_joined, dim=1)
            print(idx / n_voices, wf_joined.shape[1] / sr_target)
            torchaudio.save(fn_out, wf_joined, sample_rate=sr_target)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dir_librispeech", type=Path, help="Directory with the LibriSpeech dataset")
    parser.add_argument("dir_voice_cloners", type=Path, help="Output directory")
    parser.add_argument("--subfolder", type=Path, default="train-clean-100", help="Folder of LibriSpeech to use")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
