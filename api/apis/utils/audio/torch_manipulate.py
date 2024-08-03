import numpy as np
import torch
import torchaudio


def trim_zeros(filt, trim='fb'):
    assert filt.ndim == 2
    mono_filt = filt.sum(0)
    trim = trim.upper()
    first = 0
    if 'F' in trim:
        first = torch.argmax((mono_filt != 0).long())
    last = len(mono_filt)
    if 'B' in trim:
        mono_filt = torch.flip(mono_filt, dims=[0])
        last = last - torch.argmax((mono_filt != 0).long())
    return filt[:, first:last]


def read_to_waveform(fn, sr=None, mono=False, duration_min=None, duration_max=None, random_clip=True, trim=True,
                     pad_type="repeat", dtype=None):
    waveform, sample_rate = torchaudio.load(fn)
    # Remove start-ending zeros
    if trim:
        waveform = trim_zeros(waveform, trim='fb')
    if duration_max:
        iduration = int(sample_rate * duration_max)
        if waveform.shape[-1] >= iduration:
            idx0 = np.random.randint(0, waveform.shape[-1] - iduration) if random_clip else 0
            waveform = waveform[:, idx0:(idx0 + iduration)]
    if duration_min:
        iduration = int(sample_rate * duration_min)
        if waveform.shape[-1] < iduration:
            if pad_type == "repeat":
                while waveform.shape[-1] < iduration:
                    waveform = torch.cat([waveform, waveform], dim=1)[:, :iduration]
            elif pad_type == "pad":
                waveform = torch.nn.functional.pad(waveform, (0, iduration-waveform.shape[-1]), "constant", 0)
            else:
                raise ValueError(f"Not accepted padding type: pad_type = {pad_type}")
    if mono:
        waveform = waveform[0]
    if sr:
        waveform = torchaudio.transforms.Resample(
            sample_rate, sr,
            resampling_method="sinc_interp_hann", lowpass_filter_width=6, rolloff=0.99
        )(waveform)
    else:
        sr = sample_rate
    if dtype:
        waveform = waveform.to(dtype)
    return sr, waveform
