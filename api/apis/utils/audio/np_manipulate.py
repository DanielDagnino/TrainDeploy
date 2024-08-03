import numpy as np
from path import Path


def trim_zeros(filt, trim='fb'):
    assert filt.ndim == 2
    mono_filt = filt.sum(0)
    trim = trim.upper()
    first = 0
    if 'F' in trim:
        first = np.argmax(mono_filt != 0)
    last = len(mono_filt)
    if 'B' in trim:
        mono_filt = np.flip(mono_filt, axis=[0])
        last = last - np.argmax(mono_filt != 0)
    return filt[:, first:last]


def is_audio_file(fn):
    if Path(fn).ext in [".wav", ".mp3", ".ogg", ".flv", ".mp4", ".wma", ".aac"]:
        return True
    else:
        return False


def keep_human_freq(sample_rate, audio_data):
    # Define the frequency range for human speech
    min_frequency = 80  # Minimum frequency in Hz
    max_frequency = 3000  # Maximum frequency in Hz

    # Compute the Discrete Fourier Transform (DFT)
    dft = np.fft.fft(audio_data)

    # Compute the frequencies corresponding to the DFT bins
    frequencies = np.fft.fftfreq(len(audio_data)) * sample_rate

    # Create a mask to filter out frequencies outside the speech range
    mask = np.logical_and(frequencies >= min_frequency, frequencies <= max_frequency)

    # Apply the mask to the DFT coefficients
    filtered_dft = dft * mask

    # Perform the inverse Discrete Fourier Transform (iDFT) to obtain the filtered audio data
    filtered_audio_data = np.fft.ifft(filtered_dft).real.astype(float)
    return filtered_audio_data


def calculate_snr_in_human_freq(speech, noise, sample_rate):
    speech = keep_human_freq(sample_rate, speech)

    power_speech = np.mean(np.square(speech))
    power_noise = np.mean(np.square(noise))

    snr_db = 10 * np.log10(power_speech / power_noise)

    return snr_db, power_speech, power_noise


def join_songs(audio_data1, audio_data2, chn_first=True):
    if chn_first:
        min_length = min(len(audio_data1), len(audio_data2))
        audio_data1 = audio_data1[:min_length]
        audio_data2 = audio_data2[:min_length]
    else:
        min_length = min(audio_data1.shape[-1], audio_data2.shape[-1])
        audio_data1 = audio_data1[:, :min_length]
        audio_data2 = audio_data2[:, :min_length]
    return audio_data1 + audio_data2
