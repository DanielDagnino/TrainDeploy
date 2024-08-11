import json
import os
import random

import numpy as np
from path import Path

from data_collection.utils.audio.np_manipulate import is_audio_file

# ******************************************************************************************************************** #
split_name = 'split_training'
base_out = f'/home/razor/MyTmp/TrainDeploy/splits/{split_name}'
Path(f'{base_out}').makedirs_p()

# ******************************************************************************************************************** #
random.seed(1234)
np.random.seed(1234)

fix_idx_shuffle = np.arange(0, 1_000_000)
np.random.shuffle(fix_idx_shuffle)


def get_files(dirs_voices):
    fns_audio = [_ for dir_voices in dirs_voices
                 for _ in Path(dir_voices).walkfiles()
                 if is_audio_file(_)]
    return fns_audio


def voice_split_train_val(ds_dirs, idx_shuffle):
    ds_train, ds_val, ds_test = [], [], []
    for ds_dir, n_val, n_test, n_train in ds_dirs:
        dirs_voices = [_ for _ in sorted(Path(ds_dir).dirs())]
        idx_shuffle = idx_shuffle[idx_shuffle < len(dirs_voices)]
        dirs_voices = np.array(dirs_voices)
        # print(f"len(dirs_voices) = {len(dirs_voices)}")

        # val
        n_end = n_val if n_val is not None else None
        ds_val_tmp = get_files(dirs_voices[idx_shuffle[:n_end]])
        ds_val.extend(ds_val_tmp)
        if n_val is None:
            # print(f"len(ds_val_tmp) = {len(ds_val_tmp)}")
            continue
        # print(f"len(ds_val_tmp) = {len(ds_val_tmp)}")

        # test
        n_end = n_val + n_test if n_test is not None else None
        ds_test_tmp = get_files(dirs_voices[idx_shuffle[n_val:n_end]])
        ds_test.extend(ds_test_tmp)
        if n_test is None:
            # print(f"len(ds_test_tmp) = {len(ds_test_tmp)}")
            continue
        # print(f"len(ds_test_tmp) = {len(ds_test_tmp)}")

        # train
        n_end = n_train + n_val + n_test if n_train is not None else None
        ds_train_tmp = get_files(dirs_voices[idx_shuffle[(n_val + n_test):n_end]])
        ds_train.extend(ds_train_tmp)
        # print(f"len(ds_train_tmp) = {len(ds_train_tmp)}")

        # print(f"sum = {len(ds_train) + len(ds_val) + len(ds_test)}")
        # print()

    return ds_train, ds_val, ds_test


def split_train_val(ds_dirs, idx_shuffle, verbose=True):
    ds_train, ds_val, ds_test = [], [], []
    for ds_dir, n_val, n_test, n_train in ds_dirs:
        tmp = [_ for _ in sorted(Path(ds_dir).walkfiles()) if is_audio_file(_)]
        tmp = np.array(tmp)
        tmp_idx_shuffle = idx_shuffle[idx_shuffle < len(tmp)]

        # val
        n_end = n_val if n_val is not None else None
        ds_val_tmp = list(tmp[tmp_idx_shuffle[:n_end]])
        ds_val.extend(ds_val_tmp)
        if n_val is None:
            if verbose:
                print(len(tmp), 0, len(ds_val_tmp), 0, Path(ds_dir).relpath(os.path.expanduser("~")))
            continue

        # test
        n_end = n_val + n_test if n_test is not None else None
        ds_test_tmp = list(tmp[tmp_idx_shuffle[n_val:n_end]])
        ds_test.extend(ds_test_tmp)
        if n_test is None:
            if verbose:
                print(len(tmp), 0, len(ds_val_tmp), len(ds_test_tmp), Path(ds_dir).relpath(os.path.expanduser("~")))
            continue

        # train
        n_end = n_train + n_val + n_test if n_train is not None else None
        ds_train_tmp = list(tmp[tmp_idx_shuffle[(n_val + n_test):n_end]])
        ds_train.extend(ds_train_tmp)

        if verbose:
            print(len(tmp), len(ds_train_tmp), len(ds_val_tmp), len(ds_test_tmp),
                  Path(ds_dir).relpath(os.path.expanduser("~")))
    return ds_train, ds_val, ds_test


# ******************************************************************************************************************** #
# ds_dir, n_val, n_test, n_train

# bark
ai_voices_dirs = [
    ["/home/razor/MyData/voice_cloners/bark/LibriSpeech", 20, 20, None],
]
ai_voices_fns_train_bark, ai_voices_fns_val_bark, ai_voices_fns_test_bark = voice_split_train_val(
    ai_voices_dirs, fix_idx_shuffle)
print(f'len(ai_voices_fns_train_bark) = {len(ai_voices_fns_train_bark)}')
print(f'len(ai_voices_fns_val_bark) = {len(ai_voices_fns_val_bark)}')
print(f'len(ai_voices_fns_test_bark) = {len(ai_voices_fns_test_bark)}')
json.dump(ai_voices_fns_train_bark, open(f'{base_out}/ai_voices_fns_train_bark.json', 'w'), indent=4)
json.dump(ai_voices_fns_val_bark, open(f'{base_out}/ai_voices_fns_val_bark.json', 'w'), indent=4)
json.dump(ai_voices_fns_test_bark, open(f'{base_out}/ai_voices_fns_test_bark.json', 'w'), indent=4)
print()

# non_ai: LibriSpeech + Voxceleb
non_ai_voices_dirs = [
    ["/media/razor/dagnino/MyBackUpSSD/MyDataDownload/LibriSpeech/train-clean-360", 10, 10, None],
    ["/media/razor/dagnino/MyBackUpSSD/MyDataDownload/LibriSpeech/train-clean-100", 10, 10, None],
]
non_ai_voices_fns_train, non_ai_voices_fns_val, non_ai_voices_fns_test = voice_split_train_val(
    non_ai_voices_dirs, fix_idx_shuffle)
json.dump(non_ai_voices_fns_train, open(f'{base_out}/non_ai_voices_fns_train.json', 'w'), indent=4)
json.dump(non_ai_voices_fns_val, open(f'{base_out}/non_ai_voices_fns_val.json', 'w'), indent=4)
json.dump(non_ai_voices_fns_test, open(f'{base_out}/non_ai_voices_fns_test.json', 'w'), indent=4)
print(f'len(non_ai_voices_fns_train) = {len(non_ai_voices_fns_train)}')
print(f'len(non_ai_voices_fns_val) = {len(non_ai_voices_fns_val)}')
print(f'len(non_ai_voices_fns_test) = {len(non_ai_voices_fns_test)}')
print()

# ******************************************************************************************************************** #
# Augmentations: noise + speech + ...
augment_dirs = [
    ["/media/razor/dagnino/MyBackUpSSD/MyDataDownload/UrbanSound8K", 50, 50, None],
]
augment_fns_train, augment_fns_val, augment_fns_test = split_train_val(augment_dirs, fix_idx_shuffle)
json.dump(augment_fns_train, open(f'{base_out}/augment_train.json', 'w'), indent=4)
json.dump(augment_fns_val, open(f'{base_out}/augment_val.json', 'w'), indent=4)
json.dump(augment_fns_test, open(f'{base_out}/augment_test.json', 'w'), indent=4)
print(f'len(augment_fns_train) = {len(augment_fns_train)}')
print(f'len(augment_fns_val) = {len(augment_fns_val)}')
print(f'len(augment_fns_test) = {len(augment_fns_test)}')
print()
