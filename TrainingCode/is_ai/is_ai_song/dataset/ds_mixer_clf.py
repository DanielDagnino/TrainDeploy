#!/usr/bin/env python
import gc
import inspect
import json
import logging
import os
import random
from copy import copy
from typing import Tuple, List, Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from path import Path
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from audiocraft.data.audio import audio_write
from data_collection.utils.audio.torch_manipulate import read_to_waveform
# from is_ai_song.dataset.transf import RandomBackgroundNoise   #, RandomSpeedChange
from is_ai.is_ai_song.dataset.transf_base import AllSequential
from is_ai.is_ai_song.dataset.transf_mixers import SplitSegment
from is_ai.is_ai_song.dataset.utils import check_mono_same_shape
from is_ai.is_ai_song.dataset.utils import torch_join_segments
from is_ai.utils.logger.logger import setup_logging


def build_transforms(stage, sample_rate, segment_length_sec, hop_length_sec, n_segments_per_sample, noise_paths, speed
                     ) -> Tuple[Callable, Callable]:
    if stage in ["valid", "test"]:
        transform = [
            # RandomTimeShift(sample_rate=sample_rate, shift_sec=hop_length_sec / 2, prob=0.5),
        ]
    elif stage == "train":
        transform = [
            # RandomBackgroundNoise(
            #     sample_rate=sample_rate, augment_files=noise_paths, min_snr_db=2, max_snr_db=20, apply_both=True,
            #     prob=0.10),
            # RandomSpeedChange(sample_rate=sample_rate, pert_speed=speed, apply_both=True, prob=0.10),
            # RandomTimeShift(sample_rate=sample_rate, shift_sec=hop_length_sec / 2, prob=1),
            # RandomPitchShift(sample_rate=sample_rate, factor_mix_db=12, prob=0.1),
        ]
    else:
        raise ValueError(f"Not accepted stage = {stage}")
    transform = AllSequential(transform)

    segment_length = int(sample_rate * segment_length_sec)
    hop_length = int(sample_rate * hop_length_sec)
    chunker = [
        SplitSegment(segment_length=segment_length, hop_length=hop_length, n_segments_per_sample=n_segments_per_sample),
    ]
    chunker = AllSequential(chunker)

    return transform, chunker


class DatasetAudioMixer(Dataset):
    def __init__(self,
                 stage: str,
                 base_dir: str,
                 ai_song_paths: str,
                 non_ai_song_paths: str,
                 noise_paths: str,

                 sample_rate: int,
                 audio_length_sec: float,
                 audio_length_sec_max: float,
                 segment_length_sec: float,
                 hop_length_sec: float,
                 n_segments_per_sample: int,
                 speed: float,
                 n_tries_max: int = 20,
                 n_tries_read_max: int = 5,

                 rank: int = 0,
                 fn_error_stem: str = "files_error",
                 fn_doing_stem: str = "files_doing",
                 debug: str = None,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.base_dir = Path(base_dir)
        self.rank = rank
        self.debug = debug
        self.verbose = verbose

        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.segment_length_sec = segment_length_sec
        self.hop_length_sec = hop_length_sec
        self.n_segments_per_sample = n_segments_per_sample
        self.speed = speed
        self.duration_speed_min_sec = (1 + self.speed) * self.audio_length_sec
        self.audio_length_sec_max = audio_length_sec_max
        self.n_tries_max = n_tries_max
        self.n_tries_read_max = n_tries_read_max

        self.segment_length = int(self.sample_rate * self.segment_length_sec)
        self.hop_length = int(self.sample_rate * self.hop_length_sec)

        # Get audio reference files.
        self.ai_song_paths = json.load(open(ai_song_paths))
        self.logger.info(f'no omit - len(self.ai_song_paths) = {len(self.ai_song_paths)}')
        self.ai_song_paths = self.omit_large_files(self.ai_song_paths, size_mb=20)
        if stage == "train":
            assert len(self.ai_song_paths) > 1000
        random.shuffle(self.ai_song_paths)
        self.logger.info(f'len(self.ai_song_paths) = {len(self.ai_song_paths)}')

        self.non_ai_song_paths = json.load(open(non_ai_song_paths))
        self.logger.info(f'no omit - len(self.non_ai_song_paths) = {len(self.non_ai_song_paths)}')
        self.non_ai_song_paths = self.omit_large_files(self.non_ai_song_paths, size_mb=20)
        if stage == "train":
            assert len(self.non_ai_song_paths) > 1000
        random.shuffle(self.non_ai_song_paths)
        self.copy_non_ai_song_paths = copy(self.non_ai_song_paths)
        self.logger.info(f'len(self.non_ai_song_paths) = {len(self.non_ai_song_paths)}')

        self.same_length = len(self.ai_song_paths) == len(self.non_ai_song_paths)

        #
        self.non_ai_song_paths = np.array(self.non_ai_song_paths)
        self.copy_non_ai_song_paths = np.array(self.copy_non_ai_song_paths)

        # Get audio files for augmentations.
        self.noise_paths = json.load(open(noise_paths))
        self.noise_paths = self.omit_large_files(self.noise_paths, size_mb=10)
        random.shuffle(self.noise_paths)
        self.logger.info(f'len(self.noise_paths) = {len(self.noise_paths)}')

        # Clean and save.
        gc.collect()

        self.logger.info('build_transforms')
        self.transform, self.chunker = build_transforms(
            stage, self.sample_rate, self.segment_length_sec, self.hop_length_sec, self.n_segments_per_sample,
            self.noise_paths, self.speed)

        self.fn_error_stem = fn_error_stem
        self.fn_doing_stem = fn_doing_stem

        self.logger.info('DatasetAudioMixer initiated')

    def omit_large_files(self, paths, size_mb=20):
        paths = [_ for _ in paths if os.path.getsize(_) / 1024 ** 2 < size_mb]
        return paths

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        if self.debug is None:
            return len(self.ai_song_paths)
        else:
            return 24

    def __getitem__(self, idx0) -> Tuple[str, str, Tensor, Tensor, int, int]:
        fn_error = open(f'{self.fn_error_stem}_{self.rank}.txt', 'a')
        fn_doing = open(f'{self.fn_doing_stem}_{self.rank}.txt', 'a')
        path0, audio0 = "None", None
        path1, audio1 = "None", None
        success, n_tries = False, 0
        n_tries_read0, n_tries_read1 = 0, 0

        while (not success) and (n_tries <= self.n_tries_max):
            try:
                success_read0 = False
                while (not success_read0) and (n_tries_read0 <= self.n_tries_read_max):
                    path0 = self.ai_song_paths[idx0] if n_tries_read0 == 0 else random.choice(self.ai_song_paths)
                    try:
                        fn_doing.write(f'{path0}\n')
                        sr, audio0 = read_to_waveform(
                            path0, mono=True, sr=self.sample_rate, dtype=torch.float32,
                            duration_min=self.duration_speed_min_sec, duration_max=self.audio_length_sec_max)
                        success_read0 = True
                    except Exception as excpt:
                        msg = f'Some error occurred reading file: {path0}'
                        fn_error.write(f'{path0}\n')
                        self.logger.error(msg)
                        self.logger.error(excpt)
                        n_tries_read0 += 1

                success_read1 = False
                while (not success_read1) and (n_tries_read1 <= self.n_tries_read_max):
                    if self.same_length:
                        path1 = self.non_ai_song_paths[idx0] \
                            if n_tries_read1 == 0 else random.choice(self.non_ai_song_paths)
                    else:
                        if len(self.non_ai_song_paths) == 0:
                            self.non_ai_song_paths = np.copy(self.copy_non_ai_song_paths)
                        idx1 = np.random.randint(0, len(self.non_ai_song_paths))
                        path1 = self.non_ai_song_paths[idx1]
                        self.non_ai_song_paths = np.delete(self.non_ai_song_paths, idx1, 0)

                    try:
                        fn_doing.write(f'{path1}\n')
                        sr, audio1 = read_to_waveform(
                            path1, mono=True, sr=self.sample_rate, dtype=torch.float32,
                            duration_min=self.duration_speed_min_sec, duration_max=self.audio_length_sec_max)
                        success_read1 = True
                    except Exception as excpt:
                        msg = f'Some error occurred reading file: {path1}'
                        fn_error.write(f'{path1}\n')
                        self.logger.error(msg)
                        self.logger.error(excpt)
                        n_tries_read1 += 1

                # Random pick segment.
                iduration = int(self.duration_speed_min_sec * self.sample_rate)
                if len(audio0) > iduration:
                    idx = np.random.randint(0, len(audio0) - iduration)
                    audio0 = audio0[idx:(idx + iduration)]
                if len(audio1) > iduration:
                    idx = np.random.randint(0, len(audio1) - iduration)
                    audio1 = audio1[idx:(idx + iduration)]

                if audio0 is None or audio1 is None:
                    msg = f'It was not possible to load any data.'
                    self.logger.error(msg)
                    raise RuntimeError(msg)

                # checks
                sum0 = audio0.abs().sum()
                if sum0 == 0 or torch.isnan(sum0):
                    self.logger.error(f"1a - Found audio with all-zero/nan samples:")
                    self.logger.error(f"     audio0.abs().sum() = {audio0.abs().sum()}")
                    self.logger.error(f"     path0 = {path0}")
                    fn_error.write(f'{path0}\n')
                    n_tries += 1
                    n_tries_read0 += 1
                    continue

                sum1 = audio1.abs().sum()
                if sum1 == 0 or torch.isnan(sum1):
                    self.logger.error(f"1b - Found audio with all-zero/nan samples:")
                    self.logger.error(f"     audio1.abs().sum() = {audio1.abs().sum()}")
                    self.logger.error(f"     path1 = {path1}")
                    fn_error.write(f'{path1}\n')
                    n_tries += 1
                    n_tries_read1 += 1
                    continue

                # transform + chunk
                check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
                _, audio0, audio1 = self.transform(None, audio0, audio1)
                _, _, audio0, audio1 = self.chunker(None, None, audio0, audio1)

                # checks
                sum0 = audio0.abs().sum()
                if sum0 == 0 or torch.isnan(sum0):
                    self.logger.error(f"2a - Found audio with all-zero/nan samples:")
                    self.logger.error(f"     audio0.abs().sum() = {audio0.abs().sum()}")
                    self.logger.error(f"     path0 = {path0}")
                    fn_error.write(f'{path0}\n')
                    n_tries += 1
                    continue

                sum1 = audio1.abs().sum()
                if sum1 == 0 or torch.isnan(sum1):
                    self.logger.error(f"2b - Found audio with all-zero/nan samples:")
                    self.logger.error(f"     audio1.abs().sum() = {audio1.abs().sum()}")
                    self.logger.error(f"     path1 = {path1}")
                    fn_error.write(f'{path1}\n')
                    n_tries += 1
                    continue

                success = True

            except Exception as excpt:
                n_tries += 1
                self.logger.error(excpt)

        if not success:
            msg = f'Some error occurred reading/transform files'
            self.logger.error(msg)
            raise RuntimeError(msg)

        return path0, path1, audio0, audio1, 0, 1

    def collate_fn(self, batch) -> Tuple[List[str], List[str], Tensor, Tensor, Tensor, Tensor]:
        s_path0, s_path1, s_audio0, s_audio1, s_cls0, s_cls1 = [], [], [], [], [], []
        for (path0, path1, audio0, audio1, cls0, cls1) in batch:
            s_path0.append(path0)
            s_path1.append(path1)
            s_audio0.append(audio0)
            s_audio1.append(audio1)
            s_cls0.append(torch.tensor(cls0))
            s_cls1.append(torch.tensor(cls1))
        s_audio0 = torch.cat(s_audio0, dim=0)
        s_audio1 = torch.cat(s_audio1, dim=0)
        s_cls0 = torch.stack(s_cls0, dim=0)
        s_cls1 = torch.stack(s_cls1, dim=0)
        return s_path0, s_path1, s_audio0, s_audio1, s_cls0, s_cls1


def main_worker_for_test(this_gpu, n_gpus_per_node):
    _base = '/home/razor'
    n_segments_per_sample = 1
    ds = DatasetAudioMixer(
        stage="train",
        base_dir='/home/razor/MyData',
        non_ai_song_paths='/home/razor/MyTmp/DeepFakeDetector/splits/clf/split_prod/non_ai_song_fns_train.json',
        ai_song_paths='/home/razor/MyTmp/DeepFakeDetector/splits/clf/split_prod/ai_song_fns_train.json',
        noise_paths='/home/razor/MyTmp/DeepFakeDetector/splits/clf/split_prod/augment_train.json',
        sample_rate=16_000,
        speed=0.1,
        audio_length_sec_max=300,
        audio_length_sec=4.5,
        segment_length_sec=4,
        hop_length_sec=0.5,
        n_segments_per_sample=n_segments_per_sample,
        rank=0,
        debug="full",
        verbose=True
    )

    # Test dataset
    for _idx, (_path0, _path1, _segs0, _segs1, _cls0, _cls1) in enumerate(ds):
        _wav0 = torch_join_segments(_segs0, ib0=0, ib1=n_segments_per_sample)
        _wav1 = torch_join_segments(_segs1, ib0=0, ib1=n_segments_per_sample)
        Path("/home/razor/MyTmp/tmp_ds").makedirs_p()
        audio_write(f'/home/razor/MyTmp/tmp_ds/{Path(_path0).stem}-path0', _wav0.cpu(), ds.sample_rate,
                    strategy="loudness")
        audio_write(f'/home/razor/MyTmp/tmp_ds/{Path(_path1).stem}-path1', _wav1.cpu(), ds.sample_rate,
                    strategy="loudness")
        print(_segs0.shape, _cls0, _wav0.mean(), _wav0.std(), _path0)
        print(_segs1.shape, _cls1, _wav1.mean(), _wav1.std(), _path1)
        print()
        if _idx >= 20:
            break

    # Test loader
    world_size = 1
    node_rank = 0
    world_size *= n_gpus_per_node
    rank = node_rank * n_gpus_per_node + this_gpu
    dist.init_process_group("nccl", init_method='tcp://127.0.0.1:1234', rank=rank, world_size=world_size)

    loader = DataLoader(
        ds, batch_size=12, num_workers=12, pin_memory=True, drop_last=True, shuffle=False,
        sampler=torch.utils.data.distributed.DistributedSampler(ds),
        collate_fn=ds.collate_fn)

    for it, (_path0, _path1, _segs0, _segs1, _cls0, _cls1) in enumerate(loader):
        print(_segs0.shape, len(_path0), _path0[0])
        print(_segs1.shape, len(_path1), _path1[0])
        _wav0 = torch_join_segments(_segs0, ib0=0, ib1=n_segments_per_sample)
        _wav1 = torch_join_segments(_segs1, ib0=0, ib1=n_segments_per_sample)
        print(_wav0.shape)
        print(_wav1.shape)
        print()
        Path("/home/razor/MyTmp/tmp_dl").makedirs_p()
        audio_write(f'/home/razor/MyTmp/tmp_dl/{Path(_path0[0]).stem}-path0', _wav0.cpu(), ds.sample_rate,
                    strategy="loudness")
        audio_write(f'/home/razor/MyTmp/tmp_dl/{Path(_path1[0]).stem}-path1', _wav1.cpu(), ds.sample_rate,
                    strategy="loudness")

    dist.destroy_process_group()


if __name__ == '__main__':
    setup_logging()

    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)

    n_gpus_per_node = torch.cuda.device_count()
    print(f'n_gpus_per_node = {n_gpus_per_node}')
    mp.spawn(main_worker_for_test, nprocs=n_gpus_per_node, args=(n_gpus_per_node,))
