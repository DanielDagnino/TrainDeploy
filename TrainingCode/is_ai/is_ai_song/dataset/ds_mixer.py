#!/usr/bin/env python
import gc
import inspect
import json
import logging
import math
import random
from typing import Callable, Tuple, List

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
from is_ai.is_ai_song.dataset.transf import RandomBackgroundNoise, RandomSpeedChange, RandomTimeShift
from is_ai.is_ai_song.dataset.transf_base import AllSequential
from is_ai.is_ai_song.dataset.transf_mixers import TestMixer, SplitSegment, OriginalVocalMixOriginalInstruments, \
    OriginalVocalMixGeneratedInstruments, GeneratedVocalMixOriginalInstruments, GeneratedVocalMixGeneratedInstruments
from is_ai.is_ai_song.dataset.utils import Mix
from is_ai.is_ai_song.dataset.utils import MusicBank
from is_ai.is_ai_song.dataset.utils import check_mono_same_shape
from is_ai.is_ai_song.dataset.utils import torch_join_segments
from is_ai.utils.logger.logger import setup_logging


def build_transforms(stage, sample_rate, segment_length_sec, hop_length_sec, n_segments_per_sample, noise_paths, speed
                     ) -> Tuple[Callable, Callable]:
    if stage in ["valid", "test"]:
        transform = [
            RandomTimeShift(sample_rate=sample_rate, shift_sec=hop_length_sec / 2, prob=0.5),
        ]
    elif stage == "train":
        transform = [
            RandomBackgroundNoise(
                sample_rate=sample_rate, augment_files=noise_paths, min_snr_db=0, max_snr_db=5, prob=0.2),
            RandomSpeedChange(sample_rate=sample_rate, pert_speed=speed, prob=0.15),
            RandomTimeShift(sample_rate=sample_rate, shift_sec=hop_length_sec / 2, prob=0.5),
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


def map_mix_to_lbl(mix: Mix) -> Tuple[int, int]:
    if mix in [Mix.VocalOrigInstrOrig, Mix.VocalOrigInstrModMel, Mix.VocalOrigInstrModNoMel]:
        vocal = 0
    else:
        vocal = 1

    if mix in [Mix.VocalOrigInstrOrig, Mix.VocalModInstrOrig]:
        instrument = 0
    else:
        instrument = 1

    return vocal, instrument


class DatasetAudioMixer(Dataset):
    def __init__(self,
                 stage: str,
                 base_dir: str,
                 song_separate_fns: List[str],
                 non_modified: List[str],
                 vocal_gen_inst_gen_wmel_fns: List[str],
                 random_gen_instr_dirs: List[str],
                 noise_path_dirs: List[str],

                 sample_rate: int,
                 audio_length_sec: float,
                 segment_length_sec: float,
                 hop_length_sec: float,
                 n_segments_per_sample: int,
                 prob_mixer: Tuple[float, float, float, float],
                 speed: float,
                 factor_mix_db: Tuple[float, float],

                 rank: int = 0,
                 debug: str = None,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.base_dir = Path(base_dir)
        self.rank = rank
        self.verbose = verbose
        self.debug = debug

        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.segment_length_sec = segment_length_sec
        self.hop_length_sec = hop_length_sec
        self.n_segments_per_sample = n_segments_per_sample
        self.prob_mixer = np.array(prob_mixer) / np.array(prob_mixer).sum()
        self.speed = speed
        self.duration_speed_min_sec = (1 + self.speed) * self.audio_length_sec
        self.factor_mix_db = factor_mix_db

        self.segment_length = int(self.sample_rate * self.segment_length_sec)
        self.hop_length = int(self.sample_rate * self.hop_length_sec)

        self.local_max_idx = math.ceil(self.duration_speed_min_sec / self.segment_length_sec)

        # Get reference files.
        for song_separate_fn in song_separate_fns:
            self.song_separate = json.load(open(song_separate_fn))

        # song_separate_list = []
        # for idd, sample in self.song_separate.items():
        #     tmp = dict(idd=idd)
        #     tmp.update(sample)
        #     song_separate_list.append(tmp)
        # self.song_separate = song_separate_list
        # random.shuffle(self.song_separate)
        self.logger.info(f'len(self.song_separate) = {len(self.song_separate)}')

        # for vocal_gen_fn in vocal_gen_fns:
        #     self.vocal_gen = json.load(open(vocal_gen_fn))
        #     self.logger.info(f'len(self.vocal_gen) = {len(self.vocal_gen)}')

        for vocal_gen_inst_gen_wmel_fn in vocal_gen_inst_gen_wmel_fns:
            self.vocal_gen_inst_gen_wmel = json.load(open(vocal_gen_inst_gen_wmel_fn))
            self.logger.info(f'len(self.vocal_gen_inst_gen_wmel) = {len(self.vocal_gen_inst_gen_wmel)}')

        self.idds = list(self.vocal_gen_inst_gen_wmel.keys())

        # Get files for augmentations.
        self.random_gen_instrs = []
        for random_gen_instr_dir in random_gen_instr_dirs:
            self.random_gen_instrs += [_ for _ in Path(random_gen_instr_dir).walkfiles("*.wav")]
        self.logger.info(f'len(self.random_gen_instrs) = {len(self.random_gen_instrs)}')

        # Get files for augmentations.
        self.noise_paths = []
        for noise_path_dir in noise_path_dirs:
            self.noise_paths += [_ for _ in Path(noise_path_dir).walkfiles("*.wav")]
        self.logger.info(f'len(self.noise_paths) = {len(self.noise_paths)}')

        # Clean and save.
        gc.collect()

        self.logger.info('build_transforms')
        self.transform, self.chunker = build_transforms(
            stage, self.sample_rate, self.segment_length_sec, self.hop_length_sec, self.n_segments_per_sample,
            self.noise_paths, self.speed)

        if self.debug == "simple":
            self.mixer = 4 * [TestMixer(self.sample_rate, self.duration_speed_min_sec), ]
        else:
            self.mixer = [
                OriginalVocalMixOriginalInstruments(
                    self.sample_rate, self.duration_speed_min_sec, factor_mix_db=self.factor_mix_db),
                GeneratedVocalMixOriginalInstruments(
                    self.sample_rate, self.duration_speed_min_sec, factor_mix_db=self.factor_mix_db),
                OriginalVocalMixGeneratedInstruments(
                    self.sample_rate, self.duration_speed_min_sec, factor_mix_db=self.factor_mix_db,
                    prob_with_melody=0.5, random_gen_instrs=self.random_gen_instrs),
                GeneratedVocalMixGeneratedInstruments(
                    self.sample_rate, self.duration_speed_min_sec, factor_mix_db=self.factor_mix_db)
            ]

        self.logger.info('DatasetAudioMixer initiated')

    def print(self, msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        if self.debug is None:
            return len(self.vocal_gen_inst_gen_wmel)
        else:
            return 24

    def __getitem__(self, idx) -> Tuple[str, Tensor, Tensor, Tensor, Tensor, Tuple[int, int], Tuple[int, int]]:
        idd, audio0 = "None", None
        success, n_tries = False, 0
        while (not success) and (n_tries <= 1):
            idd = self.idds[idx] if n_tries == 0 else random.choice(self.idds)

            # Get samples.
            sample = MusicBank(
                orig=self.base_dir / self.song_separate[idd]["orig"],

                vocals=self.base_dir / self.song_separate[idd]["separate_dir"] / "vocals.wav",
                bass=self.base_dir / self.song_separate[idd]["separate_dir"] / "bass.wav",
                drums=self.base_dir / self.song_separate[idd]["separate_dir"] / "drums.wav",
                other=self.base_dir / self.song_separate[idd]["separate_dir"] / "other.wav",

                vocal_gen=self.base_dir / np.random.choice(self.vocal_gen_inst_gen_wmel[idd]["vocal_gen"]),
                instr_gen_wmel=self.base_dir / np.random.choice(self.vocal_gen_inst_gen_wmel[idd]["instr_gen_wmel"]),
            )

            # Select mixer.
            mixer = np.random.choice(self.mixer, p=self.prob_mixer)

            try:
                sr, audio0 = read_to_waveform(
                    sample.orig, mono=True, sr=self.sample_rate,
                    duration=self.duration_speed_min_sec, dtype=torch.float32)
                success = True
            except:
                msg = f'Some error occurred reading file: {sample.orig}'
                self.logger.error(msg)
                n_tries += 1

        if audio0 is None:
            msg = f'It was not possible to load any data.'
            self.logger.error(msg)
            raise RuntimeError(msg)

        audio1, cls = mixer(sample)
        check_mono_same_shape(self.logger, audio0, audio1)

        idxs0 = torch.arange(math.ceil(len(audio0) / self.segment_length)).long()
        idxs0 = idxs0.repeat_interleave(self.segment_length)
        idxs0 = idxs0[:len(audio0)]
        idxs1 = idxs0.detach().clone()

        idxs1, audio0, audio1 = self.transform(idxs1, audio0, audio1)

        idxs0, idxs1, audio0, audio1 = self.chunker(idxs0, idxs1, audio0, audio1)

        # Most frequent element in each chunks
        idxs0 = torch.mode(idxs0, dim=-1)[0]
        idxs1 = torch.mode(idxs1, dim=-1)[0]

        # Local index to global index.
        idxs0 = idx * self.local_max_idx + idxs0
        idxs1 = idx * self.local_max_idx + idxs1

        return idd, idxs0.long(), idxs1.long(), audio0, audio1, (0, 0), map_mix_to_lbl(cls)

    def collate_fn(self, batch) -> Tuple[List[str], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        s_idd, s_idxs0, s_idxs1, s_audio0, s_audio1, s_cls0, s_cls1 = [], [], [], [], [], [], []
        for (idd, idxs0, idxs1, audio0, audio1, cls0, cls1) in batch:
            s_idd.append(idd)
            s_idxs0.append(idxs0)
            s_idxs1.append(idxs1)
            s_audio0.append(audio0)
            s_audio1.append(audio1)
            s_cls0.append(torch.tensor(cls0))
            s_cls1.append(torch.tensor(cls1))
        s_idxs0 = torch.cat(s_idxs0, dim=0)
        s_idxs1 = torch.cat(s_idxs1, dim=0)
        s_audio0 = torch.cat(s_audio0, dim=0)
        s_audio1 = torch.cat(s_audio1, dim=0)
        s_cls0 = torch.stack(s_cls0, dim=0)
        s_cls1 = torch.stack(s_cls1, dim=0)

        return s_idd, s_idxs0, s_idxs1, s_audio0, s_audio1, s_cls0, s_cls1


def main_worker_for_test(this_gpu, n_gpus_per_node):
    _base = '/home/razor'
    n_segments_per_sample = 1
    ds = DatasetAudioMixer(
        stage="train",
        base_dir='/home/razor/MyData',
        song_separate_fns=['/home/razor/MyTmp/split_1/song_separate_train.json'],
        vocal_gen_fns=['/home/razor/MyTmp/split_1/vocal_gen_train.json'],
        instr_gen_wmel_fns=['/home/razor/MyTmp/split_1/instr_gen_wmel_train.json'],
        vocal_gen_inst_gen_wmel_fns=['/home/razor/MyTmp/split_1/vocal_gen_inst_gen_wmel_train.json'],
        prob_mixer=(1, 3, 3, 3),
        random_gen_instr_dirs=['/home/razor/MyData/music_generated/AudioCraft/random'],
        noise_path_dirs=['/home/razor/MyData/musan/noise', '/home/razor/MyData/musan/speech'],
        sample_rate=16_000,
        speed=0.1,
        factor_mix_db=(-5, 5),
        audio_length_sec=4.5,
        segment_length_sec=4,
        hop_length_sec=0.5,
        n_segments_per_sample=n_segments_per_sample,
        rank=0,
        debug="full",
        verbose=True
    )

    # Test dataset
    for _idx, (_idd, _idxs0, _idxs1, _segs0, _segs1, _cls0, _cls1) in enumerate(ds):
        _wav0 = torch_join_segments(_segs0, ib0=0, ib1=n_segments_per_sample)
        _wav1 = torch_join_segments(_segs1, ib0=0, ib1=n_segments_per_sample)
        Path("/home/razor/MyTmp/tmp_ds").makedirs_p()
        audio_write(f'/home/razor/MyTmp/tmp_ds/{_idd}-orig', _wav0.cpu(), ds.sample_rate, strategy="loudness")
        audio_write(f'/home/razor/MyTmp/tmp_ds/{_idd}-{_cls1}-copy', _wav1.cpu(), ds.sample_rate, strategy="loudness")
        print(_segs0.shape, _segs1.shape)
        print(_idxs0)
        print(_idxs1)
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
    for it, (_idds, _idxs0, _idxs1, _segs0, _segs1, _cls0, _cls1) in enumerate(loader):
        print(_segs0.shape)
        print(_segs1.shape)
        _wav0 = torch_join_segments(_segs0, ib0=0, ib1=n_segments_per_sample)
        _wav1 = torch_join_segments(_segs1, ib0=0, ib1=n_segments_per_sample)
        print(_wav0.shape, _idxs0)
        print(_wav1.shape, _idxs1)
        print()
        Path("/home/razor/MyTmp/tmp_dl").makedirs_p()
        for _idd in _idds:
            audio_write(f'/home/razor/MyTmp/tmp_dl/{_idd}-orig', _wav0.cpu(), ds.sample_rate, strategy="loudness")
            audio_write(f'/home/razor/MyTmp/tmp_dl/{_idd}-{_cls1[0]}-copy', _wav1.cpu(), ds.sample_rate,
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
