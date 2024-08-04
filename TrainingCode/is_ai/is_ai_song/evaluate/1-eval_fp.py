import argparse
import json
import math
import os
import random

import faiss
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict
from path import Path
from torch.utils.data import DataLoader

import plot_results
from is_ai.is_ai_song.dataset.ds_mixer import DatasetAudioMixer
from is_ai.is_ai_song.evaluate.utils import compute_metrics
from is_ai.is_ai_song.model.base_fp import AudioFP
from is_ai.is_ai_song.model.helpers import load_checkpoint
from is_ai.utils.general.custom_yaml import init_custom_yaml
from is_ai.utils.general.modifier import dict_modifier


def extract(args):
    print('seed')
    np.random.seed(1234)
    random.seed(1234)
    torch.manual_seed(1234)
    cudnn.deterministic = True

    print('Configuration')
    init_custom_yaml()
    cfg = yaml.load(open(args.cfg_fn), Loader=yaml.Loader)
    cfg = dict_modifier(config=cfg, modifiers="modifiers", pre_modifiers={"HOME": os.path.expanduser("~")})
    cfg = EasyDict(cfg)

    print("Build model")
    if cfg.engine.model.name == AudioFP.__name__:
        print(f"{AudioFP.__name__} building")
        model = AudioFP(cfg.engine.model)
        model.cuda()
    else:
        msg = f'Unknown model name = {cfg.name}'
        print(msg)
        raise ValueError(msg)

    print("Load model")
    _, _, _, _ = load_checkpoint(
        model, cfg.engine.model.resume.load_model_fn, None, None, None, torch.device('cuda'), False, False, True)

    print("Dataset")
    if cfg.dataset.name == DatasetAudioMixer.__name__:
        dataset = DatasetAudioMixer(stage="test", **cfg.dataset.get("test"), rank=0)
    else:
        raise ValueError(f"Not implemented dataset {cfg.dataset.name}")

    print("Loader")
    loader = DataLoader(dataset, **cfg.loader.get('test'), shuffle=False, sampler=None, collate_fn=dataset.collate_fn)

    if cfg.cnn_benchmark:
        cudnn.benchmark = True

    print('Compute reference and query features')
    # reference_fn, reference_id, query_id, ref_features, query_features = compute_or_load(
    #     args.base_out_dir, 'query', query_id, loader['query'], model, args.overwrite)
    # fn_eval_fps = f'{args.base_out_dir}/eval_fps.h5'
    # join_h5(fn_eval_fps, reference_fn, reference_id, query_id, ref_features, query_features)

    fn_out = Path(f'{args.base_out_dir}/features_ids.h5')
    fn_out.parent.makedirs_p()
    if not Path(fn_out).exists() or not args.no_overwrite:
        model.eval()
        torch.set_grad_enabled(False)

        ref_features, fn_idd_segs, lbl_segs, query_features = [], [], [], []
        for it, batch in enumerate(loader):
            idds, idxs0, idxs1, segs0, segs1, _, _ = batch
            nb_ref = len(segs0)
            segs = torch.cat([segs0, segs1], dim=0
                             ).detach().requires_grad_(requires_grad=False).cuda(non_blocking=cfg.train.non_blocking)

            with torch.autocast(device_type='cuda', dtype=torch.float32):
                embeddings = model(segs)

            ref_features.append(embeddings[:nb_ref].cpu().numpy())
            query_features.append(embeddings[-nb_ref:].cpu().numpy())
            fn_idd_segs.append(idds)
            lbl_segs.append(idxs0)

        ref_features = np.concatenate(ref_features, axis=0)
        fn_idd_segs = np.concatenate(fn_idd_segs, axis=0)
        lbl_segs = np.concatenate(lbl_segs, axis=0)
        query_features = np.concatenate(query_features, axis=0)

    #     h5py.File(fn_out, 'w').create_dataset("ref_features", data=ref_features)
    #     h5py.File(fn_out, 'w').create_dataset("fn_idd_segs", data=fn_idd_segs)
    #     h5py.File(fn_out, 'w').create_dataset("lbl_segs", data=lbl_segs)
    #     h5py.File(fn_out, 'w').create_dataset("query_features", data=query_features)
    # else:
    #     ref_features = np.array(h5py.File(fn_out, 'r')["ref_features"])
    #     fn_idd_segs = np.array(h5py.File(fn_out, 'r')["fn_idd_segs"])
    #     lbl_segs = np.array(h5py.File(fn_out, 'r')["lbl_segs"])
    #     query_features = np.array(h5py.File(fn_out, 'r')["query_features"])

    print('Pairs proposals using FAISS')
    fn_results = Path(f'{args.base_out_dir}/results.csv')
    if not fn_results.exists() or not args.no_overwrite:
        if args.faiss_sort == "norm_l2":
            index_reference = faiss.IndexFlatL2(ref_features.shape[1])
        elif args.faiss_sort == "inner_prod":
            index_reference = faiss.index_factory(ref_features.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Not implemented Faiss sorting mode")
        if not args.cpu:
            n_gpus, co = faiss.get_num_gpus(), faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat32 = True
            index_reference = faiss.index_cpu_to_all_gpus(index_reference, co=co, ngpu=n_gpus)
        faiss.normalize_L2(ref_features)
        index_reference.add(ref_features)
        faiss.normalize_L2(query_features)
        reference_dist, reference_ind = index_reference.search(query_features, k=min(args.nn_proposals, len(lbl_segs)))
        results = pd.DataFrame(columns=['query_id', 'reference_id', 'score'])
        results['query_id'] = np.repeat(lbl_segs, min(args.nn_proposals, len(lbl_segs))).astype(str)
        results['reference_id'] = lbl_segs[reference_ind.ravel()].astype(str)
        if args.faiss_sort == "norm_l2":
            results['score'] = (math.sqrt(2.) - reference_dist.ravel()) / math.sqrt(2.)
        elif args.faiss_sort == "inner_prod":
            results['score'] = reference_dist.ravel()
        else:
            raise ValueError("Not implemented Faiss sorting mode")

        results['pred'] = results['score'] > args.threshold
        results = results[['reference_id', 'query_id', 'score', 'pred']]
        results.to_csv(fn_results, index=False, sep=";")
    else:
        results = pd.read_csv(fn_results, sep=";")

    print(f'results["file1"].shape = {results["reference_id"].shape}')
    print(f'results["file2"].shape = {results["query_id"].shape}')
    print(f'results["score"].shape = {results["score"].shape}')
    print(f'results["pred"].shape = {results["pred"].shape}')

    print('Evaluation: float')
    gt = results["reference_id"] == results["query_id"]
    stats = compute_metrics(gt, results["pred"], results["score"])
    stats["experiment"] = args.exp_name
    json.dump(stats, open(fn_results.replace('.csv', '.json'), 'w'), indent=4)

    print('Evaluation: float')
    plot_results.main([
        fn_results.replace('.csv', '.json'),
        '--out-dir', fn_results.parent
    ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluating')
    parser.add_argument("cfg_fn", type=Path, help="Configuration file")
    parser.add_argument("exp_name", type=str, help="Experiment name")
    parser.add_argument('--base_out_dir', default='/home/razor/MyTmp/AudioSearchResults', help='Path to dataset')
    parser.add_argument('--no_overwrite', action='store_true', help='No overwrite results')
    parser.add_argument('--nn_proposals', type=int, default=10, help='Annotation file')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for the predictions')
    parser.add_argument('--faiss_sort', choices=['norm_l2', 'inner_prod'], default='inner_prod',
                        help='Faiss sorting mode.')
    _args = parser.parse_args()
    extract(_args)
