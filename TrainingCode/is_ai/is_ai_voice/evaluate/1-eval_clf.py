import argparse
import json

import pandas as pd
import torch
from path import Path
from tqdm import tqdm

from is_ai.is_ai_voice.evaluate.eval_clf import Initiate
from is_ai.is_ai_song.evaluate.utils import compute_metrics


def extract(args):
    extractor = Initiate(args)

    with torch.no_grad():
        batch_size = extractor.cfg.loader.test.batch_size
        pred_gather = []
        for path, lbl, audio in tqdm(extractor.dataset):
            try:
                segs = audio.detach().requires_grad_(requires_grad=False)
                if torch.cuda.is_available():
                    segs = segs.cuda(non_blocking=extractor.cfg.loader.test.non_blocking)
                pred = []
                while len(segs) > 0:
                    batch_segs = segs[:batch_size]
                    segs = segs[batch_size:]
                    if torch.cuda.is_available():
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            out = extractor.model(batch_segs)
                    else:
                        out = extractor.model(batch_segs)
                    out = out.reshape(-1).cpu()
                    pred.append(out)
                pred_gather.append([extractor.pred_reducer(pred), lbl, path])
            except Exception as expt:
                print(f"Some problem predicting file: path = {path}")
                print(expt)

    print('Save')
    fn_results = Path(f'{args.base_out_dir}/results.csv')
    fn_results.parent.makedirs_p()
    pred_gather = sorted(pred_gather, key=lambda _: -_[0])
    pd.DataFrame(pred_gather, columns=['pred', 'lbl', 'path']).to_csv(fn_results, index=False, float_format='%.3f')

    pred_gather = pd.read_csv(fn_results)
    scores = pred_gather['pred']
    pred = scores > 0.5
    gt = pred_gather['lbl'] == 'ai'
    res_json = compute_metrics(gt, pred, scores)
    json.dump(res_json, open(fn_results.splitext()[0] + '_metrics.json', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluating')
    parser.add_argument("cfg_fn", type=Path, help="Configuration file")
    parser.add_argument('--base_out_dir', default='/home/razor/MyTmp/DeepFakeVoiceDetector/AudioClf:beats/results', help='Path to dataset')
    parser.add_argument('--no_overwrite', action='store_true', help='No overwrite results')
    _args = parser.parse_args()
    extract(_args)
