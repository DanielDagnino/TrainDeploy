import argparse

import pandas as pd
import torch
from path import Path
from tqdm import tqdm

from is_ai.is_ai_song.evaluate.eval_clf import Initiate


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
    pd.DataFrame(pred_gather, columns=['pred', 'lbl', 'path']).to_csv(
        fn_results, index=False, float_format='%.3f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Evaluating')
    parser.add_argument("cfg_fn", type=Path, help="Configuration file")
    parser.add_argument('--base_out_dir', default='/home/razor/MyTmp/VokelAI/DeepFakeSongDetector/AudioClf:beats/results/', help='Path to dataset')
    parser.add_argument('--no_overwrite', action='store_true', help='No overwrite results')
    _args = parser.parse_args()
    extract(_args)
