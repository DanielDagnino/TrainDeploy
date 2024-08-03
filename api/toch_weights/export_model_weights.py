import argparse
import json
import os
import sys

import torch
from path import Path

from apis.utils.torch.dataparallel import add_module_dataparallel


def run(fn_in: Path, fn_out: Path, pretrained_model_fn: Path, fn_out_cfg_pret: Path):
    cfg_pretrained = torch.load(pretrained_model_fn)
    cfg_pretrained = cfg_pretrained['cfg']
    cfg_pretrained["finetuned_model"] = False
    del cfg_pretrained["predictor_dropout"]
    del cfg_pretrained["predictor_class"]
    json.dump(cfg_pretrained, open(fn_out_cfg_pret, "w"), indent=4)

    print(f'Model size in = {os.path.getsize(fn_in) / 1024:.2f} Kb')

    checkpoint = torch.load(fn_in, map_location='cpu')
    print(f'type(checkpoint) = {type(checkpoint)}')
    print(f'state_dict.keys() = {checkpoint.keys()}')
    state_dict = checkpoint['model']
    state_dict = add_module_dataparallel(state_dict)

    checkpoint = dict(state_dict=state_dict, cfg=None)

    torch.save(checkpoint, fn_out)
    print(f'Model size out = {os.path.getsize(fn_out) / 1024:.2f} Kb')



def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fn_in',
                        default='~/MyTmp/TrainDeploy/AudioClf:beats/last.ckpt',
                        type=Path, help='Input file checkpoint')
    parser.add_argument('--pretrained_model_fn',
                        default='~/MyModelsDwnld/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
                        type=Path, help='Input file checkpoint')
    parser.add_argument('--fn_out',
                        default='~/MyProjects/TrainDeploy/api/apis/clf_ai/model/weights_voice.ckpt',
                        type=Path, help='Output file checkpoint')
    parser.add_argument('--fn_out_cfg_pret',
                        default='tmp_cfg_pretrained.json',
                        type=Path, help='Output file checkpoint')
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == '__main__':
    main(sys.argv[1:])
