import inspect
import logging
import warnings
from typing import Tuple

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from easydict import EasyDict

import torch

from apis.clf_ai.model.utils import DatasetAudioMixerInfer
from apis.clf_ai.model.utils import Initiate

_CFG_BEATS_PRETR_FN = './apis/clf_ai/model/cfg_pretrained.json'
_CFG_AI_VOICE = EasyDict(cfg_fn="./apis/clf_ai/model/cfg_infer_voice.yaml")
_SMOOTH = 1
_BATCH_SIZE = 1

extractor_ai_voice = Initiate(io_file=None, args=_CFG_AI_VOICE, cfg_beats_pretr_fn=_CFG_BEATS_PRETR_FN)


def predict_is_ai(io_file, extractor) -> Tuple[float, float]:
    logger = logging.getLogger(__name__ + ': ' + inspect.currentframe().f_code.co_name)

    with torch.no_grad():
        extractor.dataset = DatasetAudioMixerInfer(io_file=io_file, **extractor.cfg.dataset.get("test"))
        for audio, duration in extractor.dataset:
            try:
                segs = audio.detach().requires_grad_(requires_grad=False)
                if torch.cuda.is_available():
                    segs = segs.cuda(non_blocking=True)
                pred = []
                while len(segs) > 0:
                    batch_segs = segs[:_BATCH_SIZE]
                    segs = segs[_BATCH_SIZE:]
                    if torch.cuda.is_available():
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            out = extractor.model(batch_segs)
                    else:
                        out = extractor.model(batch_segs)
                    out = out.reshape(-1).cpu()
                    pred.append(out)
                pred = extractor.pred_reducer(pred, smooth=_SMOOTH)
                return pred, duration
            except Exception as expt:
                logger.error(expt)
                return -1, duration
