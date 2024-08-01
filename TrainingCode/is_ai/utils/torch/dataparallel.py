from collections import OrderedDict

from torch.nn import Module


def is_state_dict_dataparallel(state_dict):
    return all([_[:7] == 'module.' for _ in state_dict.keys()])


def is_model_dataparallel(model: Module):
    return all([_[:7] == 'module.' for _ in model.state_dict().keys()])


def naked_model(model):
    if hasattr(model, "module"):
        return model.module
    return model


def add_module_dataparallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[f'module.{k}'] = v
    return new_state_dict


def rmv_module_dataparallel(state_dict):
    if is_state_dict_dataparallel(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        return new_state_dict
    else:
        return state_dict
