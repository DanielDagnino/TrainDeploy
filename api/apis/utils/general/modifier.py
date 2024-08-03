import re
from typing import Optional


def _modify(config, modifiers):
    for k_mod, v_mod in modifiers.items():
        for k, v in config.items():
            if isinstance(v, str):
                match = re.match('^\{.*\}$', v)
                if match and match.group(0) == '{' + k_mod + '}':
                    config[k] = v_mod
                else:
                    # match = re.match('\{.*\}', v)
                    # if match and match.group(0) == '{' + k_mod + '}':
                    config[k] = v.replace("{" + k_mod + "}", str(v_mod))
            elif isinstance(v, dict):
                config[k] = dict_modifier(v, None, modifiers)


def dict_modifier(config: dict, modifiers: Optional[str], pre_modifiers: dict = None) -> dict:
    # Apply modifiers to config.
    if modifiers is not None:
        modifiers = config.pop(modifiers, None)
        if modifiers is not None:
            _modify(config, modifiers)

    # Apply pre_modifier to modifier.
    if pre_modifiers is not None:
        _modify(config, pre_modifiers)

    return config
