import os
import re

import yaml


def init_custom_yaml():
    tag = '!ENV'
    pattern = re.compile(r'.*?\${(\w+)}.*?')
    yaml.add_implicit_resolver(tag, pattern)

    def path_constructor(loader, node):
        """
        Extracts the environment variable from the node's value.

        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(f'${{{g}}}', os.environ.get(g, g))
            return full_value
        return value

    yaml.add_constructor(tag, path_constructor)
