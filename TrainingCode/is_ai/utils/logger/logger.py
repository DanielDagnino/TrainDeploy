import json
import logging.config
import os
from warnings import warn

from path import Path

from is_ai.utils.general.modifier import dict_modifier


# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
# https://www.loggly.com/blog/exceptional-logging-of-exceptions-in-python/
# https://stackoverflow.com/questions/7621897/python-logging-module-globally
# https://docs.python.org/2/howto/logging.html
def setup_logging(
        log_dir: Path = None,
        log_config: Path = None,
        logger_level: int = logging.INFO,
        env_key: str = 'LOG_CFG',
        rank: int = None) -> None:
    """Setup logging configuration"""
    if not (log_dir or log_config):
        logging.basicConfig(level=logger_level)
        logging.warning(f"Logger not define. Logs will be shown only in the terminal with a "
                        f"logger_level={logger_level}")
        return

    if not log_dir:
        log_dir = os.getcwd()
        warn(f"Logger directory not defined. Default values will be used: log_dir = {log_dir} ", Warning)

    path_env = os.getenv(env_key, None)
    if path_env is not None and log_config is not None:
        warn("Two logger configuration provided, from environment variables and from path file. "
             f"path_env = {path_env} and log_config = {log_config}. In this situation, log_config prevails.", Warning)
    log_config = log_config or path_env
    if log_config:
        if not log_config.exists():
            raise ValueError(__name__ + ": " + setup_logging.__name__ + 'Logger directory does not exist. log_config = '
                                                                        f'{log_config}')
    else:
        if rank is not None or rank == 0:
            log_config = Path(__file__).abspath().parent / 'logger_config.json'
        else:
            log_config = Path(__file__).abspath().parent / 'logger_config_no_console_info.json'

    config = json.load(open(log_config, 'rt'))
    if log_dir:
        config["modifiers"] = {"LOGDIR": log_dir}
    else:
        config["modifiers"] = {"LOGDIR": "."}
    if rank is not None:
        config["modifiers"].update({"RANK": f'_{rank}'})
    else:
        config["modifiers"].update({"RANK": ''})
    config = dict_modifier(config=config, modifiers="modifiers")
    logging.config.dictConfig(config)
