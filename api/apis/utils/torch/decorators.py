""""
Info about decorators:
https://realpython.com/primer-on-python-decorators/#simple-decorators
https://dev.to/apcelent/python-decorator-tutorial-with-example-529f#:~:text=Python%20decorator%20are%20the%20function,match%20the%20function%20to%20decorate.
"""
import functools
import inspect
import logging
import torch


def preserve_trainability(stage: str = "eval"):
    """Runs a specific function changing the trainability to the specified one and leaving the original trainability
    once the function has ended.
    """

    def _inner_function(func):
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs):
            """"
            Args:
                obj: obj that can be a model or a class with a "model" attribute
                *args, **kwargs: function arguments
            """
            logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

            if hasattr(obj, "model"):
                model = getattr(obj, "model")
            else:
                model = obj

            grad_enabled = torch.is_grad_enabled()
            stage_in = model.training

            if stage == "eval":
                model.eval()
                torch.set_grad_enabled(False)
            elif stage == "train":
                model.train()
                torch.set_grad_enabled(True)
            elif stage == "frozen":
                model.eval()
            else:
                logger.error('Not accepted stage = %s. Accepted values: "eval", "train", "frozen"', stage)
                raise ValueError(preserve_trainability.__name__)

            val = func(obj, *args, **kwargs)

            if stage != "frozen":
                torch.set_grad_enabled(grad_enabled)
            model.train(stage_in)

            return val

        return wrapper

    return _inner_function
