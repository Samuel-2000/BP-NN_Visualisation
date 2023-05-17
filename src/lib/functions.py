import pickle
import torch
import re
import io


def atof(text):
    """
    Function converts char representation of number if the parameter text is number to float representation.
    Else it returns the input unchanged.

    :param text: number to be converted
    :return: float number or unchanged input
    """
    try:
        retval = float(text)
    except ValueError:
        retval = text

    return retval


def natural_keys(text):
    """
    Function creates key for natural sort.

    :param text: input to create the key from
    :return: keys
    """
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
