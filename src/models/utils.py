"""TODO"""
from fastai.vision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from fastai.vision.models.xresnet import (
    xresnet18,
    xresnet34,
    xresnet50,
    xresnet101,
    xresnet152,
)
from fastai.callbacks import SaveModelCallback, CSVLogger


def get_arch_bands(arch, bands):
    """TODO"""
    if arch == "resnet18":
        arch = resnet18
    elif arch == "resnet34":
        arch = resnet34
    elif arch == "resnet50":
        arch = resnet50
    elif arch == "resnet101":
        arch = resnet101
    elif arch == "resnet152":
        arch = resnet152
    elif arch == "xresnet18":
        arch = xresnet18
    elif arch == "xresnet34":
        arch = xresnet34
    elif arch == "xresnet50":
        arch = xresnet50
    elif arch == "xresnet101":
        arch = xresnet101
    elif arch == "xresnet152":
        arch = xresnet152
    else:
        arch = None

    rgb_nir_swir = ["B04", "B03", "B02", "B08", "B11", "B12"]
    if bands == "rgb":
        bands = rgb_nir_swir[:3]
    elif bands == "rgbnir":
        bands = rgb_nir_swir[:4]
    elif bands == "rgbnirswir":
        bands = rgb_nir_swir
    else:
        bands = None

    return arch, bands


def get_callbacks(learn, name, monitor):
    """TODO"""
    cbs = [SaveModelCallback(learn, monitor)]
    cbs += [CSVLogger(learn, filename=name, append=True)]
    return cbs
