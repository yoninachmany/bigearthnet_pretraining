# -*- coding: utf-8 -*-
"""TODO"""
import logging
from pathlib import Path
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from fastai.vision import (
    ImageList,
    List,
    pil2tensor,
    Image,
    get_transforms,
    partial,
    accuracy_thresh,
    fbeta,
    Learner,
)
from fastai.callbacks import SaveModelCallback, CSVLogger
from fastai.vision.models.xresnet import (
    xresnet18,
    xresnet34,
    xresnet50,
    xresnet101,
    xresnet152,
)
import torch


@click.command()
@click.argument("gpu", type=int)
@click.argument("arch", type=str)
@click.argument("bands", type=str)
@click.argument("scale", type=bool)
@click.argument("epochs", type=int)
def main(gpu, arch, bands, scale, epochs):
    """TODO"""
    logger = logging.getLogger(__name__)
    logger.info("TODO")

    r_g_b_nir_swir1_swir2 = ["B04", "B03", "B02", "B08", "B11", "B12"]
    bands_str = bands

    np.random.seed(42)
    torch.cuda.set_device(gpu)

    name = f"arch-{arch}-bands-{bands}-scale-{scale}-epochs-{epochs}"
    if arch == "xresnet18":
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
        return

    if bands == "rgb":
        bands = r_g_b_nir_swir1_swir2[:3]
    elif bands == "rgbnir":
        bands = r_g_b_nir_swir1_swir2[:4]
    elif bands == "rgbnirswir":
        bands = r_g_b_nir_swir1_swir2
    else:
        return

    data = get_data(bands, scale)
    model = get_model(arch, bands, data)
    learn = get_learn(data, model, bands_str)
    cbs = [SaveModelCallback(learn)]
    cbs += [CSVLogger(learn, filename=name, append=True)]

    learn.lr_find()
    learn.recorder.plot(suggestion=True, return_fig=True)
    plt.savefig(f"reports/figures/{name}-stage1-lr")
    learn_rate = learn.recorder.min_grad_lr

    learn.fit_one_cycle(epochs, slice(learn_rate), callbacks=cbs)
    learn.save(f"{name}-stage-1")

    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot(suggestion=True, return_fig=True)
    plt.savefig(f"reports/figures/{name}-stage-2-lr")
    learn_rate = learn.recorder.min_grad_lr

    learn.fit_one_cycle(epochs, slice(learn_rate), callbacks=cbs)
    learn.save(f"{bands_str}/{name}-stage-2")

    learn.export(f"{bands_str}/{name}.pkl")
    st = learn.model.state_dict()
    torch.save(st, f"models/{bands_str}/{name}-state-dict")


class BigEarthNetTiffList(ImageList):
    """TODO"""

    def __init__(self, *args, bands: List[str], scale: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.bands = bands
        self.scale = scale
        self.copy_new.append("bands")
        self.copy_new.append("scale")

    def open(self, fn):
        """TODO"""
        patch = Path(fn).name
        paths = [f"{fn}/{patch}_{band}.tif" for band in self.bands]
        bands = [
            np.array(PIL.Image.open(path).resize((120, 120))) for path in paths
        ]
        data = np.dstack(bands)
        if self.scale:
            data = np.clip(data / 2750, 0, 1)
        data = pil2tensor(data, np.float32)
        return Image(data)


def get_data(bands, scale):
    """TODO"""
    data_path = Path("data")
    raw_path = data_path / "raw"
    processed_path = data_path / "processed"
    images_path = raw_path / "BigEarthNet-v1.0"
    labels_path = processed_path / "bigearthnet_labels.csv"

    label_df = pd.read_csv(labels_path)
    src = (
        BigEarthNetTiffList.from_df(
            label_df, images_path, bands=bands, scale=scale
        )
        .split_from_df("val")
        .label_from_df(label_delim="|")
    )
    tfms = get_transforms()
    data = src.transform(tfms).databunch(bs=256).normalize()
    return data


def get_model(arch, bands, data):
    """TODO"""
    model = arch(c_in=len(bands), c_out=len(data.classes))
    return model


def get_metrics(thresh=0.2):
    """TODO"""
    acc_thresh = partial(accuracy_thresh, thresh=thresh)
    f_score = partial(fbeta, thresh=thresh)
    metrics = [acc_thresh, f_score]
    return metrics


def get_learn(data, model, bands_str):
    """TODO"""
    metrics = get_metrics()
    learn = Learner(
        data, model, metrics=metrics, path="models", model_dir=bands_str
    )
    learn = learn.mixup(stack_y=False).to_fp16()
    return learn


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()