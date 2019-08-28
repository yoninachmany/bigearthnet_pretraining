# -*- coding: utf-8 -*-
"""TODO"""
import logging
from pathlib import Path
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai.vision import (
    SegmentationItemList,
    List,
    pil2tensor,
    Image,
    get_transforms,
    FBeta,
    ConfusionMatrix,
    unet_learner,
    Tensor,
    CrossEntropyFlat,
)
from fastai.vision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from fastai.metrics import CMScores
import torch
from tqdm import tqdm
from utils import get_arch_bands, get_callbacks


@click.command()
@click.argument("gpu", type=int)
@click.argument("arch", type=str)
@click.argument("bands", type=str)
@click.argument("weighted", type=bool)
@click.argument("epochs", type=int)
def main(gpu, arch, bands, weighted, epochs):
    """TODO"""
    logger = logging.getLogger(__name__)
    logger.info("TODO")

    np.random.seed(42)
    torch.cuda.set_device(gpu)

    pre_name = f"arch-{arch}-bands-{bands}-scale-True-epochs-20"
    name = f"arch-unet-{arch}-bands-{bands}-weighted-{weighted}-epochs-{epochs}"

    arch, bands = get_arch_bands(arch, bands)
    if arch is None or bands is None:
        return

    data = get_data(bands, bs=8)
    if arch in [resnet18, resnet34, resnet50, resnet101, resnet152]:
        cut = -2
        learn = get_learn(data, arch, name, weighted, cut=cut)
    else:
        cut = -3
        model = get_model(arch, bands, pre_name)
        learn = get_learn(
            data, lambda pretrained: model, name, weighted, cut=cut
        )

    cbs = get_callbacks(learn, name, "f_beta")

    learn.lr_find()
    learn.recorder.plot(suggestion=True, return_fig=True)
    plt.savefig(f"reports/figures/{name}-lr")
    learn_rate = learn.recorder.min_grad_lr * 100
    learn.fit_one_cycle(epochs, slice(learn_rate), callbacks=cbs)

    learn.recorder.plot_losses()
    plt.savefig(f"reports/figures/{name}-losses")
    learn.recorder.plot_metrics()
    plt.savefig(f"reports/figures/{name}-metrics")

    learn.save(f"{name}")
    learn.export(f"{name}/{name}.pkl")
    state_dict = learn.model.state_dict()
    torch.save(state_dict, f"models/{name}/{name}-state-dict")


#     data = get_data(bands, bs=4)
#     model = get_model(arch, bands, pre_name)
#     learn = get_learn(data, lambda pretrained: model, name, weighted, cut=cut)
#     cbs = get_callbacks(learn, name, "f_beta")

#     learn.load(f"{name}-stage-1")
#     learn.unfreeze()
#     learn.lr_find()
#     learn.recorder.plot(suggestion=True, return_fig=True)
#     plt.savefig(f"reports/figures/{name}-lr-stage-2")
#     learn_rate = learn.recorder.min_grad_lr
#     learn.fit_one_cycle(epochs, slice(learn_rate), callbacks=cbs)

#     learn.recorder.plot_losses()
#     plt.savefig(f"reports/figures/{name}-losses-stage-2")
#     learn.recorder.plot_metrics()
#     plt.savefig(f"reports/figures/{name}-metrics-stage-2")

#     learn.save(f"{name}-stage-2")
#     learn.export(f"{name}/{name}-stage-2.pkl")
#     state_dict = learn.model.state_dict()
#     torch.save(state_dict, f"models/{name}/{name}-state-dict-stage-2")


class SloveniaImageList(SegmentationItemList):
    """TODO"""

    def __init__(self, *args, bands: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.bands = bands
        self.copy_new.append("bands")

    def open(self, fn):
        """TODO"""
        rgb_nir_swir = ["B04", "B03", "B02", "B08", "B11", "B12"]
        indices = [rgb_nir_swir.index(band) for band in self.bands]
        data = np.load(fn)[:, :, indices]
        data = pil2tensor(data, np.float32)
        return Image(data)


def get_data(bands, bs):
    """TODO"""
    codes = np.array(
        [
            "No Data",
            "Cultivated Land",
            "Forest",
            "Grassland",
            "Shrubland",
            "Water",
            "Wetland",
            "Tundra",
            "Artificial Surfaces",
            "Bareland",
            "Permanent Snow and Ice",
        ]
    )
    get_y_fn = (
        lambda x: str(x).replace("6band-", "mask-").replace(".npy", ".png")
    )

    data_path = Path("data")
    raw_path = data_path / "raw"
    processed_path = data_path / "processed"
    images_path = raw_path / f"single_frame_arrays/"
    labels_path = processed_path / f"slo_lulc_train_labels.csv"

    label_df = pd.read_csv(labels_path)
    src = (
        SloveniaImageList.from_df(
            label_df, images_path, cols="name", folder="train", bands=bands
        )
        .split_by_idx(list(range(int(0.2 * len(label_df)))))
        .label_from_func(get_y_fn, classes=codes)
    )
    tfms = get_transforms()
    data = src.transform(tfms, tfm_y=True).databunch(bs=bs).normalize()
    return data


def get_model(arch, bands, pre_name):
    """TODO"""
    model = arch(c_in=len(bands), c_out=44)
    state_dict = torch.load(f"models/{pre_name}/{pre_name}-state-dict")
    model.load_state_dict(state_dict)
    return model


def accuracy(inp, target):
    """TODO"""
    target = target.squeeze(1)
    mask = target != 0
    return (inp.argmax(dim=1)[mask] == target[mask]).float().mean()


# https://github.com/azavea/raster-vision-fastai-plugin/blob/master/fastai_plugin/utils.py#L94-L126
def on_batch_end(self, last_output: Tensor, last_target: Tensor, **kwargs):
    """TODO"""
    preds = last_output.argmax(1).view(-1).cpu()
    targs = last_target.view(-1).cpu()
    if self.n_classes == 0:
        self.n_classes = last_output.shape[1]
        self.x = torch.arange(0, self.n_classes)
    cm = ((preds == self.x[:, None]) & (targs == self.x[:, None, None])).sum(
        dim=2, dtype=torch.float32
    )
    if self.cm is None:
        self.cm = cm
    else:
        self.cm += cm


ConfusionMatrix.on_batch_end = on_batch_end


def _weights(self, avg: str):
    weights = self.cm.sum(dim=1) / self.cm.sum()
    weights[0] = 0
    weights /= weights.sum()
    return weights


CMScores._weights = _weights


def get_metrics():
    """TODO"""
    weighted_f1 = FBeta(average="weighted", beta=1)
    return [accuracy, weighted_f1]


def _resnet_split(m: torch.nn.Module):
    return (m[0][6], m[1])


def get_loss_weights(data, learn):
    """TODO"""
    # https://github.com/mapbox/robosat/blob/master/robosat/tools/weights.py#L26-L59
    n = 0
    counts = np.zeros(data.c, dtype=np.int64)

    loader = learn.data.train_dl
    for _, tile in tqdm(loader):
        tile = torch.squeeze(tile)

        tile = np.array(tile, dtype=np.uint8)
        n += tile.shape[0] * tile.shape[1] * tile.shape[2]
        counts += np.bincount(tile.ravel(), minlength=data.c)

    assert n > 0, "dataset with masks must not be empty"

    # Class weighting scheme `w = 1 / ln(c + p)` see:
    # - https://arxiv.org/abs/1707.03718
    #     LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    # - https://arxiv.org/abs/1606.02147
    #     ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation

    probs = counts / n
    weights = 1 / np.log(1.02 + probs)

    weights.round(6, out=weights)
    return weights


def get_learn(data, model, name, weighted, cut):
    """TODO"""
    metrics = get_metrics()
    learn = unet_learner(
        data,
        model,
        split_on=_resnet_split,
        cut=cut,
        metrics=metrics,
        path="models",
        model_dir=name,
        wd=1e-2,
    )
    if weighted:
        weights = get_loss_weights(data, learn)
        learn.loss_fn = CrossEntropyFlat(
            weight=Tensor(weights).cuda(), ignore_index=0
        )
    else:
        learn.loss_fn = CrossEntropyFlat(ignore_index=0)
    learn = learn.to_fp16()
    return learn


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main()
