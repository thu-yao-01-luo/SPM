import skimage.io
from dreamfuser.logger import logger
import visual_recog
import visual_words
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import util
import matplotlib
matplotlib.use('TkAgg')
from dreamfuser.configs import load_config
from dataclasses import dataclass, field

@dataclass
class Config:
    K: int = 200
    alpha: int = 300
    feature_dim: int = 60
    layer_num: int = 3
    format: list = field(default_factory=lambda: ["stdout", "csv"])
    project: str = "SPM"

if __name__ == '__main__':
    config = load_config(Config)
    logger.configure(
        "logs",
        format_strs=config.format,
        config=config,
        project=config.project,
        name=f"K{config.K}-alpha{config.alpha}-fd{config.feature_dim}-layer-num{config.layer_num}",
    )  # type: ignore
    logger.logkv("test", 1.0)
    logger.logkv("test", 2.0)
    logger.logkv("test", 3.0)
    logger.logkv("test2", 4.0)
    logger.logkv("test2", 5.0)
    logger.logkv("test", 6.0)
    logger.dumpkvs()
    # a = np.array([[1,2,3],[4,5,6]])
    # np.savetxt("foo.csv", a, delimiter=",")
