import skimage.io
from dreamfuser.logger import logger
import visual_recog
import visual_words
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import util
import matplotlib
# matplotlib.use('TkAgg')
from dreamfuser.configs import load_config
from dataclasses import dataclass, field

@dataclass
class Config:
    K: int = 200
    alpha: int = 300
    feature_dim: int = 60
    layer_num: int = 3
    format: list = field(default_factory=lambda: ["stdout", "csv", "log", "json"])
    project: str = "SPM"
    nearest_neighbor_num: int = 1
    distance: str = "intersect" # euclidean, intersect, chi2, correl
    sobel: bool = False
    prewitt: bool = False

# config=load_config(Config)
if __name__ == '__main__':
    config = load_config(Config)
    assert config.feature_dim == 60 + config.sobel * 6 + config.prewitt * 6
    job_id = f"K{config.K}-alpha{config.alpha}-fd{config.feature_dim}-layer-num{config.layer_num}-nn{config.nearest_neighbor_num}-distance:{config.distance}-sobel:{config.sobel}-prewitt:{config.prewitt}"
    logger.configure(
        "logs",
        format_strs=config.format,
        config=config,
        project=config.project,
        name=job_id,
    )  # type: ignore
    num_cores = util.get_num_CPU()
    print("number of cores: ", num_cores)
    # path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    # path_img = "../data/waterfall/sun_abcxnrzizjgcwkdn.jpg"
    # path_img = "../data/park/labelme_gmgfmbtnkpwaugo.jpg"
    # image = skimage.io.imread(path_img)
    # image = image.astype('float') / 255
    # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)

    visual_words.compute_dictionary(num_workers=num_cores, config=config)

    dictionary = np.load('dictionary.npy')
    # wordmap = visual_words.get_visual_words(image, dictionary)
    # filename = "wordmap2.jpg"
    # util.save_wordmap(wordmap, filename)
    visual_recog.build_recognition_system(config=config, num_workers=num_cores // 2)
    visual_recog.evaluate_recognition_system(config=config, num_workers=num_cores // 2)

    conf, accuracy = visual_recog.evaluate_recognition_system(
        config=config, num_workers=num_cores)
    print(conf)
    print(np.diag(conf).sum()/conf.sum())
    logger.logkv("accuracy", accuracy)
    np.save(job_id + "_confusion_matrix.npy", conf)
    #visual_recog.get_feature_from_wordmap(wordmap, visual_words.K)
    #visual_recog.get_feature_from_wordmap_SPM(wordmap, 3, visual_words.K)
