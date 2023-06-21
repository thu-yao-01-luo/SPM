import os
import time


if __name__ == "__main__":
    K = [250, 300, 350]
    alpha = [350, 400, 450]
    layer_num = [4, 5, 6]
    nearest_neighbor_num = [4, 5, 6]
    # distance = ["intersect", "chi2", "correl", "euclidean"]

    for k in K:
        os.system(f"python main.py K={k}")
    for a in alpha:
        os.system(f"python main.py alpha={a}")
    for l in layer_num:
        os.system(f"python main.py layer_num={l}")
    os.system(f"python main.py K=400 alpha=500 layer_num=5")
    for nn in nearest_neighbor_num:
        os.system(f"python main.py nearest_neighbor_num={nn}")
    # for d in distance:
    #     os.system(f"python main.py distance={d}")
    # os.system(f"python main.py feature_dim=72 sobel=True prewitt=True distance=correl K=400 alpha=500 layer_num=5 nearest_neighbor_num=5")
