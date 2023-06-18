import os
import time


if __name__ == "__main__":
    K = [100, 200, 300, 400]
    alpha = [200, 300, 400, 500]
    layer_num = [2, 3, 4, 5]
    nearest_neighbor_num = [1, 2, 3, 4, 5]
    distance = ["euclidean", "intersect", "chi2", "correl"]
    filters = [(60, False, False), (66, True, False), (66, False, True), (72, True, True)]

    for k in K:
        os.system(f"python main.py K={k}")
    for a in alpha:
        os.system(f"python main.py alpha={a}")
    for l in layer_num:
        os.system(f"python main.py layer_num={l}")
    os.system(f"python main.py K=400 alpha=500 layer_num=5")
    for nn in nearest_neighbor_num:
        os.system(f"python main.py nearest_neighbor_num={nn}")
    for d in distance:
        os.system(f"python main.py distance={d}")
    for f in filters:
        os.system(f"python main.py feature_dim={f[0]} sobel={f[1]} prewitt={f[2]}")
    # os.system(f"python main.py feature_dim=72 sobel=True prewitt=True distance=correl K=400 alpha=500 layer_num=5 nearest_neighbor_num=5")
        