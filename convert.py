import numpy as np
import os
import argparse
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    opt = parser.parse_args()

    fs = [fn for fn in os.listdir(opt.dir) if fn.endswith("txt")]
    for fn in tqdm.tqdm(fs):
        boxes = np.loadtxt(f"{opt.dir}/{fn}", ndmin=2).astype(np.int)
        color = boxes[:, 1:2] // 7
        boxes[:, 1] = boxes[:, 1] % 7
        boxes = np.concatenate([boxes[:, 0:1], color, boxes[:, 1:]], axis=1)
        np.savetxt(f"{opt.dir}/{fn}", boxes)
