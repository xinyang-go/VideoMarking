import numpy as np
import os
import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    opt = parser.parse_args()

    fas = os.listdir(opt.dir)

    fs = [fn for fn in fas if fn.endswith("txt")]
    for fn in tqdm.tqdm(fs):
        nums = np.loadtxt(f"{opt.dir}/{fn}", ndmin=2)
        filted_nums = []
        for num in nums:
            if num.tolist()[3:] == [0, 0, 0, 0]:
                print(fn)
            else:
                filted_nums.append(num)
        if filted_nums:
            np.savetxt(f"{opt.dir}/{fn}", filted_nums)
        else:
            os.system(f"rm {opt.dir}/{fn}")

    imgs = [fn for fn in fas if fn.endswith("jpg")]
    for img in tqdm.tqdm(imgs):
        if img[:-3]+"txt" not in fas:
            os.system(f"rm {opt.dir}/{img}")
            print(img)