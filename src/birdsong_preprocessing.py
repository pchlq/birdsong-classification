import numpy as np
from tqdm import tqdm
import pylab
import librosa
import librosa.display
import logging
import time
import random
from pathlib import Path
import matplotlib.pyplot as plt
import gc
from joblib import Parallel, delayed
import argparse


def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time() - tick
        avg_speed = time_diff / step
        total_str = "of %n" % total if total else ""
        print(
            "step",
            step,
            "%.2f" % time_diff,
            "avg: %.2f iter/sec" % avg_speed,
            total_str,
        )
        step += 1
        yield next(seq)


all_bar_funcs = {
    "tqdm": lambda args: lambda x: tqdm(x, **args),
    "txt": lambda args: lambda x: text_progessbar(x, **args),
    "False": lambda args: iter,
    "None": lambda args: iter,
}


def gen_spec(load_path: Path, save_dir: Path, lfnumber: int):

    level = logging.INFO
    logging.basicConfig(
        filename=save_dir / f"log_file_{lfnumber}.log",
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
    )

    imname = load_path.name[:-4]

    y, sr = librosa.load(load_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2.24, 2.24))
    pylab.axis("off")
    pylab.axes([0.0, 0.0, 1.0, 1.0], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(S)

    (save_dir / load_path.parents[1].name / load_path.parent.name).mkdir(
        parents=True, exist_ok=True
    )
    new_dir = save_dir / load_path.parents[1].name / load_path.parent.name
    save_path = str(new_dir) + "/" + f"{imname}.png"
    logging.info(f"save image: {load_path.parent.name}/{imname}.png")
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0, dpi=100)
    pylab.close()
    del S
    gc.collect()


def ParallelExecutor(use_bar="tqdm", **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun


if __name__ == "__main__":
    """
    Example of usage:
    >>> python birdsong_preprocessing.py --load_dir fold_1 --lfnumber 1 --save_dir spect_images --njobs 6
    """

    parser = argparse.ArgumentParser(description="Converting sound to image")
    parser.add_argument("--lfnumber", help="log file number", type=int)
    parser.add_argument("--load_dir", help="dir to read .wav files", type=str)
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument(
        "--save_dir", help="dir to save .png files", type=str, default="spect_images"
    )

    args = parser.parse_args()
    load_dir = args.load_dir
    save_dir = Path(args.save_dir)
    lfnumber = args.lfnumber
    njobs = args.njobs

    BASE_DIR = Path(__file__).resolve().parent
    list_of_paths = list((BASE_DIR / load_dir).glob("*/*.wav"))

    aprun = ParallelExecutor(n_jobs=njobs)
    _ = aprun(total=len(list_of_paths))(
        delayed(gen_spec)(f, save_dir, lfnumber) for f in list_of_paths
    )
