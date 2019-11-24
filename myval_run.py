import numpy as np

import os
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

import sys
import time
from os.path import dirname, realpath

file_path = realpath(__file__)
dir_of_file = dirname(file_path)
parent_dir_of_file = dirname(dir_of_file)
sys.path.insert(0, parent_dir_of_file)

from deepaugment.run_full_model import run_full_model
from deepaugment.build_features import DataOp


def main():

    X, y, input_shape = DataOp.load("cifar10")
    start = time.time()
    run_full_model(
        X, y, test_proportion=0.1,
        model="wrn_28_10", epochs=100, batch_size=32,
        policies_path="/home/acb11354uz/B4new/deep/lib/python3.6/site-packages/reports/experiments/11-02_21-47/best_policies.csv"
    )
    total = time.time() - start
    print(total)

if __name__ == "__main__":
    main()

