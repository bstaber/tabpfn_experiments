import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main(n_splits, sample_size):
    df = pd.read_csv("CASP.csv")
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    for split in range(n_splits):
        random_state = split + 1
        random.seed(random_state)
        np.random.seed(random_state)

        test_ratio = 0.2

        # divide the dataset into test and train based on the test_ratio parameter
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )

        # reshape the data
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        # compute input dimensions
        n_train = x_train.shape[0]

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half : 2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])

        # scale
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = (np.squeeze(y_train) / mean_y_train).reshape(-1, 1)
        y_test = (np.squeeze(y_test) / mean_y_train).reshape(-1, 1)

        newpath = "real_world_data/bio/data/"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        np.savez(
            f"real_world_data/bio/data/bio_train_split_{split}.npz",
            X=x_train[idx_train][:sample_size, :],
            y=y_train[idx_train][:sample_size, :],
        )
        np.savez(
            f"real_world_data/bio/data/bio_cal_split_{split}.npz",
            X=x_train[idx_cal][:sample_size, :],
            y=y_train[idx_cal][:sample_size, :],
        )
        np.savez(
            f"real_world_data/bio/data/bio_test_split_{split}.npz",
            X=x_test[:sample_size, :],
            y=y_test[:sample_size, :],
        )

    return "bio"


if __name__ == "__main__":
    main(n_splits=1, sample_size=1000)
