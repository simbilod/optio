import pathlib
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np


def plot_fiber_xposition():
    dirpath = pathlib.Path(__file__).parent / "data"

    fiber_xpositions = []
    s21_max = []

    for filepath_csv in dirpath.glob("*.csv"):
        filepath_yaml = filepath_csv.with_suffix(".yml")
        settings = OmegaConf.load(filepath_yaml)
        df = pd.read_csv(filepath_csv)
        s21 = 10 * np.log10(df.s21m)

        fiber_xposition = settings.settings.fiber_xposition
        fiber_xpositions.append(fiber_xposition)
        s21_max.append(max(s21))

        # plt.plot(10*np.log10(df.s21m))
        # plt.show()

        # filepath = 'data/fiber_2ef8b170.csv'
        # filepath = 'data/fiber_8cbfba96.csv'
        # filepath = 'data/fiber_51cc87da.csv'

    plt.plot(fiber_xpositions, s21_max, ".")
    plt.xlabel("fiber_xposition")
    plt.ylabel("max S21 (dB)")
    plt.show()


if __name__ == "__main__":
    dirpath = pathlib.Path(__file__).parent / "data"

    fiber_xpositions = []
    s21_max = []

    for filepath_csv in dirpath.glob("*.csv"):
        filepath_yaml = filepath_csv.with_suffix(".yml")
        settings = OmegaConf.load(filepath_yaml)
        df = pd.read_csv(filepath_csv)
        s21 = 10 * np.log10(df.s21m)

        fiber_xposition = settings.settings.fiber_xposition
        fiber_xpositions.append(fiber_xposition)
        s21_max.append(max(s21))

        # plt.plot(10*np.log10(df.s21m))
        # plt.show()

        # filepath = 'data/fiber_2ef8b170.csv'
        # filepath = 'data/fiber_8cbfba96.csv'
        # filepath = 'data/fiber_51cc87da.csv'

    plt.plot(fiber_xpositions, s21_max, ".")
    plt.xlabel("fiber_xposition")
    plt.ylabel("max S21 (dB)")
    plt.show()
