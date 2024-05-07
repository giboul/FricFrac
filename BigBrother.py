"""Module for analyzing fric-frac's results"""
# Math
import numpy as np
from numpy.typing import ArrayLike, NDArray
# Tables
import polars as pl
import pandas as pd
# Plotting
from matplotlib import pyplot as plt
# Signal processing
from scipy.signal import butter, sosfilt
# File managing
from pathlib import Path
from tkinter import Tk, filedialog
# Classes
from typing import NamedTuple, Iterable


class Material(NamedTuple):
    E: float
    nu: float


def rotation_matrix(angles: ArrayLike) -> NDArray:
    """Rotate stress field by `theta` degrees to be aligned to fric-frac
    s = s_xx*cos(theta)² + s_yy*sin(theta)² + s_xy*sin(2theta)"""
    a, b, c = angles
    return np.array([
        [np.sin(b)*np.sin(c)/(np.sin(a-b)*np.sin(a-c)), -np.sin(a)*np.sin(c)/(np.sin(a-b)*np.sin(b-c)), np.sin(a)*np.sin(b)/(np.sin(a-c)*np.sin(b-c))],
        [np.cos(b)*np.cos(c)/(np.sin(a-b)*np.sin(a-c)), -np.cos(a)*np.cos(c)/(np.sin(a-b)*np.sin(b-c)), np.cos(a)*np.cos(b)/(np.sin(a-c)*np.sin(b-c))],
        [-np.sin(b+c)/(np.sin(a-b)*np.sin(a-c))/2, np.sin(a+c)/(np.sin(a-b)*np.sin(b-c))/2, -np.sin(a+b)/(np.sin(a-c)*np.sin(b-c))/2],
    ])


def plane_stress_matrix(E: float, nu: float) -> NDArray:
    return E / (1-nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)]
    ])


def hooke(E: float, nu: float, strains: NDArray) -> NDArray:
    mu = E/(2*(1+nu))
    lambd = E*nu/((1+nu)*(1-2*nu))
    trace = np.trace(strains)
    strains = strains.T.reshape((-1, 3, 3), order='F')
    z = np.zeros(strains.shape[0])
    trace = np.array([
        [trace, z, z],
        [z, trace, z],
        [z, z, trace]
    ]).T.reshape((-1, 3, 3), order='F')
    return (2*mu*strains + lambd * trace).T.reshape((3, 3, -1), order='F')


def stress_strain(mat, tension, ampli):

    E, nu = mat

    e11, e22, e12 = rotation_matrix(np.deg2rad((45, 90, 135))) @ (ampli * tension)
    strains_matrix = np.array([
        [e11, e12, 0*e11],
        [e12, e22, 0*e11],
        [0*e11, 0*e11, -nu/(1-nu)*(e11+e22)]
    ])
    stresses_matrix = hooke(E, nu, strains_matrix)
    strains = strains_matrix[0, 0], strains_matrix[1, 1], np.abs(2*strains_matrix[0, 1])
    stresses = stresses_matrix[0, 0], stresses_matrix[1, 1], np.abs(stresses_matrix[0, 1])

    return strains, stresses


def read_with_polars(file: Path, ndirs: int=3, drop: Iterable[str]=None, rolling: int=1, *args, **kwargs):

    df = pl.read_csv(file, *args, **kwargs)
    if drop is not None:
        df.drop(*drop)
    # Mean values over window
    df = df.with_columns(**{
        k: pl.col(k).rolling_mean(window_size=rolling)
        for k in df.columns
    })
    df = df.filter(
        ~pl.all_horizontal(pl.col(pl.Float64, pl.Float64).is_null())
    )
    print(df)
    # Add averaged group of columns
    ng = len(df.columns[1:])//ndirs
    mean_gauges = [f"Averaged{i+1:0>2}" for i in range(ndirs)]
    df = df.with_columns(**{
        k: pl.sum_horizontal(*[df.columns[ndirs*g+i] for g in range(ng)])/ng
        for i, k in enumerate(mean_gauges, start=1)
    })
    return df


def read_with_pandas(file: Path, ndirs: int=3, drop: Iterable[str]=None, rolling: int=1, *args, **kwargs):
    df: pd.DataFrame = pd.read_csv(file, *args, **kwargs)
    if drop is not None:
        df = df.drop(drop)
    df = df.rolling(rolling).mean()
    df = df.dropna()
    ng = len(df.columns[1:])//ndirs
    mean_gauges = [f"Averaged{i+1:0>2}" for i in range(ndirs)]
    for i, k in enumerate(mean_gauges, start=1):
        cols = [f"Ch{4*g+i:0>2}" for g in range(ng)]
        df[k] = df[cols].mean(axis=1)
    return df


def MeanBrother(df: pd.DataFrame | pl.DataFrame,
                mat: Material,
                ampli: float):

    E, nu = mat
    time = df["Relative time"]
    data = df[df.columns[-3:]].to_numpy().T

    strains, stresses = stress_strain(mat, data, ampli)

    fig, axes = plt.subplots(nrows=3)
    ax1, ax2, ax3 = axes

    for lab, labb, pot, strain, stress in zip((1, 2, 3), (11, 22, 12), data, strains, stresses):
        ax1.plot(time, pot, label=rf"$\Delta R_{{{lab}}}$")
        ax2.plot(time, strain*100, label=rf"$\varepsilon_{{{labb}}}$")
        ax3.plot(time, stress/1e6, label=rf"$\sigma_{{{labb}}}$")

    ax1.set_title("Moyenne")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax1.set_ylabel('Tension [V]')
    ax2.set_ylabel('Déformation [%]')
    ax3.set_ylabel('Contrainte [MPa]')
    ax3.set_xlabel('temps [s]')

    axlines = [ax.axvline(0., ls='-.', c='gray') for ax in axes]
    def onclick_GridUpdate(event):
        if event.dblclick:
            x, y = event.xdata, event.ydata
            for al in axlines:
                al.set_xdata(np.full_like(al.get_xdata(), x))
            _name = fig._suptitle
            if _name is not None:
                _name = _name.get_text().split(':')[0]
            else:
                _name = "Coordinates"
            fig.suptitle(f"{_name}: {x=:.2e}  {y=:.2e}")
            fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick_GridUpdate)  # TODO double click

    return fig, axes


def BigBrother(df: pd.DataFrame | pl.DataFrame,
               mat: Material,
               ampli: float,
               ndirs=3):

    E, nu = mat
    time = df["Relative time"]
    df = df[df.columns[1:]]
    ng = len(df.columns[1:])//ndirs

    # Figure definition
    fig, axes = plt.subplots(nrows=3, ncols=ng+1, sharex=True, sharey='row', gridspec_kw=dict(hspace=0, wspace=0))
    # Vertical lines
    axlines = [
        [ax.axvline(0., ls='-.', c='gray') for ax in axes[i]]
        for i in range(len(axes))
    ]

    def onclick_GridUpdate(event):
        x, y = event.xdata, event.ydata
        if event.dblclick:
            for alrow in axlines:
                for al in alrow:
                    al.set_xdata(np.full_like(al.get_xdata(), x))
        _name = fig._suptitle
        if _name is not None:
            _name = _name.get_text().split(':')[0]
        else:
            _name = "Coordinates"
        fig.suptitle(f"{_name}: {x=:.2e}  {y=:.2e}")
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick_GridUpdate)  # TODO double click

    # Plotting each gauge
    for gauge, axcol in enumerate(axes.T):
        gauges = df.columns[ndirs*gauge:ndirs*gauge+ndirs]
        print(f"Processing gauges {gauges} {gauge}/{ng} ({gauge/ng:.0%})", end='\r')
        data = df[gauges].to_numpy().T

        strains, stresses = stress_strain(mat, data, ampli)

        print("Plotting gauges  ", end='\r')
        for lab, labb, pot, strain, stress in zip((1, 2, 3), (11, 22, 12), data, strains, stresses):
            axcol[0].set_title(f"Rosette {gauge}")
            axcol[0].plot(time, pot, label=rf"$\Delta U_{{{lab}}}$")
            axcol[1].plot(time, strain*100, label=rf"$\varepsilon_{{{labb}}}$")
            axcol[2].plot(time, stress/1e6, label=rf"$\sigma_{{{labb}}}$")
    print()
    axcol[0].set_title("Moyenne")
    axes[0, 0].legend(loc='upper left')
    axes[1, 0].legend(loc='upper left')
    axes[2, 0].legend(loc='upper left')
    axes[0, 0].set_ylabel('Tension [V]')
    axes[1, 0].set_ylabel('Déformation [%]')
    axes[2, 0].set_ylabel('Contrainte [MPa]')
    axes[2, 0].set_xlabel('temps [s]')
    for ax in axes[-1, :]:
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=60)

    return fig, axes


def select_files():
    Tk().withdraw()
    directory = Path(filedialog.askdirectory())
    files = filedialog.askopenfilenames(
        initialdir=directory,
        filetypes=[('Comma Separated Values', '.csv')]
    )
    return [directory/f for f in files]



def main():
    mat = Material(E=2.59e9, nu=0.35)
    files = select_files()

    with plt.style.context("ggplot"):
        for file in files:
            # df = read_with_polars(
            #     file,
            #     separator=";",
            #     skip_rows=7,
            #     skip_rows_after_header=1
            # )
            df = read_with_pandas(
                file,
                sep=";",
                skiprows=list(range(7))+[8],
                rolling=100
            )
            print(df)
            # BigBrother(df, mat, -5000e-6)
            MeanBrother(df, mat, -5000e-6)
            plt.show()


if __name__ == "__main__":
    main()
