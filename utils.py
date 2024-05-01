"""Module for analyzing fric-frac's results"""
import numpy as np
import polars as pl
import pandas as pd
from scipy.signal import butter, sosfilt
from numpy.typing import ArrayLike, NDArray
from matplotlib import pyplot as plt
from pathlib import Path
from tkinter import Tk, filedialog


class Temp:
    pass


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


def MeanBrother(file, polars=False):

    if polars:
        cols = pl.read_csv(file, separator=";", skip_rows=7, n_rows=1).columns
        df = pl.read_csv(file, separator=";", skip_rows=9, new_columns=cols, infer_schema_length=None)
    else:
        cols = pd.read_csv(file, sep=";", skiprows=7, nrows=1).columns
        df = pd.read_csv(file, sep=";", skiprows=9, names=cols)

    for col in cols[1:]:
        fs = 1/np.diff(df['Relative time']).mean()
        sos = butter(1, 1.0, 'lowpass', fs=fs, output='sos')
        # df = df.with_columns(**{col: sosfilt(sos, df[col])})
        df[col] = sosfilt(sos, df[col])
    time = df["Relative time"]
    ng = len(df.columns)//3

    mean_gauges = [f"Ch{4*ng+i:0>2}" for i in range(1, 4)]
    if polars:
        df = df.with_columns(**{
            k: pl.sum_horizontal(*[f"Ch{4*g+i:0>2}" for g in range(ng)])/ng
            for i, k in enumerate(mean_gauges, start=1)
        })
    else:
        for i, k in enumerate(mean_gauges, start=1):
            cols = [f"Ch{4*g+i:0>2}" for g in range(ng)]
            df[k] = df[cols].mean(axis=1)
    print(df)

    data = df[mean_gauges].to_numpy().T
    e11, e22, e12 = rotation_matrix(np.deg2rad((45, 90, 135))) @ (ampli * data)
    strains_matrix = np.array([
        [e11, e12, 0*e11],
        [e12, e22, 0*e11],
        [0*e11, 0*e11, -nu/(1-nu)*(e11+e22)]
    ])
    stresses_matrix = hooke(E, nu, strains_matrix)
    strains = strains_matrix[0, 0], strains_matrix[1, 1], np.abs(2*strains_matrix[0, 1])
    stresses = stresses_matrix[0, 0], stresses_matrix[1, 1], np.abs(stresses_matrix[0, 1])

    fig, axes = plt.subplots(nrows=3)
    ax1, ax2, ax3 = axes
    fig.suptitle(file.stem)

    for lab, labb, pot, strain, stress in zip((1, 2, 3), (11, 22, 12), data, strains, stresses):
        ax1.plot(time, pot, label=rf"$\Delta R_{{{lab}}}$")
        ax2.plot(time, strain*100, label=rf"$\varepsilon_{{{labb}}}$")
        ax3.plot(time, stress/1e6, label=rf"$\sigma_{{{labb}}}$")
    print()
    ax1.set_title("Moyenne")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax3.legend(loc='upper left')
    ax1.set_ylabel('Résistance [ohm]')
    ax2.set_ylabel('Déformation [%]')
    ax3.set_ylabel('Contrainte [MPa]')
    ax3.set_xlabel('temps [s]')
    return fig, axes


def BigBrother(file, polars=False, rolling=None):

    # Reading files
    if polars:
        cols = pl.read_csv(file, separator=";", skip_rows=7, n_rows=1).columns
        df = pl.read_csv(file, separator=";", skip_rows=9, new_columns=cols, infer_schema_length=None)
    else:
        cols = pd.read_csv(file, sep=";", skiprows=7, nrows=1).columns
        df = pd.read_csv(file, sep=";", skiprows=9, names=cols)

    # Filtering
    # for col in cols[1:]:
    #     fs = 1/np.diff(df['Relative time']).mean()
    #     sos = butter(1, 1.0, 'lowpass', fs=fs, output='sos')
    #     # df = df.with_columns(**{col: sosfilt(sos, df[col])})
    #     df[col] = sosfilt(sos, df[col])
    if rolling is not None:
        if polars is False:
            df = df.rolling(rolling).mean()
        else:
            df = df.with_columns(**{
                k: pl.col(k).rolling_mean(window_size=rolling)
                for k in df.columns
            })

    time = df["Relative time"]
    ng = len(df.columns)//3

    # Adding columns for mean values
    mean_gauges = [f"Ch{4*ng+i:0>2}" for i in range(1, 4)]
    if polars:
        df = df.with_columns(**{
            k: pl.sum_horizontal(*[f"Ch{4*g+i:0>2}" for g in range(ng)])/ng
            for i, k in enumerate(mean_gauges, start=1)
        })
    else:
        for i, k in enumerate(mean_gauges, start=1):
            cols = [f"Ch{4*g+i:0>2}" for g in range(ng)]
            df[k] = df[cols].mean(axis=1)
    print(df)

    # Figure definition
    fig, axes = plt.subplots(nrows=3, ncols=ng+1, sharex=True, sharey='row', gridspec_kw=dict(hspace=0, wspace=0))
    fig.suptitle(file.stem)
    # Vertical lines
    axlines = [
        [ax.axvline(0., ls='-.', c='gray') for ax in axes[i]]
        for i in range(len(axes))
    ]
    def onclick_GridUpdate(event):
        x, y = event.xdata, event.ydata
        for alrow in axlines:
            for al in alrow:
                al.set_xdata(np.full_like(al.get_xdata(), x))
        fig.suptitle(f"{file.stem}: {x=:.2e}  {y=:.2e}")
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick_GridUpdate)

    # Plotting each gauge
    for gauge, axcol in enumerate(axes.T):
        gauges = [f"Ch{4*gauge+1:0>2}",f"Ch{4*gauge+2:0>2}",f"Ch{4*gauge+3:0>2}"]
        print(f"Processing gauges {gauges} {gauge}/{ng} ({gauge/ng:.0%})", end='\r')
        data = df[gauges].to_numpy().T

        e11, e22, e12 = rotation_matrix(np.deg2rad((45, 90, 135))) @ (ampli * data)
        strains_matrix = np.array([
            [e11, e12, 0*e11],
            [e12, e22, 0*e11],
            [0*e11, 0*e11, -nu/(1-nu)*(e11+e22)]
        ])
        stresses_matrix = hooke(E, nu, strains_matrix)
        strains = strains_matrix[0, 0], strains_matrix[1, 1], np.abs(2*strains_matrix[0, 1])
        stresses = stresses_matrix[0, 0], stresses_matrix[1, 1], np.abs(stresses_matrix[0, 1])

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


def main(*args, **kwargs):
    # Get file paths
    Tk().withdraw()
    directory = Path(filedialog.askdirectory())
    files = filedialog.askopenfilenames(
        initialdir=directory,
        filetypes=[('Comma Separated Values', '.csv')]
    )
    files = [directory/f for f in files]
    # Plot
    with plt.style.context("ggplot"):
        for file in files:
            BigBrother(file, *args, **kwargs)
            # MeanBrother(file, *args, **kwargs)
            plt.show()


if __name__ == "__main__":
    ampli = -5000e-6
    E = 2.59e9
    nu = 0.35
    main(polars=True, rolling=50)
