"""Module for analyzing fric-frac's results"""
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from matplotlib import pyplot as plt


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


def main():
    """mate this is main"""
    ampli = -5000e-6
    E = 2.59e9
    nu = 0.35
    df = pd.read_csv("load_test_000.csv", skiprows=7, delimiter=";")
    # df = df.rolling(10).mean()
    time = df["Relative time"]
    ng = len(df.columns)//3

    fig, axes = plt.subplots(nrows=3, ncols=ng, sharex=True, sharey='row', gridspec_kw=dict(hspace=0, wspace=0))

    for gauge, axcol in enumerate(axes.T, start=1):
        gauges = [f"Ch{4*(gauge-1)+1:0>2}",f"Ch{4*(gauge-1)+2:0>2}",f"Ch{4*(gauge-1)+3:0>2}"]
        print(f"Processing gauges {gauges} {gauge}/{ng} ({gauge/ng:.0%})", end='\r')
        data = df[gauges].to_numpy().T

        e11, e22, e12 = rotation_matrix(np.deg2rad((45, 90, 135))) @ (ampli * data)
        strains_matrix = np.array([
            [e11, e12, 0*e11],
            [e12, e22, 0*e11],
            [0*e11, 0*e11, -nu/(1-nu)*(e11+e22)]
        ])
        stresses_matrix = hooke(E, nu, strains_matrix)
        strains = strains_matrix[0, 0, :], strains_matrix[1, 1, :], 2*strains_matrix[0, 1, :]
        stresses = stresses_matrix[0, 0], stresses_matrix[1, 1], stresses_matrix[0, 1]

        print("Plotting gauge   ", end='\r')
        for lab, labb, pot, strain, stress in zip((1, 2, 3), (11, 22, 12), data, strains, stresses):
            axcol[0].set_title(f"Rosette {gauge}")
            axcol[0].plot(time/60, pot, label=rf"$\Delta R_{{{lab}}}$")
            axcol[1].plot(time/60, strain*100, label=rf"$\varepsilon_{{{labb}}}$")
            axcol[2].plot(time/60, stress/1e6, label=rf"$\sigma_{{{labb}}}$")
    print()
    axes[0, 0].legend(loc='upper left')
    axes[1, 0].legend(loc='upper left')
    axes[2, 0].legend(loc='upper left')
    axes[0, 0].set_ylabel('Résistance [ohm]')
    axes[1, 0].set_ylabel('Déformation [%]')
    axes[2, 0].set_ylabel('Contrainte [MPa]')
    axes[2, 0].set_xlabel('temps [min]')
    plt.show()


if __name__ == "__main__":
    with plt.style.context("ggplot"):
        main()
