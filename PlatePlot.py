"""Module for interactive analysis of the stresses across the plate"""

from matplotlib import pyplot as plt
from pathlib import Path
from BigBrother import (
    read,
    select_files,
    stressdf,
    straindf,
    average_rosette,
    lowfilter,
    triads
)
import numpy as np


keys = dict(index=0, step=1)
def update_keys(event, redraw=False) -> bool:
    if event.key == 'right':
        keys["index"] +=keys['step']
        redraw = True
    elif event.key == 'left':
        keys["index"] -= keys['step']
        redraw = True
    elif event.key == 'up':
        keys['step'] += 10
    elif event.key == 'down':
        keys['step'] -= 10
    elif event.key == 'pageup':
        keys['step'] += 1000
    elif event.key == 'pagedown':
        keys['step'] -= 1000
    return redraw


def plateplot(x_gauges, E, nu, ampli, file, timecol="Relative time"):

    if 'keymap.back' in plt.rcParams:
        plt.rcParams['keymap.back'].remove('left')
        plt.rcParams['keymap.forward'].remove('right')

    fig, (ax, axBB) = plt.subplots(nrows=2)
    ax.set_xlim((0, 250))
    fig.canvas.manager.set_window_title(Path(file).stem)
    title = (
        "Index: {}, Step: {}\n"
        r"advance $\rightarrow$       rewind $\leftarrow$""\n"
        r"step+=10 $\uparrow$          step-=10 $\downarrow$""\n"
        r"step += 1000 $\Uparrow$      step-=1000$\Downarrow$"
    )
    # Read file
    df = read(file, sep=";", skiprows=list(range(7))+[8])
    time = df[timecol]

    # Apply filter
    df = lowfilter(df, cutoff=5, N=2)
    # merged MeanBrother plot
    strains = straindf(df, (45, 90, 135), ampli)
    stress = stressdf(strains, E, nu)

    stress_avg = average_rosette(stress)
    gauge_cols = stress_avg.columns[1:]
    for lab, col in zip(("//", "⊥", "xy"), gauge_cols):
        axBB.plot(time, stress_avg[col]/1e6, label=lab)
    axBB.set_xlabel("Time [s]")
    axBB.set_ylabel(r"$\sigma$ [MPa]")
    vline = axBB.axvline(time[0], ls='-.', c='none', alpha=0.6)
    # axBB.legend(loc="upper left")

    # Transpose to have one list per direction
    gauge_cols = triads(stress.columns[1:])
    dir_cols = [list(columns) for columns in zip(*gauge_cols)]
    # Initiate points plot
    points = []
    for lab, cols in zip(("//", "⊥", "xy"), dir_cols):
        stresses_i = stress[cols].iloc[0]
        points_i, = ax.plot(x_gauges, stresses_i, '-o', mfc='w', label=lab)
        points.append(points_i)

    ax.set_title(title.format(keys['index'], keys['step']))
    ax.set_xlabel("Position on plate")
    ax.set_ylabel(rf"$\sigma(t = {time[keys['index']]:.2e} s)$ [MPa]")
    ax.dataLim.y0 = stress.to_numpy().min()
    ax.dataLim.y1 = stress.to_numpy().max()
    ax.legend(loc="upper right")

    # Add key events
    def move_vlines():
        keys["index"] = max(0, min(keys["index"], time.size-1))
        for i, point_cols in enumerate(dir_cols):
            points[i].set_data(x_gauges, stress[point_cols].iloc[keys["index"]])
        vline.set_xdata(np.full_like(vline.get_xdata(), time[keys["index"]]))
        vline.set(color='gray')
        ax.set_ylabel(rf"$\sigma(t = {time[keys['index']]:.2e} s)$")
        # ax.relim()
        # ax.autoscale_view()
    def onkey(event):
        if update_keys(event) is True:
            move_vlines()
        ax.set_title(title.format(keys['index'], keys['step']))
        fig.canvas.draw()
    def onclick(event):
        if event.dblclick is True:
            keys["index"] = np.abs(time-event.xdata).argmin()
            move_vlines()
            ax.set_title(title.format(keys['index'], keys['step']))
            fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event', onkey)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Définition du problème
    x_gauges = np.array((2, 50, 123, 200, 248))
    E=2.59e9
    nu=0.35
    ampli = -5000e-6

    files = select_files()
    # files = ["data/240417/test2_000.csv"]
    with plt.style.context("ggplot"):
        for file in files:
            plateplot(x_gauges, E, nu, ampli, file)
