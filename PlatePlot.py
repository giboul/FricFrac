from matplotlib import pyplot as plt
import matplotlib
from pathlib import Path
from BigBrother import (
    read_with_pandas,
    read_with_polars,
    select_files,
    stress_strain,
    Material
)
import numpy as np
matplotlib.rcParams['keymap.back'].remove('left')
matplotlib.rcParams['keymap.forward'].remove('right')


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


def plateplot(x_gauges, mat, ampli, files, polars=False):
    for file in files:
        if polars is True:
            df = read_with_polars(file, separator=";", skip_rows=7, skip_rows_after_header=1, rolling=100).to_numpy().T
        else:
            df = read_with_pandas(file, sep=";", skiprows=list(range(7))+[8], rolling=1).to_numpy().T

        fig, (ax, axBB) = plt.subplots(nrows=2)
        fig.canvas.manager.set_window_title(Path(file).stem)
        title = (
            "Index: {}, Step: {}\n"
            r"advance $\rightarrow$       rewind $\leftarrow$""\n"
            r"step+=10 $\uparrow$          step-=10 $\downarrow$""\n"
            r"step += 1000 $\Uparrow$      step-=1000$\Downarrow$"
        )

        # Reading files
        time = df[0]
        gauge_data = []
        for gauge, x in enumerate(x_gauges):
            print(f"Treating gauge {gauge+1}/{len(x_gauges)}", end="\r")
            strains, stresses = stress_strain(mat, df[3*gauge+1:3*gauge+4], ampli)
            gauge_data.append(np.array(stresses)/1e6)
        gauge_data = np.array(gauge_data)
        print(f"\nDone")

        # merged MeanBrother plot
        strains, stresses = stress_strain(mat, df[-3:], ampli)
        for lab, stress in zip(("Parallel", "Normal", "Shear"), stresses):
            axBB.plot(time, stress/1e6, label=lab)
        axBB.set_xlabel("Time [s]")
        axBB.set_ylabel(r"$\sigma$ [MPa]")
        vline = axBB.axvline(time[0], ls='-.', c='gray', alpha=0.6)
        axBB.legend(loc="upper left")

        # Plot for each gauge
        # First plot
        points = []
        for i, lab in enumerate(("Parallel", "Normal", "Shear")):
            stresses = [g[i][keys["index"]] for g in gauge_data]
            points_i, = ax.plot(x_gauges, stresses, '-o', mfc='w', label=lab)
            points.append(points_i)

        ax.set_title(title.format(keys['index'], keys['step']))
        ax.set_xlabel("Position on plate")
        ax.set_ylabel(rf"$\sigma(t = {time[keys['index']]:.2e} s)$ [MPa]")
        ax.dataLim.y0 = gauge_data.min()
        ax.dataLim.y1 = gauge_data.max()
        # ax.legend(loc="upper right")

        # Add key events
        def move_vlines():
            keys["index"] = max(0, min(keys["index"], time.size-1))
            for i, direction in enumerate(("Parallel", "Normal", "Shear")):
                data = [g[i][keys["index"]] for g in gauge_data]
                points[i].set_data(x_gauges, data)
            vline.set_xdata(np.full_like(vline.get_xdata(), time[keys["index"]]))
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
    x_gauges = np.cumsum((20, 60, 60, 60, 20))
    mat = Material(E=2.59e9, nu=0.35)
    ampli = -5000e-6

    files = select_files()
    # files = ["data/240417/test2_001.csv"]

    plateplot(x_gauges, mat, ampli, files)
