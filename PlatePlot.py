from matplotlib import pyplot as plt
from BigBrother import (
    read_with_pandas,
    read_with_polars,
    select_files,
    stress_strain,
    Material
)
import numpy as np


def plot_loop():

    keys = dict(index=0, step=1)
    def onkey(event):
        if event.key == 'right':
            keys["index"] += keys['step']
        elif event.key == 'left':
            keys["index"] -= keys['step']
        elif event.key == 'up':
            keys['step'] += 1
        elif event.key == 'down':
            keys['step'] -= 1
        elif event.key == 'pageup':
            keys['step'] += 100
        elif event.key == 'pagedown':
            keys['step'] -= 100
        else:
            print(f"WARNING: key '{event.key}' not implemented")

        data = [g["strains"][0][keys["index"]] for g in gauge_data]
        print(data)
        ax.set_title(title.format(keys['index']-1, keys['step']))
        points.set_data(x_gauges, data)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

    data = [g["strains"][0][keys["index"]] for g in gauge_data]
    points, = ax.plot(x_gauges, data, '-o', mfc='w')
    ax.set_title(title.format(keys['index']-1, keys['step']))
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.tight_layout()
    plt.show()


x_gauges = np.cumsum((20, 30, 30, 30, 20))
mat = Material(E=2.59e9, nu=0.35)

kind_dict = dict(parallel=0, tangent=1, shear=2)

#files = select_files()
files = ["data/240417/test2_001.csv"]
for file in files:
    df = read_with_pandas(file, sep=";", skiprows=list(range(7))+[8], rolling=100).to_numpy().T
    # df = read_with_polars(file, separator=";", skip_rows=7, skip_rows_after_header=1, rolling=100).to_numpy().T

    fig, ax = plt.subplots()
    title = (
        "Index: {}, Step: {}\n"
        r"advance $\rightarrow$       rewind $\leftarrow$""\n"
        r"step+=1 $\uparrow$          step-=1 $\downarrow$""\n"
        r"step += 100 $\Uparrow$      step-=100$\Downarrow$"
    )

    gauge_data = []
    for gauge, x in enumerate(x_gauges):  # TODO
        print(f"Treating gauge {gauge}/{len(x_gauges)}", end="\r")
        strains, stresses = stress_strain(mat, df[3*gauge+1:3*gauge+4], -5000e-6)
        gauge_data.append(dict(strains=strains, stresses=stresses))
    print(f"\nDone")

    plot_loop()
