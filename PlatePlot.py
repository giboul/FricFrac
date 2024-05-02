from matplotlib import pyplot as plt
from BigBrother import (
    read_with_pandas,
    select_files,
    stress_strain,
    Material
)
import pandas as pd
import numpy as np


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

    ax.set_title(title.format(keys['index']-1, keys['step']))
    line.set_data(x[:keys["index"]], y[:keys["index"]])
    fig.canvas.draw()

x_gauges = np.cumsum((20, 30, 30, 30, 20))
mat = Material(E=2.59e9, nu=0.35)

files = select_files()
for file in files:
    df = read_with_pandas(file, sep=";", skiprows=list(range(7))+[8]).to_numpy().T

    fig, ax = plt.subplots()
    title = (
        "Index: {}, Step: {}\n"
        r"advance $\rightarrow$       rewind $\leftarrow$""\n"
        r"step+=1 $\uparrow$          step-=1 $\downarrow$""\n"
        r"step += 100 $\Uparrow$      step-=100$\Downarrow$"
    )

    for gauge, x in enumerate(x_gauges):  # TODO
        strains, stresses = stress_strain(mat, df[3*gauge+1:3*gauge+4], -5000e-6)
        points = ax.scatter(x_gauges, strains[keys["index"]][0])

    ax.set_title(title.format(keys['index']-1, keys['step']))
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.tight_layout()
    plt.show()