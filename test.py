import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import Wheatstone, Cable, Hooke, Gauje, Rosette

df = pd.read_csv("load_test_000.csv", skiprows=7, sep=';', low_memory=False, dtype=float)
# df = df[df["Relative time"] <= 450]

gauge = 1
gauges = [f"Ch{4*(gauge-1)+1:0>2}",f"Ch{4*(gauge-1)+2:0>2}",f"Ch{4*(gauge-1)+3:0>2}"]
print(f"{gauges = }")

V1, V2, V3 = df[gauges].to_numpy().T
Vin = np.full_like(V2, 2)

bridge = Wheatstone(120, 120, 120, Vin)
cable = Cable(1e6, 1e-4, 10e-6)
claw = Hooke(2.59e9, 0.35)
gauge = Gauje(bridge, cable)
rosette = Rosette(gauge, gauge, gauge, claw)

with plt.style.context('ggplot'):
    rosette.BigBrother(1e3, V1, V2, V3)
    plt.show()
