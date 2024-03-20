from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from typing import Tuple, NamedTuple
import numpy as np


def wheatstone(R1: float, R2: float, R3: float, Vin: float, Vout: np.ndarray) -> np.ndarray:
    """Solve for the value of the unknown resistance R3
    https://en.wikipedia.org/wiki/Wheatstone_bridge"""
    return R3 * (Vin*(R1+R2)/(Vin*R1 - Vout*(R1+R2)) - 1)


def cable_deformation(l0: float, S0: float, rho: float, dR: np.ndarray) -> np.ndarray:
    """Computes the deformation 'ε' of a cable from its resistance variation, initial geometry (l0, S0) and resistivity 'rho'"""
    return np.sqrt(l0**2 + dR*S0*l0/rho) / l0 - 1


class Wheatstone(NamedTuple):
    R1: float
    R2: float
    R3: float
    Vin: float


class Cable(NamedTuple):
    l0: float
    S0: float
    rho: float


class Hooke(NamedTuple):
    E: float
    nu: float

    def __matmul__(self, strains: np.ndarray) -> np.ndarray:
        lambd = self.E*self.nu/(1+self.nu)/(1-2*self.nu)
        mu = self.E/(1+self.nu)
        strain = strains.reshape(-1, 3, 3)
        stresses = lambd * np.trace(strain) + 2*mu*strain
        return stresses.reshape(3, 3, -1)


class Gauje:
    bridge: Wheatstone
    cable: Cable

    def __init__(self, bridge: Wheatstone, cable: Cable) -> None:
        self.bridge = bridge
        self.cable = cable

    def dRx(self, Vout: np.ndarray) -> np.ndarray:
        return wheatstone(*self.bridge, Vout) - self.bridge.R3

    def deformation(self, dR: np.ndarray) -> np.ndarray:
        return cable_deformation(*self.cable, dR)


class Rosette:
    gauges: Tuple[Gauje]
    claw: Hooke

    def __init__(self, gauge1: Gauje, gauge2: Gauje, gauge3: Gauje, claw: Hooke) -> None:
        self.gauges = gauge1, gauge2, gauge3
        self.claw = claw

    @staticmethod
    def strains(dl1: np.ndarray, dl2: np.ndarray, dl3: np.ndarray, plane='stress') -> np.ndarray:
        e11 = dl1 + dl3 - dl2
        e12 = dl1 - dl3
        e22 = dl2
        zer = np.zeros(e11.size)
        if plane == 'strain':  # Review this # TODO
            strains2d = np.array([
                [e11, e12, zer],
                [e12, e22, zer],
                [zer, zer, zer],
            ])
        elif plane == "stress":  # Review this # TODO
            zer = np.zeros_like(e11)
            strains2d = np.array([
                [e11, e12, zer],
                [e12, e22, zer],
                [zer, zer, -e11-e22],
            ])
        else:
            raise ValueError(f"Plane '{plane}' is not supported")
        return strains2d

    def stress(self, strains: np.ndarray) -> np.ndarray:
        return self.claw @ strains

    def BigBrother(self, f: float, V1: np.ndarray, V2: np.ndarray, V3: np.ndarray) -> Tuple[Figure, plt.Axes]:

        R1, R2, R3 = self.gauges[0].dRx(V1), self.gauges[1].dRx(V2), self.gauges[2].dRx(V3)
        dl1, dl2, dl3 = self.gauges[0].deformation(R1), self.gauges[1].deformation(R2), self.gauges[2].deformation(R3)
        # strains = self.strains(dl1, dl2, dl3)
        e1, e2, g = -5000 * 1e-6 * np.array((V1, V2, V3))
        zeros = np.zeros(V1.size)
        strains = np.array([
            [e1, g, zeros],
            [g, e2, zeros],
            [zeros, zeros, -e1-e2]
        ])
        stresses = self.claw @ strains

        strains = strains[0, 0], strains[0, 1], strains[1, 1]
        stresses = stresses[0, 0], stresses[0, 1], stresses[1, 1]

        t = np.arange(0, V1.size) / f

        fig, axes = plt.subplots()
        plt.plot(t, stresses[0], label = r"$\sigma_{{11}}$")
        plt.plot(t, stresses[1], label = r"$\sigma_{{12}}$")
        plt.plot(t, stresses[2], label = r"$\sigma_{{22}}$")
        plt.legend()
        plt.xlabel("t [s]")

        # fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True)

        # for axs, lab, v, r, dl, sn, ss in zip(axes, (11, 12, 22), (V1, V2, V3), (R1, R2, R3), (dl1, dl2, dl3), strains, stresses):
        #     axV, axsn, axss = axs
        #     axV.set_ylabel(f"$V_{{G{lab}}}$")
        #     # axR.set_ylabel(rf"$\Delta R_{{x{lab}}}$")
        #     # axl.set_ylabel(rf"$\Delta\ell_{{{lab}}}$")
        #     axsn.set_ylabel(rf"$\varepsilon_{{{lab}}}$")
        #     axss.set_ylabel(rf"$\sigma_{{{lab}}}$")
        #     axV.plot(t, v)
        #     # axR.plot(t, r)
        #     # axl.plot(t, dl)
        #     axsn.plot(t, sn)
        #     axss.plot(t, ss)
        # for ax in axs:
        #     ax.set_xlabel("$t$ [s]")
        # for axs in axes[:-1]:
        #     for ax in axs:
        #         ax.set_xticklabels([])

        return fig, axes

if __name__ == "__main__":

    Vin, V1, V2, V3 = np.loadtxt("Ulog.txt", skiprows=1).T

    bridge = Wheatstone(120, 120, 120, Vin)
    cable = Cable(1e6, 1e-4, 10e-6)
    claw = Hooke(3e9, 0.4)
    gauge = Gauje(bridge, cable)
    rosette = Rosette(gauge, gauge, gauge, claw)

    with plt.style.context('ggplot'):
        rosette.BigBrother(10e6, V1, V2, V3)
        plt.show()
