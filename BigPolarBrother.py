"""Module for analyzing fric-frac's results"""
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from typing import List
from numpy.typing import ArrayLike, NDArray
from tkinter import Tk, filedialog, messagebox
from scipy.signal import butter, filtfilt
from pathlib import Path
import logging


logging.basicConfig(
    format="%(levelname)s %(asctime)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()


def rotation_matrix(angles: ArrayLike) -> NDArray:
    """
    Rotate stress field by `theta` degrees to be aligned to fric-frac.
    s = s_xx*cos(theta)² + s_yy*sin(theta)² + s_xy*sin(2theta)

    Example of a deformation gauge pointed upwards:

    Example
    -------

    135°  90°  45°
     \    |    /    >>> rotation_matrix(angles=(45, 90, 135))
      \   |   /
       \  |  / 
        \ | /
          O

    Parameters
    ----------
    angles: ArrayLike
        The 3 angles of the deformation wires.

    Return
    ------
    NDArray
        The rotation matrix (3x3) to apply on stresses of strains.
    """
    a, b, c = np.deg2rad(angles)
    return np.array([
        [np.sin(b)*np.sin(c)/(np.sin(a-b)*np.sin(a-c)),
         -np.sin(a)*np.sin(c)/(np.sin(a-b)*np.sin(b-c)),
         np.sin(a)*np.sin(b)/(np.sin(a-c)*np.sin(b-c))],
        [np.cos(b)*np.cos(c)/(np.sin(a-b)*np.sin(a-c)),
         -np.cos(a)*np.cos(c)/(np.sin(a-b)*np.sin(b-c)),
         np.cos(a)*np.cos(b)/(np.sin(a-c)*np.sin(b-c))],
        [-np.sin(b+c)/(np.sin(a-b)*np.sin(a-c))/2,
         np.sin(a+c)/(np.sin(a-b)*np.sin(b-c))/2,
         -np.sin(a+b)/(np.sin(a-c)*np.sin(b-c))/2],
    ])


def plane_stress_matrix(E: float, nu: float) -> NDArray:
    """
    Create the plane stress matrix for linear elasticity.
    ┌    ┐             ┌              ┐ ┌    ┐
    |s_11|             | 1   nu   0   | |e_11|
    |s_22| = E/(1-nu²).| nu  1    0   |.|e_22|
    |t_12|             | 0   0  (1-nu)| |e_12|
    └    ┘             └              ┘ └    ┘

    Parameters
    ----------
    E: float
        Young's modulus.
    nu: float
        Poisson ration.

    Return
    ------
    NDArray
        The plane stress matrix.
    """
    return E / (1-nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)]
    ])


def _usual_gauge_channels(columns: List[str]) -> List[str]:
    """
    Just to make my life easier.
    Removes all channels whose id is a multiple of 4.
    Then groups the colums by groups of 3.

    Parameters
    ----------
    columns: List[str]
        The columns to treat.

    Return
    ------
    List[List[str]]
        The true columns.
    """
    # Filter out multiples of 4
    gauge_channels = [g for g in columns if g.removeprefix(
        "Ch").isdecimal() and int(g.removeprefix("Ch")) % 4 != 0]
    # Group by triads
    gauge_channels = [gauge_channels[3*i:3*i+3]
                      for i, _ in enumerate(gauge_channels[::3])]

    return gauge_channels


def read(file: str, gauge_channels: List[List[str]] = None, *csv_args, **csv_kwargs):
    """
    Read a csv file containing groups of gauge results.
    The file should be similar to this (with the time column FIRST):

    ┌───────────────┬───────────┬──────┬──────┬───┬───────────┬──────┬───────────┐
    │ Relative time ┆ Ch01      ┆ Ch02 ┆ Ch03 ┆ … ┆ Ch18      ┆ Ch19 ┆ Ch20      │
    │ ---           ┆ ---       ┆ ---  ┆ ---  ┆ … ┆ ---       ┆ ---  ┆ ---       │
    │ f64           ┆ f64       ┆ f64  ┆ f64  ┆ … ┆ f64       ┆ f64  ┆ f64       │
    ╞═══════════════╪═══════════╪══════╪══════╪═══╪═══════════╪══════╪═══════════╡
    │ -0.005        ┆ -0.177943 ┆ -2.0 ┆ -2.0 ┆ … ┆ -0.367605 ┆ -2.0 ┆ 0.017335  │
    │ -0.004998     ┆ -0.263445 ┆ -2.0 ┆ -2.0 ┆ … ┆ -0.43433  ┆ -2.0 ┆ 0.013566  │
    │ …             ┆ …         ┆ …    ┆ …    ┆ … ┆ …         ┆ …    ┆ …         │
    │ 0.0049975     ┆ -0.20223  ┆ -2.0 ┆ -2.0 ┆ … ┆ -0.367605 ┆ -2.0 ┆ 0.000502  │
    │ 0.0049995     ┆ -0.3772   ┆ -2.0 ┆ -2.0 ┆ … ┆ -0.354459 ┆ -2.0 ┆ -0.006029 │
    └───────────────┴───────────┴──────┴──────┴───┴───────────┴──────┴───────────┘

    Parameters
    ----------
    file: str | Path
        The name of the file to read.
    gauge_channels: List[List[str]]
        The name of the columns to use, grouped by gauge.
    *csv_args
        Passed to pandas.read_csv.
    **csv_kwargs
        Pased to pandas.read_csv.

    Return
    ------
    pl.DataFrame
        The DataFrame with the tension measures for each gauge channel.
    """

    logger.info("Reading the DataFrame.")
    df = pl.read_csv(file, *csv_args, **csv_kwargs)

    if gauge_channels is None:
        gauge_channels = _usual_gauge_channels(df.columns)
        logger.warning(f"Default channels are used.\n\t>>> {gauge_channels = }")

    flattened_gauge_channels = [
        gn for gnames in gauge_channels for gn in gnames]
    df = df[[df.columns[0]] + flattened_gauge_channels]

    if not all(len(gns) == 3 for gns in gauge_channels):
        raise ValueError(
            "Not all gauges have the same number of columns.\n"f"{gauge_channels = }")

    return df


def lowfilter(df: pl.DataFrame, cutoff: float = 5, N: int = 2, time_col: str = "Relative time") -> NDArray:
    """
    Filtering high frequencies in the signal to remove noise.

    Parameters
    ----------
    df: pl.DataFrame
        The DataFrame to filter
    cutoff: float
        The cutof frequency
    N: int
        The order of the filter

    Return
    ------
    pl.DataFrame
        With the filtered time-series
    """
    sample_frequency = 1/np.diff(df[time_col]).mean()
    b, a = butter(N, cutoff, fs=sample_frequency)

    logger.info("Smoothing DataFrame.\n"
                f"\t>>> {sample_frequency = }\n"
                f"\t>>> {cutoff = }")

    cols = df.columns
    cols.remove(time_col)

    for col in cols:
        df = df.with_columns(**{col: filtfilt(b, a, df[col])})

    return df


def straindf(tensions: pl.DataFrame, angles: ArrayLike, amplification: float, gauge_channels: List[List[str]] = None, time_col: str = "Relative time") -> pl.DataFrame:
    """
    Convert the tension measures into strains through the amplification factor and the rotation matrix.

    Parameters
    ----------
    tensions: pl.DataFrame
        The DataFrame contaning the tension for each channel.
    angles: ArrayLike
        The angles in which the gauges are oriented.
    amplification: float
        The amplification factor (`volt_to_epsilon`).
    gauge_channels: List[List[str]]
        To specify non-standard columns.
    time_col: str = "Relative time"
        To specify the the time column.

    Return
    ------
    pl.DataFrame
        The dataframe containing the strain for each rosette, in each direction (xx, yy, xy).
    """
    logger.info("Computing all strains.\n"
                f"\t>>> amplification factor = {amplification}\n"
                f"\t>>> {angles = }\n"
                f"\t>>> {gauge_channels = }")

    strains = pl.DataFrame(tensions[time_col])
    rot_mat = rotation_matrix(angles)

    if gauge_channels is None:
        gauge_channels = _usual_gauge_channels(tensions.columns)

    for channels in gauge_channels:
        elongations = tensions[channels] * amplification
        strains[channels] = np.einsum('ij, kj -> ki', rot_mat, elongations)

    return strains


def stressdf(strains: pl.DataFrame, E: float, nu: float, time_col: str = "Relative time") -> pl.DataFrame:
    """
    Convert the strains into stresses through Hooke's law in plane stress.

    Parameters
    ----------
    strains: pl.DataFrame
        The DataFrame contaning the strains for each gauge, for each direction.
    E: float
        Young's modulus.
    nu: float
        Poisson ration.
    time_col: str = "Relative time"
        To specify the the time column

    Return
    ------
    pl.DataFrame
        The dataframe containing the stresses for each rosette, in each direction (xx, yy, xy)
    """
    logger.info(f"INFO: Computing all stresses.\n\t>>> {E, nu = }")
    stresses = pl.DataFrame(strains[time_col])
    hooke = plane_stress_matrix(E, nu)

    gauge_channels = [strains.columns[3*i+1:3*(i+1)+1]
                      for i, _ in enumerate(strains.columns[1::3])]

    for channels in gauge_channels:
        stresses[channels] = np.einsum('ij,kj->ki', hooke, strains[channels])

    return stresses


def BigBrother(strains: pl.DataFrame, stresses: pl.DataFrame, timecol: str = "Relative time"):
    """
    Plot stresses and strains for all gauges and their averaged values.

    Parameters
    ----------
    strains: pl.DataFrame
        The DataFrame with the first columns being the time and the rest being the strains.
    stresses: pl.DataFrame
        The DataFrame with the first columns being the time and the rest being the stresses.
    timecol: str = None
        The name of the time column if it is not first

    Return
    ------
    matplotlib.Figure
    Tuple[matplotlib.Axes]

    Plot
    ----
    The strains and stresses grouped by gauge
    """

    logger.info("Time to plots all gauges separately.")

    time = strains[timecol]
    cols = strains.columns
    cols.remove(timecol)
    gauge_columns = [cols[3*i:3*(i+1)] for i, _ in enumerate(cols[::3])]

    fig, axes = plt.subplots(nrows=2,
                             ncols=len(gauge_columns),
                             sharex=True,
                             sharey='row',
                             gridspec_kw=dict(hspace=0, wspace=0),
                             figsize=(10, 5))
    # Vertical lines
    axlines = [[ax.axvline(0., ls='-.', c='none') for ax in axe]
               for axe in axes]

    def onclick_GridUpdate(event):
        """Function to redraw figure on click events"""
        # Retrieve clicked coordinates
        x, y = event.xdata, event.ydata
        if event.dblclick:
            # Update all vertical lines
            for alrow in axlines:
                for al in alrow:
                    al.set_xdata(np.full_like(al.get_xdata(), x))
                    al.set(color='gray')
        # Update title to include cliked coordinates
        _name = fig._suptitle
        if _name is not None:
            _name = _name.get_text().split(':')[0]
        else:
            _name = "Coordinates"
        fig.suptitle(f"{_name}: {x=:.2e}  {y=:.2e}")
        # Redraw figure
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick_GridUpdate)

    directions = (r"//", "⊥", "xy")
    # Plotting each gauge
    for gauge, (cols, axcol) in enumerate(zip(gauge_columns, axes.T), start=1):

        gauge_strains = strains[cols].to_numpy().T
        gauge_stresses = stresses[cols].to_numpy().T

        for lab, strain, stress in zip(directions, gauge_strains, gauge_stresses):
            axcol[0].set_title(f"Rosette {gauge}")
            axcol[0].plot(time, strain*100, label=rf"$\varepsilon_{{{lab}}}$")
            axcol[1].plot(time, stress/1e6, label=rf"$\sigma_{{{lab}}}$")

    axes[0, 0].legend(loc='upper left')
    axes[1, 0].legend(loc='upper left')
    axes[0, 0].set_ylabel('Déformation [%]')
    axes[1, 0].set_ylabel('Contrainte [MPa]')
    axes[1, 0].set_xlabel('temps [s]')
    # Setting up the xlims (more complicated than it should be...)
    tmin, tmax = time.min(), time.max()
    axes[0, 0].set_xticks(
        [t for t in axes[0, 0].get_xticks() if tmin <= t <= tmax])
    axes[0, 0].dataLim.x0 = time.min()
    axes[0, 0].dataLim.x1 = time.max()
    # Rotate ticks by 60°
    for ax in axes[-1, :]:
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=60)

    return fig, axes


def average_rosette(df: pl.DataFrame, timecol="Relative time") -> pl.DataFrame:

    time = df[timecol]
    columns = df.columns
    columns.remove(timecol)
    gauge_columns = [columns[3*i:3*(i+1)] for i, _ in enumerate(columns[::3])]
    # Transpose gauge columns for easier indexing later
    gauge_columns = [list(columns) for columns in zip(*gauge_columns)]

    # Averaging
    df = pl.DataFrame({timecol: time} | {
        f"Av0{i}": df[cols].sum_horizontal()/len(cols)
        for i, cols in enumerate(gauge_columns, start=1)
    })

    return df


def MeanBrother(strains: pl.DataFrame, stresses: pl.DataFrame, timecol: str = "Relative time"):
    """
    Plot stresses and strains for averaging all rosettes, each gauge.

    Parameters
    ----------
    strains: pl.DataFrame
        The DataFrame with the first columns being the time and the rest being the strains (averaged values).
    stresses: pl.DataFrame
        The DataFrame with the first columns being the time and the rest being the stresses (averaged values).
    timecol: str = None
        The name of the time column if it is not first

    Return
    ------
    matplotlib.Figure
    Tuple[matplotlib.Axes]

    Plot
    ----
    The averged strains and stresses over all gauges
    """

    logger.info("Time to plots the averaged stresses and strains.")

    time = strains[timecol]
    columns = strains.columns
    columns.remove(timecol)

    fig, (ax1, ax2) = plt.subplots(nrows=2,
                                   sharex=True,
                                   sharey='row',
                                   gridspec_kw=dict(hspace=0, wspace=0))
    # Vertical lines
    axlines = [ax.axvline(0., ls='-.', color='none') for ax in (ax1, ax2)]

    def onclick_GridUpdate(event):
        """Function to redraw figure on click events"""
        # Retrieve clicked coordinates
        x, y = event.xdata, event.ydata
        if event.dblclick:
            # Update all vertical lines
            for axl in axlines:
                axl.set_xdata(np.full_like(axl.get_xdata(), x))
                axl.set(color='gray')
        # Update title to include cliked coordinates
        _name = fig._suptitle
        if _name is not None:
            _name = _name.get_text().split(':')[0]
        else:
            _name = "Coordinates"
        if x is not None and y is not None:  # Essentially, if the click is in the plot
            fig.suptitle(f"{_name}: {x=:.2e}  {y=:.2e}")
        # Redraw figure
        fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick_GridUpdate)

    # Plot each gauge
    directions = ("//", "⊥", "xy")
    strains = strains[columns].to_numpy().T
    stresses = stresses[columns].to_numpy().T
    for lab, strain, stress in zip(directions, strains, stresses):
        ax1.plot(time, strain*100, label=rf"$\varepsilon_{{{lab}}}$")
        ax2.plot(time, stress/1e6, label=rf"$\sigma_{{{lab}}}$")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    ax1.set_ylabel('Déformation [%]')
    ax2.set_ylabel('Contrainte [MPa]')
    ax2.set_xlabel('temps [s]')
    ax1.set_xlim((time.min(), time.max()))

    return fig, (ax1, ax2)


def select_files():
    """
    Select an array of files and return the absolute paths.
    A message box prompting the user to select more files (potentially from another directory) will appear at each selection.

    Return
    ------
    List[str]
        The selected absolute paths
    """
    root = Tk()
    root.withdraw()

    def main():
        files = filedialog.askopenfilenames(parent=root, title='Choose files')
        msgbox = messagebox.askquestion(
            'Add files',
            'add extra files',
            icon='question'
        )
        return files, msgbox

    files, msgbox = main()

    while msgbox == 'yes':
        files_2, msgbox = main()
        files += files_2

    root.destroy()
    return files


def main(E: float = 2.59e9, nu: float = 0.35, angles=(45, 90, 135), amplification=-5000e-6):

    files = select_files()
    # files = ["data/240410/1bar_000.csv"]

    with plt.style.context("ggplot"):
        for file in files:

            tensiondf = read(file,
                             separator=";",
                             skip_rows=7,
                             skip_rows_after_header=1)
            tensiondf = lowfilter(tensiondf, cutoff=500, N=3)

            if False:
                gauge_channels = None
                plot_func = BigBrother
            else:
                tensiondf = average_rosette(tensiondf)
                gauge_channels = [["Av01", "Av02", "Av03"]]
                plot_func = MeanBrother

            strains = straindf(tensiondf,
                               angles,
                               amplification,
                               gauge_channels)
            stresses = stressdf(strains, E, nu)

            fig, ax = plot_func(strains[::10], stresses[::10])
            fig.suptitle(Path(file).stem)
            fig.savefig("Magnificent plot.pdf", bbox_inches='tight')
            plt.show()


if __name__ == "__main__":
    main()
