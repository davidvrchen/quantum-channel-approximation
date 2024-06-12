"""
Provides some common plotting routines.
"""

import os
import itertools

from matplotlib.patches import Patch
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

# When using custom style
style = "presentation"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, f"plot_styles/{style}.mplstyle")
plt.style.use(os.path.join(dirname, filename))

# to use a predefined style
# plt.style.use("default")


# Legacy plotting, left to not break old stuff
def plot_ess(
    ts,
    Ess,
    labels,
    ax: Axes = None,
    alpha: float = 1,
    colors: list[str] = None,
) -> Axes:

    if ax is None:
        ax = plt.gca()

    if colors is None:
        for k, Es in enumerate(Ess):
            ax.plot(ts, Es, label=rf"{labels[k]}", alpha=alpha)
    else:
        for k, Es in enumerate(Ess):
            ax.plot(ts, Es, label=rf"{labels[k]}", alpha=alpha, c=colors[k])

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    # plt.ylim(0, 1)
    plt.legend()
    return ax


def compare_ess(ref: tuple, approx: tuple, labels: list[str]) -> Axes:
    """ref is a tuple (ts, Ess, name),
    approx is similarly (ts, Ess, name)
    """
    ts_ref, Ess_ref, name_ref = ref
    ts_approx, Ess_approx, name_approx = approx

    fig, ax = plt.subplots()

    for k, Es in enumerate(Ess_approx):
        ax.plot(ts_approx, Es, label=rf"{labels[k]}", linestyle=":")
    plt.gca().set_prop_cycle(None)
    for k, Es in enumerate(Ess_ref):
        ax.plot(ts_ref, Es, label=rf"{labels[k]}", linestyle="-")

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.suptitle("Evolution", weight="bold")
    plt.title(f"{name_approx}: dashed line, {name_ref}: solid line")
    # plt.ylim(0, 1)
    plt.legend()
    return ax


def plot_evolution_computational_bs(
    ts: np.ndarray,
    Ess: list[np.ndarray],
) -> Axes:

    m = len(Ess).bit_length() - 1

    for i, Es in enumerate(Ess):
        plt.plot(
            ts,
            Es,
            label=rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$",
        )

    # some formatting to make plot look nice
    plt.ylabel("Population")
    plt.xlabel("Time")
    plt.ylim(0, 1)
    plt.legend()

    return plt.gca()


def plot_evolution_individual_qs(ts: np.ndarray, Ess: list[np.ndarray]) -> Axes:
    """Plots the evolution of all rhos as a function of ts
    with some basic formatting.

    Args:
        ts (np.ndarray): times t_i
        rhoss (list[np.ndarray]): list of rho evolutions (for each rhos: rho_i at time t_i
    """

    fig, ax = plt.subplots()

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = itertools.cycle(prop_cycle.by_key()["color"])

    for i, Es in enumerate(Ess):
        state = i % 2
        linestyle = "-" if i % 2 == 0 else ":"

        if i % 2 == 0:
            color = next(colors)
        ax.plot(
            ts,
            Es,
            label=rf"$q_{i//2} : |{state}\rangle \langle{state}|$",
            linestyle=linestyle,
            color=color,
        )

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.ylim(0, 1)
    plt.legend()

    return ax


# New skool plotting, for report and presentation
def plot_in_computational_bs(
    ts: np.ndarray,
    Ess: list[np.ndarray],
    marker: str,
    linestyle: str,
    alpha: float,
) -> Axes:

    for Es in Ess:
        plt.plot(
            ts,
            Es,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )

    return plt.gca()


def plot_approx(ts, Ess) -> Axes:
    plt.gca().set_prop_cycle(None)
    return plot_in_computational_bs(ts, Ess, marker="o", linestyle="none", alpha=0.6)


def plot_ref(ts, Ess) -> Axes:
    plt.gca().set_prop_cycle(None)
    return plot_in_computational_bs(ts, Ess, marker="none", linestyle=":", alpha=1)


def legend_comp(m: int):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_items = [
    Patch(
        color=colors[j],
        label=rf"$|{format(j, f'0{m}b')}\rangle \langle{format(j, f'0{m}b')}|$",
    )
    for j in range(2**m)
]
    return legend_items
