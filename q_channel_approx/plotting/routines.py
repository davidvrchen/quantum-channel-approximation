"""
Provides some common plotting routines


Info:
    Created on Wed March 13 2024

    @author: davidvrchen
"""

import os
import itertools

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "plot_styles/report_style.mplstyle")

plt.style.use(os.path.join(dirname, filename))


def plot_ess(
    ts, Ess, labels, ax: Axes = None, alpha: float = 1, colors: list[str] = None
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


def compare_ess(ref: tuple, approx: tuple, labels: list[str]):
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
    ts: np.ndarray, rhoss: list[np.ndarray]
) -> plt.axes:

    m = len(rhoss).bit_length() - 1

    fig, ax = plt.subplots()

    for i, rhos in enumerate(rhoss):
        ax.plot(
            ts,
            rhos,
            label=rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$",
        )

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.ylim(0, 1)
    plt.legend()

    return ax


def plot_evolution_individual_qs(ts: np.ndarray, rhoss: list[np.ndarray]) -> plt.axes:
    """Plots the evolution of all rhos as a function of ts
    with some basic formatting.

    Args:
        ts (np.ndarray): times t_i
        rhoss (list[np.ndarray]): list of rho evolutions (for each rhos: rho_i at time t_i
    """

    fig, ax = plt.subplots()

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = itertools.cycle(prop_cycle.by_key()["color"])

    for i, rhos in enumerate(rhoss):
        state = i % 2
        linestyle = "-" if i % 2 == 0 else ":"

        if i % 2 == 0:
            color = next(colors)
        ax.plot(
            ts,
            rhos,
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
