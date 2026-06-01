import warnings
import os
import nemo.tools
import nemo.analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.path import Path
from scipy.interpolate import interp1d
from scipy.linalg import expm
import pandas as pd
from IPython.display import display
try:
    from colour import (
        SpectralDistribution,
        SDS_ILLUMINANTS,
        sd_to_XYZ,
        XYZ_to_sRGB,
    )
except ImportError:
    SpectralDistribution = None
    SDS_ILLUMINANTS = None
    sd_to_XYZ = None
    XYZ_to_sRGB = None


# pylint: disable=unbalanced-tuple-unpacking

THECOLOR = "black"
cmap = plt.get_cmap("cividis")


def set_fontsize(ax):
    fig_size = ax.get_figure().get_size_inches()
    # define font size based dynamically on figure size
    fontsize = max(fig_size[0] * 100 / 72, 14)
    return fontsize


def check(ax, xmin, xmax):
    x = sorted([xmin, xmax])
    y = None
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xs = list(sorted(vert[:, 0]))
            if xs == x and 0 not in vert[:, 1]:
                y = vert[1, 1]
        except (IndexError, AttributeError):
            pass
    return y


def fill(ax, xmin, xmax, y, text):
    fontsize = set_fontsize(ax)
    newy = check(ax, xmin, xmax)
    try:
        ax.fill_between([xmin, xmax], y, newy, alpha=0.22, hatch="x", color=cmap(0.5))
        txt_x = xmin + (xmax - xmin) / 2
        for txt in ax.texts:
            if txt.get_position()[0] == txt_x and txt.get_position()[1] != -0.4:
                txt.set_visible(False)
        ax.text(
            x=txt_x,
            y= 0.95 * min(newy, y),
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )
    except TypeError :
        ax.text(
            x=xmin + (xmax - xmin) / 2,
            y=0.95 * y,
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )

def format_number(rate, error_rate, unit="s^-1"):
    # Check if the rate is zero
    if rate <= 1e-99:
        return f"0 ± 0 {unit}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = np.floor(np.nan_to_num(np.log10(rate)))

    # Adjust exponent to ensure the first number is >= 1
    if rate / 10**exp < 1:
        exp -= 1

    # Determine the number of significant figures for rate and error_rate
    if np.isnan(error_rate) or error_rate == 0:
        rate_sig_figs = 2
        error_rate_sig_figs = 2  # No error rate provided
    else:    
        rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure
        error_rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure

    # Format the string without using LaTeX
    if exp != 0:
        formatted_rate = f"{rate/10**exp:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate/10**exp:.{error_rate_sig_figs}f}"
        formatted_string = f"({formatted_rate} ± {formatted_error_rate}) x 10^{int(exp)} {unit}"
    else:
        formatted_rate = f"{rate:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate:.{error_rate_sig_figs}f}"
        formatted_string = f"{formatted_rate} ± {formatted_error_rate} {unit}"

    return formatted_string

def format_rate(rate, error_rate, unit="$s^{-1}$"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = np.floor(np.nan_to_num(np.log10(rate)))

    # Adjust exponent to ensure the first number is >= 1
    if rate / 10**exp < 1:
        exp -= 1

    # Determine the number of significant figures for rate and error_rate
    if np.isnan(error_rate) or error_rate == 0:
        rate_sig_figs = 2
        error_rate_sig_figs = 2  # No error rate provided
    else:    
        rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure
        error_rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure

    if exp != 0:
        formatted_string = f"${rate/10**exp:.{rate_sig_figs}f}\\pm{error_rate/10**exp:.{error_rate_sig_figs}f}\\times10^{{{exp:.0f}}}$ " + unit
    else:
        formatted_string = f"${rate:.{rate_sig_figs}f}\\pm{error_rate:.{error_rate_sig_figs}f}$ " + unit

    return formatted_string

#def format_rate(rate, error_rate, unit="$s^{-1}$"):
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        exp = np.round((np.nan_to_num(np.log10(rate))),0)
#    if exp < -100:
#        exp = -100
#    if exp != 0:
#        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}\\times10^{{{exp:.0f}}}$ "+unit
#    else:
#        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}$ "+unit
#    return formatted_string


class State:
    def __init__(self):
        self.smin, self.smax = 0, 3.3
        self.tmin, self.tmax = 3.7, 7
        #singlets = [i for i in alvos if "S" in i and "0" not in i]
        #triplets = [i for i in alvos if "T" in i and "0" not in i]
        self.size = 0.5

    def color(self, state):
        num = int(state[1:])
        if "S" in state:
            return cmap(0.1 + (num - 1) * 0.4 / 3)
        else:
            return cmap(0.9 - (num - 1) * 0.4 / 3)

    def x(self, state):
        factor = 0.1
        num = int(state[1:])
        if "S" in state:
            xmin = self.smax - num * (1 + factor) * self.size
            return xmin, xmin + self.size
        else:
            xmin = self.tmin + (num - 1) * (1 + factor) * self.size
            return xmin, xmin + self.size

    def arrow(self, state, alvo, transition_type="-"):
        if transition_type == "~":
            is_upward_singlet_transfer = (
                state[0] == "S"
                and alvo[0] == "S"
                and int(alvo[1:]) > int(state[1:])
            )
            x_i = self.x(state)[0] if is_upward_singlet_transfer else self.x(state)[1]
            x_f = self.x(alvo)[0] + self.size / 4
            factor = -1 if "S" in state else 1
            if state[0] == alvo[0]:
                factor *= 0.75
        elif "T" in state and "T" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + 3 * self.size / 4, 1
        elif "T" in state and "S" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + 3 * self.size / 4, 1
        elif "S" in state and "T" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + self.size / 4, -1
        else:
            x_i, x_f, factor = self.x(state)[0], self.x(alvo)[0], 1
        return x_i, x_f, factor


def relu(x):
    return np.maximum(0.3, x)


def state_label(state):
    return f"{state[0]}$_{{{state[1:]}}}$"


def transition_label(state, target, transition_type, rate, error):
    arrow = "\\to" if transition_type == "-" else "\\leadsto"
    return (
        f"{state[0]}$_{{{state[1:]}}}{arrow}$"
        f"{target[0]}$_{{{target[1:]}}}$: "
        + format_rate(rate, error)
    )


def transition_style(weight, fontsize, color):
    scale = np.sqrt(np.clip(weight, 0.0, 1.0))
    return {
        "linewidth": 1.35 + 2.15 * scale,
        "mutation_scale": max(11.5, min(15.0, fontsize * 0.8)),
        "alpha": 0.9,
        "color": color,
    }


def add_wavy_arrow(ax, x, y_start, y_end, style, label=None):
    distance = abs(y_start - y_end)
    if distance <= 1e-8:
        return

    waves = max(2, int(np.ceil(distance / 0.85)))
    amp = 0.022 + 0.0035 * style["linewidth"]
    straight_fraction = min(0.18, max(0.08, 0.14 / distance))
    wave_fraction = 1 - straight_fraction

    t_wave = np.linspace(0, wave_fraction, 150)
    local_t = t_wave / wave_fraction
    y_wave = y_start + (y_end - y_start) * t_wave
    envelope = np.sin(np.pi * local_t)
    x_wave = x + amp * np.sin(2 * np.pi * waves * local_t) * envelope

    t_tip = np.linspace(wave_fraction, 1, 26)
    y_tip = y_start + (y_end - y_start) * t_tip
    x_tip = np.full_like(y_tip, x)

    xvals = np.concatenate([x_wave, x_tip[1:]])
    y = np.concatenate([y_wave, y_tip[1:]])
    path = Path(np.column_stack([xvals, y]))
    arrow = patches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>,head_length=0.32,head_width=0.22",
        mutation_scale=style["mutation_scale"],
        color=style["color"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=12,
        label=label,
    )
    ax.add_patch(arrow)


def plot_transitions(data, ax, cutoff):
    if data.empty:
        return

    cutoff = cutoff / 100
    fontsize = set_fontsize(ax)
    level_lw = max(1.6, fontsize / 5)
    rates = data["Rate"].to_numpy()
    error = data["Error"].to_numpy()
    transitions = data["Transition"].to_numpy()
    weights = np.nan_to_num(data["Prob"].to_numpy() / 100, nan=0.0)
    weights = np.clip(weights, 0.0, 1.0)
    energies = data["AvgDE+L"].to_numpy()
    energies[1:] += energies[0]
    base = energies[0]
    state = transitions[0].split(">")[0][:-1]
    num = int(state[1:])
    alvos = [i.split(">")[1] for i in transitions]
    trans = [i.split(">")[0][-1] for i in transitions]
    S = State()
    ##Makes S0 lines
    xmin, xmax = S.x(state)
    fill(ax, xmin, xmax, base, state_label(state))
    ax.hlines(y=base, xmin=xmin, xmax=xmax, lw=level_lw, color=S.color(state), zorder=6)
    ax.hlines(y=0, xmin=xmin, xmax=xmax, lw=level_lw, color=S.color(state), zorder=6)
    ax.text(
        x=xmin + abs(xmax - xmin) / 2,
        y=-0.4,
        s="S$_{0}$",
        ha="center",
        va="center",
        color=THECOLOR,
        fontsize=fontsize,
    )
    ##
    for i, _ in enumerate(energies):
        if weights[i] <= cutoff:
            continue

        style = transition_style(weights[i], fontsize, S.color(state))
        kw = dict(
            arrowstyle="-|>",
            color=style["color"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            zorder=10,
            mutation_scale=style["mutation_scale"],
            shrinkA=0,
            shrinkB=0,
        )
        label = transition_label(state, alvos[i], trans[i], rates[i], error[i])

        if alvos[i] == "S0":
            xmin, xmax = S.x(state)
            if trans[i] == "-":
                a3 = patches.FancyArrowPatch(
                    (xmin, base),
                    (xmin, 0),
                    **kw,
                    label=label,
                )
                ax.add_patch(a3)
            else:
                add_wavy_arrow(ax, xmax, base, 0, style, label=label)
            continue

        xmin, xmax = S.x(alvos[i])
        _ = check(ax, xmin, xmax)
        fill(ax, xmin, xmax, energies[i], state_label(alvos[i]))
        ax.hlines(
            y=energies[i],
            xmin=xmin,
            xmax=xmax,
            lw=level_lw,
            color=S.color(state),
            zorder=6,
        )
        fx, tx, curve = S.arrow(state, alvos[i], trans[i])
        a3 = patches.FancyArrowPatch(
            (fx, base),
            (tx, energies[i]),
            connectionstyle=f"arc3,rad={curve*0.35}",
            **kw,
            label=label,
        )
        ax.add_patch(a3)


def _iter_level_segments(ax):
    for elem in ax.get_children():
        try:
            paths = elem.get_paths()
        except AttributeError:
            continue
        for path in paths:
            vert = path.vertices
            if len(vert) < 2:
                continue
            yvals = vert[:, 1]
            xvals = vert[:, 0]
            if np.ptp(xvals) <= 1e-8:
                continue
            if np.nanmax(np.abs(yvals - yvals[0])) > 1e-8:
                continue
            yield float(np.nanmin(xvals)), float(np.nanmax(xvals)), float(yvals[0])


def _artist_x_bounds(ax):
    values = []
    for elem in ax.get_children():
        try:
            paths = elem.get_paths()
        except AttributeError:
            paths = []
        for path in paths:
            xvals = path.vertices[:, 0]
            values.extend([np.nanmin(xvals), np.nanmax(xvals)])

        if isinstance(elem, Line2D):
            xvals = np.asarray(elem.get_xdata(), dtype=float)
            if xvals.size:
                values.extend([np.nanmin(xvals), np.nanmax(xvals)])

    values = [value for value in values if np.isfinite(value)]
    if not values:
        return None
    return min(values), max(values)


def _spaced_positions(levels, min_gap, bottom, top):
    positions = np.array(sorted(levels), dtype=float)
    if len(positions) <= 1:
        return positions

    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + min_gap)

    overflow = positions[-1] - top
    if overflow > 0:
        positions -= overflow

    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - min_gap)

    underflow = bottom - positions[0]
    if underflow > 0:
        positions += underflow

    return positions


def _place_energy_labels(ax, levels, side, xmin, xmax, fontsize):
    if not levels:
        return

    levels = sorted(set(np.round(levels, 6)))
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1.0)
    min_gap = max(0.44, 0.030 * fontsize)
    bottom = ymin + 0.04 * span
    top = ymax - 0.04 * span
    label_positions = _spaced_positions(levels, min_gap, bottom, top)
    label_pad = 0.05 * span
    if label_positions.size:
        new_ymin = min(ymin, float(label_positions[0] - label_pad))
        new_ymax = max(ymax, float(label_positions[-1] + label_pad))
        if new_ymin < ymin or new_ymax > ymax:
            ax.set_ylim(new_ymin, new_ymax)
    width = max(xmax - xmin, 1.0)
    pad = 0.055 * width

    if side == "left":
        text_x = xmin - pad
        edge_x = xmin
        ha = "right"
        guide_text_x = text_x + 0.20 * pad
    else:
        text_x = xmax + pad
        edge_x = xmax
        ha = "left"
        guide_text_x = text_x - 0.20 * pad

    for level, label_y in zip(levels, label_positions):
        if abs(label_y - level) > 0.02:
            ax.plot(
                [edge_x, guide_text_x],
                [level, label_y],
                color="0.55",
                lw=0.6,
                alpha=0.65,
                clip_on=False,
                zorder=3,
            )
        ax.text(
            x=text_x,
            y=label_y,
            s=f"{level:.2f} eV",
            ha=ha,
            va="center",
            fontsize=fontsize,
            color=THECOLOR,
            clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
            zorder=20,
        )


def write_energies(ax):
    fontsize = set_fontsize(ax)
    level_segments = list(_iter_level_segments(ax))
    if not level_segments:
        return

    xmin = min(segment[0] for segment in level_segments)
    xmax = max(segment[1] for segment in level_segments)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        return
    artist_bounds = _artist_x_bounds(ax)
    plot_xmin, plot_xmax = artist_bounds if artist_bounds is not None else (xmin, xmax)

    midpoint = xmin + (xmax - xmin) / 2
    yleft, yright = [], []
    for x0, x1, y in level_segments:
        if np.isclose(y, 0):
            continue
        if (x0 + x1) / 2 <= midpoint:
            yleft.append(y)
        else:
            yright.append(y)

    _place_energy_labels(ax, yleft, "left", plot_xmin, plot_xmax, fontsize)
    _place_energy_labels(ax, yright, "right", plot_xmin, plot_xmax, fontsize)

    width = max(plot_xmax - plot_xmin, 1.0)
    ax.set_xlim([plot_xmin - 0.11 * width, plot_xmax + 0.11 * width])


def consolidate_ground_labels(ax):
    ground_labels = [
        text for text in ax.texts
        if text.get_text() == "S$_{0}$" and text.get_visible()
    ]
    if len(ground_labels) <= 1:
        return

    xs = [text.get_position()[0] for text in ground_labels]
    ys = [text.get_position()[1] for text in ground_labels]
    fontsize = ground_labels[0].get_fontsize()
    for text in ground_labels:
        text.set_visible(False)
    ax.text(
        x=float(np.mean(xs)),
        y=float(np.mean(ys)),
        s="S$_{0}$",
        ha="center",
        va="center",
        color=THECOLOR,
        fontsize=fontsize,
    )


def _legend_proxy(handle, label):
    alpha = handle.get_alpha()
    alpha = 1.0 if alpha is None else alpha
    linewidth = 2.0
    color = THECOLOR

    if isinstance(handle, Line2D):
        color = handle.get_color()
        linewidth = handle.get_linewidth()
    elif isinstance(handle, patches.Patch):
        linewidth = handle.get_linewidth()
        edgecolor = handle.get_edgecolor()
        facecolor = handle.get_facecolor()
        if edgecolor is not None and len(edgecolor) and edgecolor[-1] > 0:
            color = edgecolor
        elif facecolor is not None and len(facecolor):
            color = facecolor

    return Line2D(
        [0, 1],
        [0, 0],
        color=color,
        alpha=alpha,
        lw=3.0,
        linestyle="-",
        solid_capstyle="round",
    )


def collect_legend_items(axes):
    seen = set()
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_") or label in seen:
                continue
            seen.add(label)
            handles.append(_legend_proxy(handle, label))
            labels.append(label)
    return handles, labels


def clear_diagram_legends(fig, axes):
    for legend_artist in list(fig.legends):
        legend_artist.remove()
    for ax in axes:
        legend_artist = ax.get_legend()
        if legend_artist is not None:
            legend_artist.remove()


def diagram_legend_fontsize(ax, multi_panel=False):
    scale = 0.72 if multi_panel else 0.82
    upper = 11 if multi_panel else 12
    return max(8.5, min(upper, set_fontsize(ax) * scale))


def panel_legend_items(axes):
    items = []
    for ax in axes:
        handles, labels = collect_legend_items([ax])
        if handles:
            items.append((ax, handles, labels))
    return items


def add_panel_legends(fig, items):
    for ax, handles, labels in items:
        position = ax.get_position()
        center_x = position.x0 + position.width / 2
        legend_y = max(0.02, position.y0 - 0.018)
        fontsize = diagram_legend_fontsize(ax, multi_panel=True)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(center_x, legend_y),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=fontsize,
            handlelength=1.8,
            labelspacing=0.42,
            borderaxespad=0,
        )


def finalize_diagram_layout(fig, axes, legend=False):
    if not axes:
        return
    clear_diagram_legends(fig, axes)

    bottoms = [ax.get_ylim()[0] for ax in axes]
    tops = [ax.get_ylim()[1] for ax in axes]
    bottom = min(min(bottoms), -0.65)
    top = max(tops)
    span = max(top - bottom, 1.0)
    top += 0.08 * span

    for ax in axes:
        ax.set_ylim(bottom, top)
        left, right = ax.get_xlim()
        width = max(right - left, 1.0)
        ax.set_xlim(left - 0.04 * width, right + 0.04 * width)

    right_margin = 0.98
    bottom_margin = 0.03
    axis_legend_items = []
    if legend:
        if len(axes) == 1:
            handles, labels = collect_legend_items(axes)
            if handles:
                legend_fontsize = diagram_legend_fontsize(axes[0])
                fig.legend(
                    handles,
                    labels,
                    loc="center left",
                    bbox_to_anchor=(0.80, 0.5),
                    frameon=False,
                    fontsize=legend_fontsize,
                    handlelength=1.8,
                    labelspacing=0.42,
                )
                right_margin = 0.78
        else:
            axis_legend_items = panel_legend_items(axes)
            if axis_legend_items:
                max_rows = max(len(labels) for _, _, labels in axis_legend_items)
                bottom_margin = min(0.48, max(0.16, 0.070 + 0.042 * max_rows))

    try:
        fig.tight_layout(rect=(0.02, bottom_margin, right_margin, 0.98), pad=0.4)
    except ValueError:
        fig.subplots_adjust(
            left=0.06,
            right=right_margin,
            bottom=max(0.10, bottom_margin),
            top=0.94,
        )

    if axis_legend_items:
        add_panel_legends(fig, axis_legend_items)

def make_diagram(files, dielec, cutoff=0.01):
    _, ax = plt.subplots()
    ax.set_xticklabels([])
    plt.axis("off")
    for file in files:
        data, _ = nemo.analysis.rates(file.split("_")[1], dielec, data=file)
        data.rename(columns=lambda x: x.split("(")[0], inplace=True)
        plot_transitions(data, ax, cutoff)
    # medium = plt.legend(handles=[],title=f'Medium:\n$\epsilon ={dielec[0]}$\n$n={dielec[1]}$',title_fontsize=10, loc='best',frameon=False)
    # ax.add_artist(medium)
    # leg = plt.legend(loc='best',fontsize=10,title=f'$\epsilon ={dielec[0]}$ $n={dielec[1]}$',title_fontsize=10)
    # for item in leg.legendHandles:
    #    item.set_visible(False)
    consolidate_ground_labels(ax)
    write_energies(ax)
    # arquivo = nemo.tools.naming('diagram.png')
    # plt.savefig(arquivo,facecolor='white',dpi=300)#, transparent=True)
    return ax


def make_ensemble_diagram(
    molecules,
    dielec,
    initial_state=None,
    cutoff=10,
    ensemble_average=False,
    states=None,
    legend=False,
    figsize=None,
    axes=None,
):
    """
    Build Jablonski-style diagrams from Molecule objects.

    Parameters
    ----------
    molecules : Molecule or iterable of Molecule
        Objects with the NEMO Molecule API.
    dielec : tuple
        ``(epsilon, refractive_index)`` used for the rate calculation.
    initial_state : str, sequence, dict, optional
        Initial state passed to ``Molecule.rates``. If omitted, the first state
        in each Molecule is used.
    cutoff : float, default 10
        Minimum yield percentage displayed in the diagram.
    ensemble_average : bool, default False
        Passed to ``Molecule.rates``.
    states : tuple, optional
        ``(max_s, max_t)`` state limits passed to ``Molecule.rates``.
    legend : bool, default False
        Display transition-rate labels as in the dashboard.
    figsize : tuple, optional
        Size passed to ``plt.subplots`` when ``axes`` is not provided.
    axes : matplotlib Axes or sequence of Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one diagram axis per Molecule.
    """
    if hasattr(molecules, "rates"):
        molecules = [molecules]
    else:
        molecules = list(molecules)

    if not molecules:
        raise ValueError("At least one Molecule object is required.")
    for molecule in molecules:
        if not hasattr(molecule, "rates"):
            raise TypeError("make_ensemble_diagram expects Molecule objects.")

    eps, refractive_index = dielec
    if refractive_index**2 > eps:
        raise ValueError("n_r^2 must be less than or equal to epsilon.")

    molecule_titles = [getattr(molecule, "name", None) or "" for molecule in molecules]
    if len(molecules) > 1:
        molecule_titles = [
            f"{chr(97 + index)}) {title}".rstrip()
            for index, title in enumerate(molecule_titles)
        ]

    if initial_state is None:
        selected_initials = {}
    elif isinstance(initial_state, dict):
        selected_initials = initial_state
    elif isinstance(initial_state, str):
        selected_initials = {index: initial_state for index in range(len(molecules))}
    else:
        selected_initials = dict(zip(range(len(molecules)), initial_state))

    if axes is None:
        if figsize is None:
            width = max(11, 5.5 * len(molecules))
            height = 5.6 if legend and len(molecules) > 1 else 4
            figsize = (width, height)
        fig, axes = plt.subplots(1, len(molecules), figsize=figsize)
        axes = np.atleast_1d(axes).ravel().tolist()
    else:
        axes = np.atleast_1d(axes).ravel().tolist()
        if len(axes) < len(molecules):
            raise ValueError("Not enough axes were provided for the molecule groups.")
        fig = axes[0].get_figure()

    used_axes = axes[:len(molecules)]
    for ax in used_axes:
        ax.clear()
        ax.axis("off")
        ax.set_xticklabels([])

    for index, molecule in enumerate(molecules):
        ax = used_axes[index]
        fontsize = set_fontsize(ax)
        ax.set_title(molecule_titles[index], loc="left", fontsize=fontsize)

        initial = selected_initials.get(index)
        if initial is None:
            initial = selected_initials.get(getattr(molecule, "name", ""))
        if initial is None:
            molecule_states = getattr(molecule, "states", ())
            if not molecule_states:
                raise ValueError(f"Molecule at index {index} has no states.")
            initial = molecule_states[0]

        total_rates = molecule.rates(
            dielec,
            ensemble_average=ensemble_average,
            states=states,
            initial_state=initial,
        )
        if total_rates.empty:
            ax.text(
                0.5,
                0.5,
                "No transitions",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
                color=THECOLOR,
            )
            continue

        transition_sources = total_rates["Transition"].str.split(">").str[0].str[:-1]
        for state in pd.unique(transition_sources):
            plot_data = total_rates[transition_sources == state]
            plot_transitions(plot_data, ax, cutoff)

        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(bottom=-0.15)
        consolidate_ground_labels(ax)
        write_energies(ax)

    solvent_y = 0.96 if legend and len(used_axes) > 1 else 0
    used_axes[-1].text(
        1,
        solvent_y,
        f"$\\epsilon ={eps:.3f}$\n$n={refractive_index:.3f}$",
        transform=used_axes[-1].transAxes,
        fontsize=set_fontsize(used_axes[-1]),
        verticalalignment="top",
        horizontalalignment="right",
    )
    finalize_diagram_layout(fig, used_axes, legend=legend)
    return fig


def make_ensemble_diagrams(*args, **kwargs):
    return make_ensemble_diagram(*args, **kwargs)


#################################################################################################################################
##PREVENTS OVERWRITING#########################################
def naming(arquivo, folder="."):
    new_arquivo = arquivo
    if arquivo in os.listdir(folder):
        duplo = True
        vers = 2
        while duplo:
            new_arquivo = str(vers) + arquivo
            if new_arquivo in os.listdir(folder):
                vers += 1
            else:
                duplo = False
    return new_arquivo


###############################################################


def spectrum(dx, gran):
    num = int((max(dx) - min(dx)) / gran)
    if num == 0:
        bins = [dx[0],dx[0]]
        hist = [0,1]
        return hist, bins
    else:
        bins = np.linspace(min(dx), max(dx), num)
        hist, bins = np.histogram(dx, bins=bins, density=True)
        bins = bins[:-1] + (bins[1:] - bins[:-1]) / 2
        return hist, bins


def drift(data):
    t = data["Time"].to_numpy(dtype=float)
    dx = data["DeltaX"].to_numpy()
    dy = data["DeltaY"].to_numpy()
    dz = data["DeltaZ"].to_numpy()
    mux = np.mean(dx / t)
    muy = np.mean(dy / t)
    muz = np.mean(dz / t)
    return np.array([mux, muy, muz])


def get_peak(y, x, err=None, wavelength=False, n_samples=5000, seed=0):
    HC = 1239.84193  # eV*nm
    y = np.asarray(y)
    x = np.asarray(x)

    if err is None:
        max_idx = np.nanargmax(y)
        peak = x[max_idx]

        if wavelength:
            # x is in nm -> return energy first, then wavelength
            E = HC / peak
            lam = peak
            return f"{E:.2f}", f"{lam:.0f}"
        else:
            # x is in eV -> return energy first, then wavelength
            E = peak
            lam = HC / E
            return f"{E:.2f}", f"{lam:.0f}"

    # --- With uncertainties: Monte Carlo on y, pick argmax each time ---
    rng = np.random.default_rng(seed)
    err = np.asarray(err)
    yy = rng.normal(loc=y, scale=err, size=(n_samples, y.size))

    idx = np.argmax(yy, axis=1)          # index of max for each draw
    x_peaks = x[idx]                      # peak positions in the native x-units

    if wavelength:
        # Native x is wavelength (nm); convert each sample to energy (eV)
        lam_samples = x_peaks
        E_samples = HC / lam_samples

        E_mean, E_std = np.mean(E_samples), np.std(E_samples, ddof=1)
        lam_mean, lam_std = np.mean(lam_samples), np.std(lam_samples, ddof=1)

        return f"{E_mean:.2f} ± {E_std:.2f}", f"{lam_mean:.0f} ± {lam_std:.0f}"

    else:
        # Native x is energy (eV); convert each sample to wavelength (nm)
        E_samples = x_peaks
        lam_samples = HC / E_samples

        E_mean, E_std = np.mean(E_samples), np.std(E_samples, ddof=1)
        lam_mean, lam_std = np.mean(lam_samples), np.std(lam_samples, ddof=1)

        return f"{E_mean:.2f} ± {E_std:.2f}", f"{lam_mean:.0f} ± {lam_std:.0f}"

def vertical_tanh(x, a, b):
    return (a - b) / 2 * np.tanh(3 * (x - 1)) + (a + b) / 2


def network_spectrum(breakdown, ax, initial, process, wave):
    ax1, ax2 = ax
    # get x limits of ax2
    xmin, xmax = ax1.get_xlim()
    x2min, x2max = ax2.get_xlim()
    func = vertical_tanh
    x = np.linspace(-1, 1.5, 100)
    color_map = plt.get_cmap("coolwarm")
    # make list of colors from 0 to 1
    if process == "emi":
        transition = initial + "->S0"
        width = breakdown[transition.upper()].to_numpy()
        d_initial = breakdown["chi_" + initial.lower()].to_numpy()
        d_final = breakdown["eng"].to_numpy()
    else:
        transitions = [col for col in breakdown.columns if "->" in col]
        width = breakdown[transitions].to_numpy().flatten()
        width /= np.max(width)
        d_initial = (
            breakdown[[col for col in breakdown.columns if "chi_" in col]]
            .to_numpy()
            .flatten()
        )
        d_final = (
            breakdown[[col for col in breakdown.columns if "eng_" in col]]
            .to_numpy()
            .flatten()
        )
    width /= np.max(width)
    if wave:
        d_final = 1239.8 / d_final
    scale = (x2max - x2min) / (xmax - xmin)
    d_final = (d_final - xmin) * scale + x2min
    for i in range(breakdown.shape[0]):
        if width[i] > 0.01:
            y = func(x, d_initial[i], d_final[i])
            ax2.plot(y, x, lw=2, alpha=width[i], color=color_map(d_initial[i] / x2max))


# define function that equals a for x=-5 and b for x=5 using tanh
def left_tanh(x, a, b):
    return (b - a) / 2 * np.tanh(3 * x) + (a + b) / 2


def right_tanh(x, a, b):
    return (a - b) / 2 * np.tanh(3 * x) + (a + b) / 2


def plot_network(breakdown, ax, side, transition):
    scheme = {
        "left": {"color": "#4477AA", "func": left_tanh},
        "right": {"color": "#EE6677", "func": right_tanh},
    }
    color = scheme[side]["color"]
    func = scheme[side]["func"]
    initial = transition.split("~>")[0]
    final = transition.split("~>")[1]
    width = breakdown[transition.upper()].to_numpy()
    width /= np.max(width)
    d_initial = breakdown["chi_" + initial.lower()].to_numpy()
    d_final = breakdown["chi_" + final.lower()].to_numpy()
    x = np.linspace(-1, 1, 100)
    for i in range(breakdown.shape[0]):
        if width[i] > 0.01:
            y = func(x, d_initial[i], d_final[i])
            if width[i] == 1:
                ax.plot(
                    x,
                    y,
                    lw=2,
                    alpha=width[i],
                    color=color,
                    label=f"{initial[0].upper()}$_{{{initial[1:]}}}\\leadsto$ {final[0].upper()}$_{{{final[1:]}}}$",
                )
            else:
                ax.plot(x, y, lw=2, alpha=width[i], color=color)
    # hist, bins = np.histogram(width,bins=100)
    # ax22.plot((bins[1:]+bins[:-1])/2,hist/np.sum(hist),color=color)

##CALCULATES FORSTER RADIUS####################################
def radius(acceptor, donor, kappa2):
    acceptor = acceptor.to_numpy()
    xa = acceptor[:, 0]
    ya = acceptor[:, -2]
    dya = acceptor[:, -1]

    xd = donor["Energy"].to_numpy()
    yd = donor["Diffrate"].to_numpy()
    dyd = donor["Error"].to_numpy()

    # Speed of light
    c = 299792458  # m/s

    # Finds the edges of interpolation
    minA = min(xa)
    minD = min(xd)
    maxA = max(xa)
    maxD = max(xd)
    MIN = max(minA, minD)
    MAX = min(maxA, maxD)

    if MIN > MAX:
        return 0, 0
    X = np.linspace(MIN, MAX, 1000)
    f1 = interp1d(xa, ya, kind="cubic")
    f2 = interp1d(xd, yd, kind="cubic")
    f3 = interp1d(xa, dya, kind="cubic")
    f4 = interp1d(xd, dyd, kind="cubic")

    YA = f1(X)
    YD = f2(X)
    DYA = f3(X)
    DYD = f4(X)

    # Calculates the overlap
    Overlap = YA * YD / (X**4)

    # Overlap error
    OverError = Overlap * np.sqrt((DYA / YA) ** 2 + (DYD / YD) ** 2)

    # Integrates overlap
    IntOver = np.trapz(Overlap, X)

    # Integrated Overlap Error
    DeltaOver = np.sqrt(np.trapz((OverError**2), X))

    # Gets lifetime
    emi_rate, emi_error = donor.rate, donor.error
    tau = 1 / emi_rate
    delta_tau = (1/emi_rate)*(emi_error/emi_rate)
    
    # Calculates radius sixth power
    c *= 1e10
    const = (nemo.parser.HBAR_EV**3) * (9 * (c**4) * kappa2 * tau) / (8 * np.pi)
    radius6 = const * IntOver

    # Relative error in radius6
    delta = np.sqrt((DeltaOver / IntOver) ** 2 + (delta_tau / tau) ** 2)

    # Calculates radius
    forster_radius = radius6 ** (1 / 6)

    # Error in radius
    error_forster_radius = forster_radius * delta / 6
    return forster_radius, error_forster_radius

def make_matrix(df2):
    trans = df2['Transition'].to_list()
    initials = [i.split('>')[0][:-1] for i in trans]
    finals = [i for i in trans if 'S0' in i]
    initials = list(set(initials))
    labels = initials + finals
    df = df2.copy()
    #keep only Transition and Rate columns
    df = df[['Transition','Rate']]
    final = []
    initial = []
    # Iterate over the dataframe
    for i in range(0, len(df)):
        transition = df.at[i, 'Transition']
        target_state = transition.split('>')[-1]  # Get the target state
        initial_state = transition.split('>')[0][:-1]
        if target_state not in labels and transition not in labels:
            # Add the rate to the preceding row
            df.at[ifin, 'Rate'] += df.at[i, 'Rate']
            # Mark the current row for removal
            df.at[i, 'Remove'] = True
        elif target_state in labels:
            ifin = i
            final.append(target_state)
            initial.append(initial_state)
        else:
            ifin = i
            final.append(transition)    
            initial.append(initial_state)
    
    # Remove the marked rows
    df = df[df['Remove'] != True].drop(columns=['Remove'])
    df['Initial'] = initial
    df['Final'] = final

    M = np.zeros((len(labels),len(labels)))
    for ini in labels:
        for fin in labels:
            try:
                rate = df['Rate'][(df.Initial == ini) & (df.Final == fin)].to_numpy()[0]
                if 'S0' in fin and 'S0' not in ini:
                    fin2 = fin.split('>')[0][:-1]
                else:
                    fin2 = fin
                if ini == fin2:
                    M[labels.index(ini),labels.index(fin2)] += -rate
                    M[labels.index(fin),labels.index(ini)] += rate
                else:
                    #print(ini, fin2, rate)
                    M[labels.index(ini),labels.index(ini)] += -rate
                    M[labels.index(fin),labels.index(ini)] += rate          
            except IndexError:
                pass
    M = pd.DataFrame(M)
    M.columns = labels
    M.index = labels
    return M, df

def kinetics(total_rates, initial, debug=False):
    M, df = make_matrix(total_rates)
    if debug:
        #format as .2e
        M = M.applymap(lambda x: f'{x:.2e}')
        df['Rate'] = df['Rate'].apply(lambda x: f'{x:.2e}')
        display(df)
        display(M)
    # get index of initial state
    states = M.columns.to_list()
    #count elements that contain 'S0'
    num = sum('S0' not in i for i in states)
    rows = M.index.to_list()
    #take numbers from M without column and row names
    M = M.to_numpy(float)
    dpop = np.zeros((M.shape[0],1))
    dpop[states.index(initial),0] = 100 # Initial population
    pop = dpop
    time = [0] # Initial time
    deltat = 1e-1/np.max(np.abs(M)) # Time step (s)
    while  np.sum(dpop[num:,0]) < 99.0:
        dpop = np.matmul(expm(M*deltat),dpop)
        dpop = (dpop / np.sum(dpop)) * 100
        pop = np.hstack((pop,dpop))
        time.append(time[-1]+deltat)
        deltat = max(0.01*(time[-1]+deltat),deltat)
        # To check progress
        #print(f'Computing... {np.sum(dpop[2:,0]):.1f}%',end="\r", flush=True)
    time = np.array(time)
    #make dataframe with time and populations
    pop = pd.DataFrame(pop)
    pop.index = rows
    pop.columns = time
    return time, pop

###############################################################

def compile(dielec, datas, ensemble_average=False, states=None):
    warnings.warn(
        "visualization.compile is deprecated; use Molecule.rates() or "
        "nemo.nemo.compile_rates() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from nemo.nemo import compile_rates

    return compile_rates(
        dielec,
        datas,
        ensemble_average=ensemble_average,
        states=states,
    )


def trpl(time, pop):
    states = [i for i in pop.index.to_list() if '->S0' in i]
    emission = pop.loc[states].sum().to_numpy()
    #compute derivative of emission
    emission_derivative = np.diff(emission)/np.diff(time)
    y_data = max(emission)*emission_derivative/max(emission_derivative)
    x_data = time[:-1] + (time[1:] - time[:-1])/2
    return x_data, y_data


class FluorescentVialPlotter:
    def __init__(self, ax, vial_width=0.08, vial_height=0.25, spacing=0.04):
        """
        Parameters:
            ax (matplotlib.axes.Axes): Axes to draw the vials near.
            vial_width (float): Width of each vial in Axes-relative coordinates.
            vial_height (float): Height of each vial in Axes-relative coordinates.
            spacing (float): Horizontal space between vials.
        """
        self.ax = ax
        self.vial_width = vial_width
        self.vial_height = vial_height
        self.spacing = spacing
        self.vials = []
        self.vial_artists = []

    def spectrum_to_color(self, wavelengths_nm, intensities):
        if SpectralDistribution is None:
            return (127, 127, 127)

        sort_idx = np.argsort(wavelengths_nm)
        wavelengths_nm = wavelengths_nm[sort_idx]
        intensities = intensities[sort_idx]

        mask = (wavelengths_nm >= 380) & (wavelengths_nm <= 780)
        wavelengths_nm = wavelengths_nm[mask]
        intensities = intensities[mask]

        if len(wavelengths_nm) == 0:
            return (127, 127, 127)

        intensities /= np.max(intensities)
        spectrum_dict = dict(zip(wavelengths_nm, intensities))
        sd = SpectralDistribution(spectrum_dict, name="Emission Spectrum")
        XYZ = sd_to_XYZ(sd, illuminant=SDS_ILLUMINANTS["D65"], method="integration")
        RGB = XYZ_to_sRGB(XYZ)
        RGB = np.clip(RGB, 0, 1)
        return tuple((RGB * 255).astype(int))

    def add_vial(self, x_eV, y_emission, label=None):
        rgb = self.spectrum_to_color(x_eV, y_emission)
        self.vials.append((rgb, label))

    def render(self):
        # save the current data limits
        orig_xlim = self.ax.get_xlim()
        orig_ylim = self.ax.get_ylim()

        # disable autoscale just for the vignette
        self.ax.set_autoscale_on(False)

        base_x = 1.08
        base_y = 0.1
        for i, (rgb, label) in enumerate(self.vials):
            x0 = base_x + i * (self.vial_width + self.spacing)
            self._draw_vial(x0, base_y, rgb, label)

        # restore the original data limits
        self.ax.set_xlim(orig_xlim)
        self.ax.set_ylim(orig_ylim)

        #re‐enable autoscale for future data additions
        self.ax.set_autoscale_on(True)


    def _draw_vial(self, x0, y0, rgb_triplet, label=None):
        ax = self.ax
        transform = ax.transAxes
        fr_w, fr_h = self.vial_width, self.vial_height

        liq_off_x, liq_off_y = 0.025 * fr_w, 0.02 * fr_h
        liq_w, liq_h = 0.52 * fr_w, 0.84 * fr_h
        round_liq = 0.08 * min(fr_w, fr_h)

        vial_w, vial_h = 0.60 * fr_w, 1.10 * fr_h
        round_vial = 0.10 * min(fr_w, fr_h)
        cap_h = 0.20 * fr_h

        srgb = tuple(v / 255 for v in rgb_triplet)
        artists = []
        # Liquid
        liquid = patches.FancyBboxPatch(
            (x0 + liq_off_x, y0 + liq_off_y), liq_w, liq_h,
            boxstyle=f"round,pad=0.01,rounding_size={round_liq}",
            linewidth=0, facecolor=srgb, alpha=1,
            transform=transform, clip_on=False
        )
        ax.add_patch(liquid); artists.append(liquid)

        # Vial outline
        vial_outline = patches.FancyBboxPatch(
            (x0, y0), vial_w, vial_h,
            boxstyle=f"round,pad=0.02,rounding_size={round_vial}",
            linewidth=2, edgecolor='gray', facecolor='none',
            transform=transform, clip_on=False
        )
        ax.add_patch(vial_outline); artists.append(vial_outline)

        # Cap
        cap = patches.Rectangle(
            (x0, y0 + vial_h), vial_w, cap_h,
            edgecolor='black', facecolor='dimgray', linewidth=1,
            transform=transform, clip_on=False
        )
        ax.add_patch(cap); artists.append(cap)

        # Label
        label = label.split()
        first = label[:-1]
        last = label[-1]
        label = " ".join(first) + "\n" + last if len(label) > 1 else label[0]
        if label:
            txt = ax.text(
                x0 + vial_w/2, y0 - 0.2*fr_h, label,
                ha='center', va='top', fontsize=10, clip_on=False,
                transform=transform, zorder=6
            )
            artists.append(txt)

        self.vial_artists.extend(artists)
    

    def clear(self):
        """
        Clears all vial visuals and resets internal state.
        """
        for artist in self.vial_artists:
            artist.remove()
        self.vial_artists.clear()
        self.vials.clear()
