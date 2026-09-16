"""Publication renderer: transition-derived energies, separate spin columns."""
from io import BytesIO
import threading
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib import rc_context
from nemoview.dashboard_core import _format_rate_components, relaxation_targets

_LOCK = threading.RLock()


def render_energy_landscape(rates, cutoff=5, labels=False, output_format="png", mode="Dark"):
    if mode not in ("Light", "Dark"):
        raise ValueError("Diagram mode must be Light or Dark.")
    if output_format not in ("png", "pdf"):
        raise ValueError("Diagram export must be PNG or PDF.")
    dark = mode == "Dark"
    background, ink, muted = ("#101b2b", "#e7eef8", "#9fadc1") if dark else ("#ffffff", "#172b42", "#66788c")
    colors = {"S": "#79b5ff" if dark else "#24558b", "T": "#ecd563" if dark else "#b99816"}
    levels, edges = {}, []
    initials = rates.Transition.str.extract(r"^([ST]\d+)")[0]
    relaxations = relaxation_targets(rates)
    for state, table in rates.groupby(initials, sort=True):
        ground = table.loc[table.Transition == state + "->S0"]
        if ground.empty:
            continue
        base = float(ground.iloc[0]["AvgDE+L"])
        if not np.isfinite(base):
            continue
        levels.setdefault(state, []).append(base)
        for _, row in table.iterrows():
            # Prob contains corrected integrated yields for ground-state
            # channels; recycling must not hide low-branching final decay.
            if not np.isfinite(row.Rate) or row.Rate <= 0 or row.Prob < cutoff:
                continue
            target = row.Transition.split(">")[-1]
            end = 0. if target == "S0" else base + float(row["AvgDE+L"])
            if not np.isfinite(end):
                continue
            if target != "S0":
                levels.setdefault(target, []).append(end)
            edges.append((state, target, base, end, row))
    if not levels:
        raise ValueError("No finite transition-derived energy levels to display.")
    spins = {spin: sorted((s for s in levels if s[0] == spin), key=lambda s: int(s[1:]),
                          reverse=spin == "S") for spin in "ST"}
    # Equal-width lanes. Higher states expand their spin column instead of colliding.
    centers = {}
    cursor = 0.
    for spin in "ST":
        for state in spins[spin]:
            centers[state] = cursor
            cursor += 1.35
        cursor += .7
    half = .43
    all_y = [0.] + [v for values in levels.values() for v in values]
    low, high = min(all_y), max(all_y)
    span = max(high-low, .5)
    # A fixed 89 mm canvas preserves readable type at single-column size.
    # Compress the physical height, not the energy scale: all y positions
    # remain linear in eV, including the shared S0 baseline.
    width = 3.5
    legend_rows = len(edges) if labels else 0
    legend_height = .17 * legend_rows + (.08 if legend_rows else 0.)
    plot_height = max(1.5, .19 * max(sum(len(set(round(v, 6) for v in levels[state])) for state in spins[spin]) for spin in "ST") + .65)
    height = plot_height + legend_height
    with _LOCK, rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                            "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig = Figure(figsize=(width, height), facecolor=background)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([.16, (legend_height+.10)/height, .68, (plot_height-.20)/height])
        ax.set_facecolor(background)
        ax.axis("off")
        ax.set_xlim(-1., max(centers.values())+1.)
        ax.set_ylim(low-.20*span, high+.30*span)
        ax.text(-.17, .5, "Energy (eV)", transform=ax.transAxes, rotation=90,
                color=ink, fontsize=8, ha="center", va="center")
        for state, center in centers.items():
            values = sorted(set(round(v, 6) for v in levels[state]))
            color = colors[state[0]]
            if len(values) > 1:
                ax.fill_between([center-half, center+half], values[0], values[-1], color=color, alpha=.10, linewidth=0)
            for energy in values:
                ax.plot([center-half, center+half], [energy]*2, color=color, lw=2.5, solid_capstyle="butt", zorder=4)
            ax.text(center, values[0]-.065*span, f"{state[0]}$_{{{state[1:]}}}$",
                    ha="center", va="top", color=ink, fontsize=10)
        # All singlet energies go in the outer left margin, triplets at right.
        # Spacing is coordinated across each entire spin manifold, not per bar.
        for spin in "ST":
            entries = sorted((energy, centers[state]) for state in spins[spin]
                             for energy in sorted(set(round(v, 6) for v in levels[state])))
            if not entries:
                continue
            values = [e for e, _ in entries]
            positions = np.array(values, dtype=float)
            for i in range(1, len(positions)):
                positions[i] = max(positions[i], positions[i-1]+.16*span)
            positions -= (positions.mean()-np.mean(values))
            left = spin == "S"
            side = -1 if left else 1
            outer = min(centers.values())-half if left else max(centers.values())+half
            xtext = outer+side*.18
            for (energy, center), ytext in zip(entries, positions):
                xline = center+side*half
                ax.plot([xline, outer+side*.045, xtext-side*.025], [energy, energy, ytext],
                        color=muted, lw=.6, alpha=.60, zorder=1)
                ax.text(xtext, ytext, f"{energy:.2f}", ha="right" if left else "left",
                        va="center", color=ink, fontsize=8.5)
        # A single shared ground-state baseline anchors the composition.
        ax.plot([min(centers.values())-half, max(centers.values())+half], [0, 0], color=muted, lw=1.4)
        ax.text(np.mean(list(centers.values())), -.085*span, "S$_0$", ha="center", va="top", color=ink, fontsize=10)
        handles, legend_text = [], []
        for index, (state, target, base, end, row) in enumerate(edges):
            color = colors[state[0]]
            radiative = "->" in row.Transition
            if target == "S0":
                x = centers[state] + (-.34 if radiative else .34)
                start, finish, curvature = (x, base), (x, 0), 0
            else:
                # Cross-spin endpoints depend on originating spin: S right→left,
                # T left→right. Mirrored state order keeps both paths separated.
                direction = (1 if state[0] == "S" else -1) if state[0] != target[0] else (
                    1 if centers[target] > centers[state] else -1)
                start = (centers[state]+direction*half, base)
                finish = (centers[target]-direction*half, end)
                # Reversing endpoints already reverses the visual bend.
                # Keep the same signed radius so ISC goes above, rISC below.
                curvature = -.25
            arrow = FancyArrowPatch(start, finish, connectionstyle=f"arc3,rad={curvature}",
                arrowstyle="-|>", mutation_scale=10, linewidth=1.6,
                linestyle="solid" if radiative or target != "S0" else (0, (3, 2)),
                color=color, shrinkA=2, shrinkB=2, zorder=5, clip_on=False)
            ax.add_patch(arrow)
            if labels:
                handles.append(Line2D([0], [0], color=color, lw=1.6))
                mantissa, error, exponent = _format_rate_components(float(row.Rate), float(row.Error))
                rate_label = rf"$({mantissa}\pm{error})\times10^{{{exponent}}}\;\mathrm{{s}}^{{-1}}$"
                legend_text.append(row.Transition.replace("~>", " ⇝ ").replace("->", " → ")
                                   + ": " + rate_label)
        if labels and handles:
            ax.legend(handles, legend_text, loc="lower center", bbox_to_anchor=(.5, .02/height), bbox_transform=fig.transFigure,
                       frameon=False, ncol=1, fontsize=8, labelcolor=ink,
                       handlelength=1.8, borderaxespad=0., labelspacing=.30)
        # Show the assumed pathways explicitly without inventing finite rates.
        for source, target in relaxations.items():
            if source not in centers or target not in centers:
                continue
            direction = 1 if centers[target] > centers[source] else -1
            start = (centers[source]+direction*half, min(levels[source]))
            end = (centers[target]-direction*half, min(levels[target]))
            ax.add_patch(FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.15",
                arrowstyle="-|>", mutation_scale=9, linewidth=1.4, linestyle=":",
                color=colors[source[0]], shrinkA=2, shrinkB=2, clip_on=False, zorder=5))
        output = BytesIO()
        fig.savefig(output, format=output_format, dpi=600, facecolor=background,
                    bbox_inches=None)
        fig.clear()
        # Release the high-resolution Agg buffer before another theme/export.
        fig.set_canvas(None)
        return output.getvalue()
