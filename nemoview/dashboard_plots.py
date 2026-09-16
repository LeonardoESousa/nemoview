"""Interactive plots and small vector vials; no image renderer dependency."""
from html import escape
import threading

import numpy as np
import plotly.graph_objects as go

COLORS = ["#39c6bc", "#ffb85c", "#a69cff", "#f47caa", "#6bb8ff", "#afd77c"]


def original_energy_diagram(rates, cutoff=5, labels=False, output_format="png", mode="Dark"):
    """Render transition-derived levels using the publication layout."""
    from nemoview.energy_landscape import render_energy_landscape
    import gc
    result = render_energy_landscape(rates, cutoff, labels, output_format, mode)
    gc.collect()
    return result


_DIAGRAM_LOCK = threading.RLock()


def figure(xlabel="", ylabel="", height=480):
    fig = go.Figure()
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=height,
                      font=dict(family="Arial, sans-serif", size=13, color="#cbd5e1"),
                      margin=dict(l=45, r=25, t=30, b=55),
                      colorway=COLORS, hovermode="closest",
                      hoverlabel=dict(bgcolor="#142033", bordercolor="#60758d",
                                      font=dict(color="#ffffff", size=13), namelength=-1),
                      legend=dict(orientation="h", y=-.2, x=0),
                      xaxis_title=xlabel, yaxis_title=ylabel)
    fig.update_xaxes(gridcolor="#253347", zeroline=False)
    fig.update_yaxes(gridcolor="#253347", zeroline=False)
    return fig


def add_spectrum(fig, data, label, color, emission, normalize=True, uncertainty=True, decompose=False, axis="y"):
    ycol = "Diffrate" if emission else "Total"
    y = data[ycol].to_numpy()
    scale = max(float(y.max()), 1e-300) if normalize else 1.
    x = data.Energy.to_numpy()
    # Preserve complete data for exports; display at most 1800 points.
    idx = np.unique(np.r_[np.linspace(0, len(x)-1, min(len(x), 1800), dtype=int), np.argmax(y)])
    x, y = x[idx], y[idx] / scale
    error = np.nan_to_num(data.Error.to_numpy()[idx] / scale)
    if uncertainty:
        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[y+error, np.maximum(y-error, 0)[::-1]],
                                fill="toself", fillcolor=f"rgba{(*rgb, .12)}", line_width=0,
                                hoverinfo="skip", showlegend=False, legendgroup=label, yaxis=axis))
    fig.add_trace(go.Scatter(x=x, y=y, name=label, legendgroup=label, mode="lines",
                            yaxis=axis, line=dict(color=color, width=2.8, dash="solid" if emission else "dash"),
                            hovertemplate="%{x:.3f}<br>%{y:.4g}<extra>%{fullData.name}</extra>"))
    if decompose and not emission:
        for col in data.columns[1:-2]:
            fig.add_trace(go.Scatter(x=x, y=data[col].to_numpy()[idx]/scale,
                                    name=f"{label} · {col}", yaxis=axis, line=dict(width=1.3, dash="dot"),
                                    hovertemplate="%{x:.3f}<br>%{y:.4g}<extra>%{fullData.name}</extra>"))


def spectral_limits(spectra):
    """Union of each curve's significant support, plus 5% padding.

    A 0.5%-of-peak threshold excludes negligible long-wavelength tails without
    imposing a visible-range cutoff on genuinely infrared spectra.
    """
    bounds = []
    for x, y in spectra:
        x, y = np.asarray(x), np.asarray(y)
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
        if not valid.any() or y[valid].max() <= 0:
            continue
        indices = np.flatnonzero(valid & (y >= .005*y[valid].max()))
        low, high = max(0, indices[0]-1), min(len(x)-1, indices[-1]+1)
        bounds.extend([x[low], x[high]])
    if not bounds:
        return None
    lo, hi = min(bounds), max(bounds)
    pad = max((hi-lo)*.05, hi*.005)
    return [max(lo-pad, np.finfo(float).eps), hi+pad]


def vial_svg(color):
    color = color or "#566477"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="240" height="220" viewBox="0 0 240 220" role="img" aria-label="Emission color vial">
    <defs>
      <radialGradient id="halo"><stop stop-color="{color}" stop-opacity=".32"/><stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>
      <linearGradient id="liquid"><stop stop-color="{color}" stop-opacity=".65"/><stop offset=".5" stop-color="{color}"/><stop offset="1" stop-color="{color}" stop-opacity=".75"/></linearGradient>
      <linearGradient id="glass"><stop stop-color="white" stop-opacity=".3"/><stop offset=".18" stop-color="white" stop-opacity=".02"/><stop offset=".8" stop-color="white" stop-opacity=".02"/><stop offset="1" stop-color="white" stop-opacity=".22"/></linearGradient>
      <linearGradient id="cap"><stop stop-color="#344459"/><stop offset=".5" stop-color="#63768d"/><stop offset="1" stop-color="#253449"/></linearGradient>
    </defs>
    <ellipse cx="120" cy="143" rx="105" ry="72" fill="url(#halo)"/>
    <ellipse cx="120" cy="197" rx="48" ry="7" fill="#030913" opacity=".5"/>
    <path d="M85 66 L85 83 Q77 91 77 101 L77 180 Q77 194 91 194 L149 194 Q163 194 163 180 L163 101 Q163 91 155 83 L155 66 Z" fill="url(#glass)" stroke="#c5d5e8" stroke-opacity=".6"/>
    <path d="M81 119 L81 178 Q81 190 93 190 L147 190 Q159 190 159 178 L159 119 Z" fill="url(#liquid)"/>
    <ellipse cx="120" cy="119" rx="39" ry="5" fill="{color}" stroke="white" stroke-opacity=".35"/>
    <rect x="88" y="90" width="5" height="85" rx="3" fill="white" opacity=".35"/>
    <rect x="148" y="98" width="2" height="72" rx="1" fill="white" opacity=".22"/>
    <rect x="82" y="45" width="76" height="28" rx="5" fill="url(#cap)" stroke="#7c8ea5"/>
    <path d="M90 49v20 M98 49v20 M106 49v20 M114 49v20 M122 49v20 M130 49v20 M138 49v20 M146 49v20" stroke="#15253b" opacity=".6"/>
    </svg>'''


def energy_diagram(rates, cutoff=5, labels=False):
    fig = figure("", "Transition-derived energy (eV)", 510)
    positions = {}
    levels = {"S0": [0.]}
    groups = []
    for initial, table in rates.groupby(rates.Transition.str.extract(r"^([ST]\d+)")[0]):
        ground = table[table.Transition == f"{initial}->S0"]
        if ground.empty:
            continue
        base = float(ground.iloc[0]["AvgDE+L"])
        levels.setdefault(initial, []).append(base)
        for _, row in table.iterrows():
            target = row.Transition.split(">")[-1]
            end = 0 if target == "S0" else base + float(row["AvgDE+L"])
            if np.isfinite(end):
                levels.setdefault(target, []).append(end)
                groups.append((initial, target, base, end, row))
    ordered = sorted(levels, key=lambda s: (s[0] == "T", int(s[1:])))
    for i, state in enumerate(ordered):
        positions[state] = i
        color = COLORS[0] if state[0] == "S" else COLORS[1]
        for energy in sorted(set(round(e, 4) for e in levels[state])):
            if not np.isfinite(energy):
                continue
            fig.add_trace(go.Scatter(x=[i-.28, i+.28], y=[energy, energy], mode="lines",
                                    line=dict(color=color, width=4), showlegend=False,
                                    hovertemplate=f"{state}: {energy:.3f} eV<extra></extra>"))
        fig.add_annotation(x=i, y=max(levels[state]), text=state, showarrow=False, yshift=15,
                           font=dict(color=color, size=15))
    for initial, target, base, end, row in groups:
        if row.Prob < cutoff:
            continue
        x0, x1 = positions[initial], positions[target]
        color = COLORS[0] if "->" in row.Transition else COLORS[2]
        fig.add_annotation(x=x1, y=end, ax=x0, ay=base, xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowhead=3, arrowsize=1,
                           arrowwidth=1+2*np.clip(row.Prob/100, 0, 1), arrowcolor=color, opacity=.7)
        fig.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(base+end)/2], mode="markers+text" if labels else "markers",
                                marker=dict(size=8, color=color), showlegend=False,
                                text=[f"{row.Rate:.1e} s⁻¹"] if labels else None, textposition="top center",
                                hovertemplate=f"{escape(row.Transition)}<br>Rate: {row.Rate:.3e} ± {row.Error:.2e} s⁻¹<br>Yield: {row.Prob:.2f}%<extra></extra>"))
    fig.update_xaxes(showgrid=False, showticklabels=False, range=[-.6, max(len(ordered)-.4, 1)])
    return fig


def geometry_network(data, transitions, maximum=100, wavelength=False, spectral=False):
    fig = figure("Susceptibility (eV)" if spectral else "Initial → final state", "", 450)
    for j, trans in enumerate(transitions):
        initial, target = trans.replace("~>", "->").split("->")
        if spectral:
            chi = "chi_" + (initial.lower() if "eng" in data else target.lower())
            energy = "eng" if "eng" in data else "eng_" + target.lower()
            cols = (chi, energy)
        else:
            cols = ("chi_" + initial.lower(), "chi_" + target.lower())
        if not all(c in data for c in (*cols, trans)):
            continue
        top = data.nlargest(maximum, trans)
        denominator = max(float(data[trans].max()), 1e-300)
        if spectral:
            # One trace per transition keeps mapping responsive for large ensembles.
            top = top.loc[np.isfinite(top[list(cols)]).all(axis=1) & (top[trans] > 0)].copy()
            weight = top[trans].to_numpy() / denominator
            y = top[cols[1]].to_numpy()
            valid = y > 0
            top, weight, y = top.loc[valid], weight[valid], y[valid]
            if top.empty:
                continue
            if wavelength:
                y = 1239.84193 / y
            fig.add_trace(go.Scatter(x=top[cols[0]], y=y, name=trans, mode="markers",
                customdata=np.column_stack((top.Geometry, weight)),
                marker=dict(size=4+12*np.sqrt(weight), color=COLORS[j % 6], opacity=.65),
                hovertemplate="Geometry %{customdata[0]:.0f}<br>χ=%{x:.3f} eV<br>Position=%{y:.3f}<br>Relative weight=%{customdata[1]:.2%}<extra>%{fullData.name}</extra>"))
            continue
        for _, row in top.iterrows():
            a, b = float(row[cols[0]]), float(row[cols[1]])
            if spectral and wavelength:
                if b <= 0:
                    continue
                b = 1239.84193 / b
            weight = float(row[trans]) / denominator
            if weight < .01:
                continue
            if spectral:
                fig.add_trace(go.Scatter(x=[a], y=[b], mode="markers", showlegend=False,
                                        marker=dict(size=4+12*np.sqrt(weight), color=COLORS[j % 6], opacity=.55),
                                        hovertemplate=f"Geometry {int(row.Geometry)} · {escape(trans)}<br>χ=%{{x:.3f}} eV<br>Position=%{{y:.3f}}<extra></extra>"))
            else:
                t = np.linspace(0, 1, 20)
                fig.add_trace(go.Scatter(x=t, y=a+(b-a)*(3*t*t-2*t*t*t), mode="lines", showlegend=False,
                                        line=dict(color=COLORS[j % 6], width=.5+2*weight), opacity=.15+.65*weight,
                                        hovertemplate=f"Geometry {int(row.Geometry)} · {escape(trans)}<br>Relative weight: {weight:.2%}<extra></extra>"))
    if spectral:
        fig.update_yaxes(title="Wavelength (nm)" if wavelength else "Transition energy (eV)")
    else:
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Initial state", "Final state"], showgrid=False)
        fig.update_yaxes(title="Susceptibility (eV)")
    return fig
