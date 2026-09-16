"""Streamlit molecular photophysics workbench."""
from io import BytesIO
from pathlib import Path
from nemo.__version__ import __version__ as nemo_version
from nemoview.__version__ import __version__ as nemoview_version
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nemoview.dashboard_core import (HC, Workspace, example_uploads, forster_radius,
    group_uploads, kinetics, spectrum_color, spectrum_display, validate_upload,
    format_number, relaxation_yields, relaxation_targets, emission_weighted_susceptibility)
from nemoview.dashboard_plots import (COLORS, add_spectrum, energy_diagram, figure,
    geometry_network, vial_svg, original_energy_diagram, spectral_limits)


def chart(fig, key):
    dark = st.context.theme.type == "dark"
    ink, grid, background = ("#e6edf6", "#28374b", "#142033") if dark else ("#172b42", "#dce3eb", "#ffffff")
    fig.update_layout(font_color=ink, legend_font_color=ink,
        hoverlabel=dict(bgcolor=background, bordercolor=grid, font=dict(color=ink), namelength=-1))
    fig.update_xaxes(gridcolor=grid, tickfont_color=ink, title_font_color=ink)
    fig.update_yaxes(gridcolor=grid, tickfont_color=ink, title_font_color=ink)
    for trace in fig.data:
        if "hoverlabel" in trace._valid_props:
            trace.hoverlabel = dict(bgcolor=background, bordercolor=grid, font=dict(color=ink), namelength=-1)
        if "hovertemplate" in trace._valid_props and trace.hovertemplate:
            # Plotly's secondary name box ignores the main hover contrast.
            template = trace.hovertemplate
            if "<extra>%{fullData.name}</extra>" in template:
                trace.hovertemplate = "<b>%{fullData.name}</b><br>" + template.replace("<extra>%{fullData.name}</extra>", "<extra></extra>")
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 3,
            "filename": "nemoview-" + key}})


def download_table(frame, label, key):
    st.download_button(label, frame.to_csv(index=False).encode(), key + ".csv", "text/csv",
        key="download-" + key, icon=":material/download:")


def upload_panel():
    with st.expander("Ensemble library", expanded="workspace" not in st.session_state,
                     icon=":material/folder_open:"):
        st.caption("Upload .lx ensembles, then give every file a molecule name. Use the same name for all states of one molecule.")
        files = st.file_uploader("Ensemble files", type=["lx", "csv"], accept_multiple_files=True,
                                 max_upload_size=40, key="uploads")
        if files:
            if len(files) > 24 or sum(f.size for f in files) > 100 * 1024**2:
                st.error("Use up to 24 files and 100 MB total per workspace.")
                return
            signature = tuple((f.name, f.file_id, f.size) for f in files)
            if st.session_state.get("upload_signature") != signature:
                parsed, errors = [], []
                for f in files:
                    try:
                        parsed.append(validate_upload(f.name, f.getvalue()))
                    except ValueError as exc:
                        errors.append(f"{f.name}: {exc}")
                st.session_state.upload_signature = signature
                st.session_state.parsed_uploads = parsed
                st.session_state.upload_errors = errors
            for error in st.session_state.upload_errors:
                st.error(error)
            parsed = st.session_state.parsed_uploads
            if parsed and not st.session_state.upload_errors:
                with st.form("molecule_names"):
                    names = []
                    for i, u in enumerate(parsed):
                        a, b = st.columns([1.2, 1])
                        a.write(f"**{u.filename}**")
                        a.caption(f"{u.state} ensemble · {u.rows:,} geometries")
                        names.append(b.text_input("Molecule name", placeholder="e.g. Coumarin 153",
                            key=f"name-{i}-{u.digest[:12]}", max_chars=80))
                    submitted = st.form_submit_button("Apply molecule names", type="primary", icon=":material/check:")
                if submitted:
                    try:
                        st.session_state.workspace = Workspace(group_uploads(parsed, names))
                        st.session_state.manifest = [{"file": u.filename, "molecule": n.strip(),
                            "state": u.state, "sha256": u.digest} for u, n in zip(parsed, names)]
                        st.session_state.is_example = False
                        st.session_state.active_signature = signature
                        st.rerun()
                    except (ValueError, ImportError) as exc:
                        st.error(str(exc))
            if "workspace" in st.session_state and signature != st.session_state.get("active_signature"):
                st.caption("The current analysis uses the applied library. Apply names to replace it with these uploads.")
        a, b = st.columns(2)
        if a.button("Explore synthetic example", icon=":material/science:"):
            uploads, names = example_uploads()
            st.session_state.workspace = Workspace(group_uploads(uploads, names))
            st.session_state.is_example = True
            st.session_state.manifest = [{"file": u.filename, "molecule": n, "state": u.state,
                "sha256": u.digest} for u, n in zip(uploads, names)]
            st.rerun()
        if "workspace" in st.session_state and b.button("Clear analysis", icon=":material/delete:"):
            for key in ["workspace", "manifest", "is_example", "active_signature"]:
                st.session_state.pop(key, None)
            st.rerun()


def solvent_panel():
    defaults = st.session_state.get("conditions", (2.38, 1.4, None, False))
    with st.sidebar:
        st.subheader("Environment")
        st.caption("Plots update when you change a control.")
        eps = st.number_input("Dielectric constant ε", min_value=1., max_value=200., value=defaults[0], step=.1, key="live-eps")
        nr = st.number_input("Refractive index n", min_value=1., max_value=5., value=defaults[1], step=.01, key="live-nr")
        override = st.checkbox("Override ensemble temperature", value=defaults[2] is not None, key="live-override")
        temperature = st.number_input("Temperature (K)", min_value=1., max_value=1500.,
            value=float(defaults[2] or 300.), step=5., disabled=not override, key="live-temperature")
        average = False
        if nr**2 > eps + 1e-10:
            st.error("Use ε ≥ n². Showing the last valid conditions until corrected.")
        else:
            defaults = (eps, nr, temperature if override else None, average)
            st.session_state.conditions = defaults
        st.caption(f"Applied: ε = {defaults[0]:.2f} · n = {defaults[1]:.3f}")
        st.caption("Temperature: " + (f"{defaults[2]:.0f} K" if defaults[2] else "from ensemble files"))
    return (*defaults[:3], False)


def calculate(ws, conditions, name, operation, state, **kw):
    eps, nr, temp, average = conditions
    return ws.calculate(name, operation, state, eps, nr, temp, average, **kw)


def contribution_table(data, transitions, count=10):
    for trans in transitions:
        total = float(data[trans].sum())
        if total <= 0:
            continue
        table = data[["Geometry", trans]].copy()
        table["Contribution (%)"] = table[trans] / total * 100
        st.caption(trans)
        st.dataframe(table.nlargest(count, "Contribution (%)").drop(columns=trans), hide_index=True)


def spectra_view(ws, conditions):
    options, default = {}, []
    for name, mol in ws.molecules.items():
        for state in sorted(mol.states):
            if any(c.startswith("osc_") for c in mol.ensemble(state).data):
                label = f"{name} · {state} · Absorption"
                options[label] = (name, state, False)
                if state == "S0":
                    default.append(label)
            if state != "S0":
                label = f"{name} · {state} · Emission"
                options[label] = (name, state, True)
                if state == "S1":
                    default.append(label)
    selection = st.multiselect("Compare spectra", list(options), default=default[:6], max_selections=12)
    a, b, c, d = st.columns(4)
    wavelength = a.segmented_control("Horizontal axis", ["nm", "eV"], default="nm") == "nm"
    normalize = b.toggle("Normalize spectra", value=True)
    uncertainty = c.toggle("Uncertainty bands", value=True)
    vials = d.toggle("Emission vials", value=True)
    with st.expander("Spectral controls", icon=":material/tune:"):
        a, b, c = st.columns(3)
        nstates = a.number_input("Absorption states (−1 = all)", min_value=-1, max_value=30, value=-1)
        decompose = b.checkbox("Decompose absorption")
        mapping = b.checkbox("Map susceptibility to spectra", key="spectral-mapping")
        extinction = not normalize
        kappa = c.slider("Förster orientation factor κ²", 0., 4., 2/3, .01)
        st.caption("Solid: emission · dashed: absorption. Bands show NEMO ensemble error. Hover, zoom, or click legend entries to inspect.")
    if nstates == 0:
        st.info("Select −1 for all absorption states, or a positive state count.")
        return
    if not selection:
        st.info("Choose at least one spectrum to begin.")
        return
    fig = figure("Wavelength (nm)" if wavelength else "Energy (eV)", "Normalized intensity")
    if not normalize:
        fig.update_layout(
            yaxis=dict(title=dict(text="Molar extinction (M⁻¹ cm⁻¹)", standoff=18),
                       rangemode="tozero", automargin=True, visible=True, anchor="x"),
            yaxis2=dict(title=dict(text="Differential emission rate", standoff=18),
                        overlaying="y", side="right", showgrid=False, rangemode="tozero", automargin=True),
            margin=dict(l=100, r=100))
        if not any(not options[label][2] for label in selection):
            # Plotly otherwise omits a primary axis when all traces use y2.
            fig.add_trace(go.Scatter(x=[1], y=[0], yaxis="y", opacity=0, showlegend=False, hoverinfo="skip"))
            fig.update_layout(yaxis_showticklabels=False, yaxis_range=[0, 1])
        if not any(options[label][2] for label in selection):
            fig.add_trace(go.Scatter(x=[1], y=[0], yaxis="y2", opacity=0, showlegend=False, hoverinfo="skip"))
            fig.update_layout(yaxis2_showticklabels=False, yaxis2_range=[0, 1])
    support = []
    results, stats, exports, vial_data = [], [], {}, []
    for i, label in enumerate(selection):
        name, state, emission = options[label]
        try:
            value = calculate(ws, conditions, name, "emission" if emission else "absorption", state, nstates=nstates)
            if emission:
                rates, spectrum, breakdown = value
                rate, error = float(spectrum.rate), float(spectrum.error)
            else:
                spectrum, breakdown = value
                rate, error = None, None
            if spectrum.empty or not np.isfinite(spectrum.to_numpy()).all():
                raise ValueError("Spectrum has no finite values at these conditions.")
            view = spectrum_display(spectrum, emission, wavelength)
            if extinction and not emission:
                view.iloc[:, 1:] *= 1e-16 * 6.02214076e23 / (1000 * np.log(10))
            add_spectrum(fig, view, label, COLORS[i % 6], emission, normalize, uncertainty, decompose,
                         axis="y2" if emission and not normalize else "y")
            support.append((view.Energy.to_numpy(), view["Diffrate" if emission else "Total"].to_numpy()))
            total = "Diffrate" if emission else "Total"
            p = float(view.loc[view[total].idxmax(), "Energy"])
            weighted_chi = None
            if emission and f"chi_{state.lower()}" in breakdown:
                weights = breakdown[f"{state}->S0"].to_numpy()
                if weights.sum() > 0:
                    weighted_chi = float(np.average(breakdown[f"chi_{state.lower()}"], weights=weights))
            else:
                terms = [(c, f"{state}->{c[4:].upper()}") for c in breakdown if c.startswith("chi_")]
                terms = [(c, t) for c, t in terms if t in breakdown]
                weight = sum(float(breakdown[t].sum()) for c, t in terms)
                if weight > 0:
                    weighted_chi = sum(float((breakdown[c]*breakdown[t]).sum()) for c, t in terms)/weight
            stats.append({"Spectrum": label, "Peak (nm)": p if wavelength else HC/p,
                "Peak (eV)": HC/p if wavelength else p, "Weighted χ (eV)": weighted_chi,
                "Radiative lifetime (s)": 1/rate if rate and rate > 0 else None,
                "Lifetime error (s)": error/rate**2 if rate and rate > 0 else None})
            results.append((label, emission, spectrum, rate, error, breakdown))
            exports[f"spectrum-{i+1}.csv"] = view.rename(columns={"Energy": "Wavelength (nm)" if wavelength else "Energy (eV)"}).to_csv(index=False)
            exports[f"geometry-{i+1}.csv"] = breakdown.to_csv(index=False)
            if emission and vials:
                nm = spectrum_display(spectrum, True, True)
                vial_data.append((label, spectrum_color(nm.Energy, nm.Diffrate)))
        except Exception as exc:
            st.error(f"{label}: {exc}")
    limits = spectral_limits(support)
    if limits is not None:
        fig.update_xaxes(range=limits)
    fig.update_yaxes(rangemode="tozero")
    if vial_data:
        spectrum_panel, vial_panel = st.columns([3, 1], gap="medium")
    else:
        spectrum_panel, vial_panel = st.container(), None
    with spectrum_panel:
        with st.container(border=True):
            st.subheader("Steady-state Spectra")
            chart(fig, "spectra")
            st.caption("Automatic range includes signals above 0.5% of each peak. Zoom out to inspect tails; downloads retain all data.")
    if vial_data:
        with vial_panel:
            with st.container(border=True):
                st.markdown("**Visible emission**")
                for label, color in vial_data:
                    st.image(vial_svg(color), width=150)
                    st.caption(label)
                    st.caption(color or "No visible emission")
                st.caption("Relative emission hue · CIE 1931 · 380–780 nm. Brightness is normalized; this is not solution color under room light.")
    if stats:
        st.subheader("Peak positions & lifetimes")
        st.caption("Peaks refer to the displayed density; converting an emission density from eV to nm can shift its maximum.")
        peak_table = pd.DataFrame(stats)
        peak_display = peak_table.drop(columns=["Radiative lifetime (s)", "Lifetime error (s)"]).copy()
        peak_display["Radiative lifetime ± error (s)"] = [
            format_number(row["Radiative lifetime (s)"], row["Lifetime error (s)"], "")
            if pd.notna(row["Radiative lifetime (s)"]) else "—" for row in stats]
        st.dataframe(peak_display, hide_index=True, width="stretch", column_config={
            "Peak (nm)": st.column_config.NumberColumn(format="%.1f"),
            "Peak (eV)": st.column_config.NumberColumn(format="%.3f"),
            "Weighted χ (eV)": st.column_config.NumberColumn(format="%.3f")})
    if mapping and results:
        st.subheader("Susceptibility → spectral position")
        emissions = {r[0]: r for r in results if r[1]}
        if not emissions:
            st.info("Select an emission spectrum above to calculate emission-weighted susceptibility.")
        else:
            selected = st.multiselect("Emission curves to compare", list(emissions),
                                      default=list(emissions), key="map-emission-curves")
            map_figure = figure("Wavelength (nm)" if wavelength else "Energy (eV)", "Emission-weighted χ (eV)")
            mapping_tables = []
            for label in selected:
                result = emissions[label]
                curve = emission_weighted_susceptibility(result[2], result[5], wavelength)
                xcol = curve.columns[0]
                color = COLORS[selection.index(label) % len(COLORS)]
                map_figure.add_trace(go.Scatter(x=curve[xcol], y=curve["Weighted χ (eV)"],
                    name=label, mode="lines", connectgaps=False, line=dict(color=color, width=2.5),
                    hovertemplate="%{x:.3f}<br>Weighted χ: %{y:.3f} eV<extra>%{fullData.name}</extra>"))
                curve.insert(0, "Spectrum", label)
                mapping_tables.append(curve)
            if mapping_tables:
                valid_x = np.concatenate([c.loc[c["Weighted χ (eV)"].notna()].iloc[:, 1].to_numpy() for c in mapping_tables])
                if len(valid_x):
                    pad = max(float(np.ptp(valid_x))*.05, .01)
                    map_figure.update_xaxes(range=[max(0, valid_x.min()-pad), valid_x.max()+pad])
                chart(map_figure, "spectral-map")
                st.caption("At each wavelength: Σ χᵢ Iᵢ / Σ Iᵢ, using every geometry’s radiative strength and Gaussian emission profile. Curves are omitted below 0.1% of their peak emission.")
                table = pd.concat(mapping_tables, ignore_index=True).dropna(subset=["Weighted χ (eV)"])
                st.dataframe(table, hide_index=True, height=220, column_config={
                    "Wavelength (nm)": st.column_config.NumberColumn(format="%.1f"),
                    "Energy (eV)": st.column_config.NumberColumn(format="%.3f"),
                    "Weighted χ (eV)": st.column_config.NumberColumn(format="%.3f")})
                download_table(table, "Download weighted susceptibility curves", "weighted-susceptibility")
                exports["weighted-susceptibility.csv"] = table.to_csv(index=False)
    donors, acceptors = [r for r in results if r[1]], [r for r in results if not r[1]]
    if donors and acceptors:
        radii = []
        for donor in donors:
            for acceptor in acceptors:
                radius, error = forster_radius(acceptor[2], donor[2], donor[3], donor[4], kappa)
                radii.append({"Donor": donor[0], "Acceptor": acceptor[0], "R₀ (Å)": radius, "Error (Å)": error})
        st.subheader("Förster overlap")
        st.caption("Notebook model using radiative lifetime (unit donor quantum yield). Energy-domain integration; first-order uncertainty estimate.")
        st.dataframe(pd.DataFrame(radii), hide_index=True)
        exports["forster.csv"] = pd.DataFrame(radii).to_csv(index=False)
    if exports:
        exports["peaks.csv"] = pd.DataFrame(stats).to_csv(index=False)
        exports["spectra.html"] = fig.to_html(include_plotlyjs=True)
        exports["settings.json"] = json.dumps({"environment": conditions, "selections": selection,
            "wavelength": wavelength, "extinction": extinction, "normalized_plot": normalize,
            "nstates": nstates, "kappa2": kappa, "files": st.session_state.manifest}, indent=2)
        output = BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name, data in exports.items():
                bundle.writestr(name, data)
        st.download_button("Download spectra & report", output.getvalue(), "nemoview-spectra.zip",
            "application/zip", icon=":material/download:")


def levels_view(ws, conditions):
    eligible = [name for name, mol in ws.molecules.items() if any(s != "S0" for s in mol.states)]
    if not eligible:
        st.info("Upload an excited-state ensemble to calculate energy levels and kinetics.")
        return
    names = st.multiselect("Compare molecules", eligible, default=eligible[:2], max_selections=4)
    diagram_controls, kinetics_controls = st.columns([1.1, 1], gap="medium")
    with diagram_controls:
        st.markdown("**Photophysics controls**")
        c1, c2 = st.columns(2)
        cutoff = c1.slider("Yield cutoff (%)", 0, 100, 5,step=5)
        diagram_mode = c2.segmented_control("Diagram appearance", ["Dark", "Light"], default="Dark", key="diagram-appearance") or "Dark"
        c1, c2, c3 = st.columns(3)
        labels = c1.toggle("Show rate labels")
        average = c2.toggle("Ensemble-average statistics", help="Changes diagram and rate-table averages only.", value=True)
        limits_on = c3.toggle("Custom state limits", help="NEMO excludes the highest two states when more than two are available by default.")
        limits = None
        if limits_on:
            c1, c2 = st.columns(2)
            limits = (c1.number_input("Maximum singlet", 1, 30, 3), c2.number_input("Maximum triplet", 0, 30, 3))
    with kinetics_controls:
        st.markdown("**Time-resolved controls**")
        c1, c2 = st.columns(2)
        custom_time = c1.toggle("Custom time window", help="By default, show the complete decay to S₀.")
        time_range = st.slider("Time window · log₁₀(seconds)", -14, 5, (-12, 0)) if custom_time else (-12, 0)
        show_ground = c2.toggle("Show ground-state population")
    conditions = (*conditions[:3], average)
    for name in names:
        states = sorted(s for s in ws.molecules[name].states if s != "S0")
        _, initial_column = st.columns([1.1, 1])
        initial = initial_column.selectbox("Initially populated state", states, key="initial-" + name)
        try:
            rates = calculate(ws, conditions, name, "rates", initial, limits=limits)
            rates = relaxation_yields(rates, initial)
            if rates.empty:
                st.info("No transitions remain within the selected state limits.")
                continue
            pop = kinetics(rates, initial, *time_range)
            rates.loc[pop.attrs["terminal_rows"], "Prob"] = pop.attrs["terminal_yields_percent"]
            a, b = st.columns([1.1, 1])
            with a:
                st.subheader(name)
                with st.container(border=True):
                    st.subheader("Predicted photophysics")
                    diagram = original_energy_diagram(rates, cutoff, labels, mode=diagram_mode)
                    st.image(diagram, width="stretch")
                    png_button, pdf_button = st.columns(2)
                    png_button.download_button("Download PNG", diagram, "energy-diagram.png",
                                       "image/png", key="diagram-" + name, width="stretch")
                    pdf_button.download_button("Download PDF", original_energy_diagram(rates, cutoff, labels, "pdf", mode=diagram_mode),
                                       "energy-diagram.pdf", "application/pdf", key="diagram-pdf-" + name, width="stretch")
                    st.caption("Transition-derived energy landscape. Shading indicates non-radiative relaxation.")
                    targets = relaxation_targets(rates)
                    if targets:
                        st.caption("Assumed instantaneous relaxation: " + ", ".join(f"{s} → {t}" for s, t in targets.items())
                                   + ". Dotted arrows show these assumed IC pathways; the same model is used for kinetics and yields.")
            with b:
                with st.container(border=True):
                    st.subheader("Time-resolved spectra")
                    fig = figure("Time (s)", "Population (%)", 510)
                    for state in pop.columns[1:-1]:
                        values = pop[state].iloc[1:].to_numpy()
                        # Zero curves (e.g. instantaneously relaxed T2) and curves
                        # below the displayed log-axis floor have no visible trace.
                        visible = np.isfinite(values) & (values >= 1e-3)
                        if (state != "S0" or show_ground) and visible.any():
                            fig.add_trace(go.Scatter(x=pop.iloc[1:, 0], y=pop[state].iloc[1:], name=state, mode="lines"))
                    pl = pop["PL (s⁻¹)"].to_numpy()[1:]
                    if pl.max() > 0:
                        fig.add_trace(go.Scatter(x=pop.iloc[1:, 0], y=pl/pl.max(), name="PL (normalized)",
                            yaxis="y2", line=dict(dash="dot", color=COLORS[3])))
                    fig.update_layout(yaxis2=dict(title="Relative PL", overlaying="y", side="right", showgrid=False))
                    fig.update_xaxes(type="log", range=list(time_range) if custom_time else None)
                    fig.update_yaxes(type="log", range=[-3, 2.05])
                    chart(fig, "kinetics-" + name)
                    c1, c2 = st.columns(2)
                    c1.caption("State populations and normalized time-resolved photoluminescence."+f"{pop.attrs['ground_percent']:.3f}% S₀ at "
                               f"{pop.attrs['convergence_time']:.2e} s.")
                    key = "kinetics-" + name
                    c2.download_button("Download kinetics & PL", pop.to_csv(index=False).encode(), key + ".csv", "text/csv",
                            key="download-" + key, icon=":material/download:")
                    
            table = rates.rename(columns={"Rate": "Rate (s⁻¹)", "Error": "Error (s⁻¹)", "Prob": "Yield (%)",
                "AvgDE+L": "Mean gap (eV)", "AvgCoupling": "Coupling (meV)",
                "AvgSigma": "Width (eV)", "AvgConc": "Participation (%)"})
            display_table = table.drop(columns=["Rate (s⁻¹)", "Error (s⁻¹)"]).copy()
            display_table.insert(1, "Rate ± error (s⁻¹)",
                                 [format_number(r, e, "") for r, e in zip(rates.Rate, rates.Error)])
            st.dataframe(display_table, hide_index=True, width="stretch", column_config={
                **{c: st.column_config.NumberColumn(format="%.2f") for c in ["Yield (%)", "Participation (%)"]},
                **{c: st.column_config.NumberColumn(format="%.3f") for c in ["Mean gap (eV)", "Coupling (meV)", "Width (eV)"]}})
            st.caption("S0 channels: cumulative kinetic yields, checked against the infinite-time solution, including recycling. Other channels: local branching percentages. A displayed 0.00% may be nonzero below the rounding precision.")
            a, _ = st.columns([1.1, 1])
            with a:
                download_table(table, "Download rates", "rates-" + name)
        except Exception as exc:
            st.error(f"{name}: {exc}")


def susceptibility_view(ws, conditions):
    options = {}
    for name, mol in ws.molecules.items():
        for state in sorted(mol.states):
            for col in mol.ensemble(state).data.filter(regex="^chi_"):
                options[f"{name} · {col[4:].upper()} @ {state}"] = (name, state, col)
    selected = st.multiselect("State susceptibilities", list(options), default=list(options)[:3], max_selections=12)
    a, b = st.columns(2)
    width = 10. ** a.slider("Histogram bin width · log₁₀(eV)", -3, -1, -2)
    maximum = b.number_input("Color scale / ranking threshold (eV)", min_value=.01, max_value=10., value=1., step=.1)
    fig = figure("Solvent susceptibility (eV)", "Probability density (eV⁻¹)")
    quantiles, labels, tables = [], [], []
    for label in selected:
        name, state, col = options[label]
        data = ws.molecules[name].ensemble(state).data
        values = data[col].to_numpy()
        low, high = float(values.min()), float(values.max())
        bins = np.linspace(low-width/2, high+width/2, min(1501, max(2, int((high-low)/width)+2)))
        density, edges = np.histogram(values, bins=bins, density=True)
        fig.add_trace(go.Scatter(x=(edges[1:]+edges[:-1])/2, y=density, name=label, mode="lines"))
        quantiles.append(np.quantile(values, np.linspace(0, 1, 200)))
        labels.append(label)
        top = data.loc[data[col] < maximum, ["geometry", col]].nlargest(5, col).copy()
        top.columns = ["Geometry", "χ (eV)"]
        top.insert(0, "Ensemble / state", label)
        tables.append(top)
    a, b = st.columns([1.3, 1])
    with a:
        chart(fig, "susceptibility-distribution")
    with b:
        heat = figure("", "Ensemble percentile")
        if quantiles:
            heat.add_trace(go.Heatmap(z=np.array(quantiles).T, x=labels, y=np.linspace(0,100,200),
                colorscale="Tealrose", zmin=0, zmax=maximum, colorbar=dict(title="χ (eV)")))
        chart(heat, "susceptibility-composition")
    st.caption("Deterministic ensemble quantiles; no random resampling. Stored susceptibilities do not change with the display solvent.")
    if tables:
        st.subheader("Geometries below the threshold")
        table = pd.concat(tables, ignore_index=True)
        st.dataframe(table, hide_index=True)
        download_table(table, "Download ranked geometries", "susceptibility-ranking")


def network_view(ws, conditions):
    name = st.selectbox("Molecule", list(ws.molecules))
    states = [s for s in sorted(ws.molecules[name].states) if s != "S0"]
    if not states:
        st.info("Upload an excited-state ensemble to inspect transitions.")
        return
    state = st.selectbox("Ensemble state", states)
    try:
        data = calculate(ws, conditions, name, "breakdown", state)
        options = [c for c in data if "~>" in c and not c.endswith("S0")]
        transitions = st.multiselect("Transitions", options, default=options[:1])
        maximum = st.slider("Maximum geometries per transition", 10, 150, 60, 10)
        chart(geometry_network(data, transitions, maximum), "transition-network")
        st.caption("Line opacity and width show relative contribution. Only the strongest contributions are drawn; downloads include the full ensemble.")
        contribution_table(data, transitions)
        download_table(data, "Download geometry breakdown", "network-" + name + "-" + state)
    except Exception as exc:
        st.error(str(exc))


def main():
    st.set_page_config(page_title="NEMOview · Molecular photophysics", page_icon=":material/science:", layout="wide")
    with st.sidebar:
        st.image(str(Path(__file__).parent / "figs" / "nemoview.png"), width="stretch")
        st.caption(f"NEMO {nemo_version} · NEMOview {nemoview_version}")
        st.caption("Dashboard appearance: open ⋮ → Settings → Theme to choose Light or Dark. Diagram appearance can be set separately.")
    conditions = solvent_panel()
    st.caption("ENSEMBLE WORKBENCH")
    upload_panel()
    if "workspace" not in st.session_state:
        st.info("Start with your ensemble files, or explore the synthetic example above.")
        a, b, c = st.columns(3)
        for column, title, text in [(a, "01 · Name your molecules", "Group S0, S1, and T1 ensembles with a shared molecule name."),
            (b, "02 · Set the environment", "Adjust dielectric response, refractive index, and temperature."),
            (c, "03 · Follow the light", "Compare interactive spectra, pathways, and visible emission hues.")]:
            with column:
                with st.container(border=True):
                    st.subheader(title)
                    st.write(text)
        return
    ws = st.session_state.workspace
    if st.session_state.get("is_example"):
        st.badge("Synthetic example · illustrative data", color="orange", icon=":material/science:")
    st.subheader("Analysis")
    view = st.segmented_control("Analysis", ["Spectra", "Energy & kinetics", "Susceptibility", "Transition network"],
        default="Spectra", width="stretch", label_visibility="collapsed")
    routes = {"Spectra": spectra_view, "Energy & kinetics": levels_view,
        "Susceptibility": susceptibility_view, "Transition network": network_view}
    if view in routes:
        routes[view](ws, conditions)
    with st.sidebar:
        st.caption("NEMO twocalc · Molecule API")
        with st.expander("Analysis notes"):
            st.write("Only the selected analysis runs. Recent results are reused within your session. Uploads stay in memory.")
            st.write("Chart camera buttons export high-resolution PNGs. Spectral reports include an offline interactive plot and complete data.")


if __name__ == "__main__":
    main()
