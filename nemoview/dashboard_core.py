"""Validated, session-local adapter for NEMO's twocalc Molecule API.

All spectra are retained in energy units; display conversions never mutate NEMO
results. No notebook, matplotlib, or Streamlit imports are needed here.
"""
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO, StringIO
import csv
import hashlib
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.linalg import expm

HC = 1239.84193
MAX_BYTES = 40 * 1024 * 1024
MAX_ROWS = 20000
MAX_STATES = 30


@dataclass
class Upload:
    filename: str
    content: bytes
    digest: str
    state: str
    rows: int


def validate_upload(filename, content):
    if not content or len(content) > MAX_BYTES:
        raise ValueError("Each ensemble must be nonempty and at most 40 MB.")
    try:
        text = content.decode("utf-8-sig")
        header = next(csv.reader(StringIO(text)))
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not supported.")
        data = pd.read_csv(StringIO(text), nrows=MAX_ROWS + 1)
    except (UnicodeError, pd.errors.ParserError, StopIteration) as exc:
        raise ValueError("Use a UTF-8, comma-separated NEMO .lx ensemble.") from exc
    if data.empty or len(data) > MAX_ROWS:
        raise ValueError(f"Use 1–{MAX_ROWS:,} geometries per file.")
    required = {"ensemble", "geometry", "kbT", "gamma_s0"}
    missing = required - set(data)
    energies = [c for c in data if re.fullmatch(r"e_[st]\d+", c)]
    for c in energies:
        missing.update({"chi_" + c[2:], "gamma_" + c[2:]} - set(data))
    if missing or not energies:
        raise ValueError("This file is not a twocalc ensemble. Generate it with NEMO twocalc; "
                         "legacy d_s/d_t files cannot be converted by renaming columns. Missing: "
                         + ", ".join(sorted(missing or {"e_s1"})) + ".")
    states = data.ensemble.astype(str).str.upper().unique()
    if len(states) != 1 or not re.fullmatch(r"S\d+|T[1-9]\d*", states[0]):
        raise ValueError("Each file must contain one ensemble state (S0, S1, T1, …).")
    if data.ensemble.iloc[0] != states[0]:
        raise ValueError("Use uppercase ensemble state labels, such as S1.")
    for spin in "st":
        numbers = sorted(int(c[3:]) for c in energies if c.startswith("e_" + spin))
        if numbers and numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("State columns must be contiguous, starting at state 1.")
        if len(numbers) > MAX_STATES:
            raise ValueError(f"At most {MAX_STATES} states per spin are supported.")
    numeric = data.drop(columns="ensemble").apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Numeric columns contain missing, nonnumeric, or infinite values.")
    if (data.kbT <= 0).any() or data.kbT.nunique() != 1:
        raise ValueError("kbT must be positive and constant within each ensemble.")
    if (data.filter(regex=r"^chi_") < 0).any(axis=None):
        raise ValueError("Susceptibilities must be nonnegative.")
    if data.geometry.duplicated().any() or (data.geometry % 1 != 0).any():
        raise ValueError("Geometry identifiers must be unique integers.")
    state = states[0]
    if state != "S0" and "e_" + state.lower() not in data:
        raise ValueError(f"The file has no energy column for {state}.")
    return Upload(filename, content, hashlib.sha256(content).hexdigest(), state, len(data))


def group_uploads(uploads, names):
    from nemo.nemo import Molecule

    groups = OrderedDict()
    seen = set()
    for upload, name in zip(uploads, names):
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError("Give every file a molecule name (1–80 characters).")
        key = (name, upload.state)
        if key in seen:
            raise ValueError(f"{name} has two {upload.state} ensembles. Use different molecule names or remove one file.")
        seen.add(key)
        groups.setdefault(name, []).append(upload)
    if len(names) != len(uploads):
        raise ValueError("Every uploaded file needs a molecule name.")
    molecules = {}
    # NEMO readers can require filesystem paths (not BytesIO), especially on
    # Windows. Constructors load eagerly, so temporary files can be removed.
    with TemporaryDirectory(prefix="nemoview-") as directory:
        for index, (name, files) in enumerate(groups.items()):
            paths = []
            for number, upload in enumerate(files):
                path = Path(directory) / f"{index}-{number}.lx"
                path.write_bytes(upload.content)
                paths.append(str(path))
            try:
                molecules[name] = Molecule(*paths, name=name)
            except Exception as exc:
                raise ValueError(
                    "NEMO could not read these ensembles. This dashboard requires "
                    "the CSV-based twocalc revision pinned in pyproject.toml. "
                    "Reinstall the patched project with: python -m pip install --upgrade -e . "
                    f"Reader details: {exc}"
                ) from exc
    return molecules


class Workspace:
    """Per-session, bounded result cache; mutable Molecules are never global."""
    def __init__(self, molecules):
        self.molecules = molecules
        self.cache = OrderedDict()

    def calculate(self, name, operation, state, eps, nr, temperature=None,
                  average=False, nstates=-1, limits=None):
        if eps < 1 or nr < 1 or nr * nr > eps + 1e-10:
            raise ValueError("Use ε ≥ n², with ε and n at least 1.")
        key = (name, operation, state, eps, nr, temperature, average, nstates, limits)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        mol = self.molecules[name]
        originals = {s: ens.data.kbT.copy() for s, ens in mol.ensembles.items()}
        try:
            if temperature is not None:
                for ens in mol.ensembles.values():
                    ens.data["kbT"] = float(temperature) * 8.617333262e-5
            if operation == "emission":
                result = mol.complete_emi(state, (eps, nr), ensemble_average=average)
            elif operation == "absorption":
                result = mol.complete_abs(state, (eps, nr), nstates=nstates)
            elif operation == "rates":
                # S0 has absorption, but no outgoing emission/rates. Exclude it
                # while retaining the actual Molecule.rates corrected-yield API.
                ground = mol.ensembles.pop("S0", None)
                try:
                    result = mol.rates((eps, nr), ensemble_average=average,
                                       states=limits, initial_state=state)
                finally:
                    if ground is not None:
                        mol.ensembles["S0"] = ground
            else:
                result = mol.breakdown(state, (eps, nr))
        finally:
            # Restore by state because removing S0 can change insertion order.
            for s, values in originals.items():
                mol.ensembles[s].data["kbT"] = values
        self.cache[key] = result
        while len(self.cache) > 12:
            self.cache.popitem(last=False)
        return result

def spectrum_display(spectrum, emission, wavelength):
    out = spectrum.copy()
    out = out.loc[out.Energy > 0].copy()
    if wavelength:
        energy = out.Energy.to_numpy().copy()
        out["Energy"] = HC / energy
        if emission:
            # dR/dλ = dR/dE * |dE/dλ|. The twocalc helper currently
            # uses its reciprocal, so explicitly use the density Jacobian.
            out["Diffrate"] *= energy**2 / HC
            out["Error"] *= energy**2 / HC
    return out.sort_values("Energy").reset_index(drop=True)


def spectrum_color(wavelength, intensity):
    """CIE 1931 self-luminous color; normalize luminance before sRGB mapping."""
    import colour

    x, y = np.asarray(wavelength, float), np.asarray(intensity, float)
    valid = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x, y = x[valid], y[valid]
    if len(x) < 2:
        return None
    order = np.argsort(x)
    grid = np.arange(380.0, 781.0, 5.0)
    values = np.interp(grid, x[order], y[order], left=0, right=0)
    if values.max() <= 0:
        return None
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
        colour.SpectralShape(380, 780, 5))
    xyz = np.trapz(values[:, None] * cmfs.values, grid, axis=0)
    if xyz[1] <= 0:
        return None
    rgb = colour.XYZ_to_sRGB(xyz / xyz[1])
    rgb = np.maximum(rgb, 0)
    rgb /= max(float(rgb.max()), 1e-12)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def emission_weighted_susceptibility(spectrum, breakdown, wavelength=True):
    """Mean chi at each spectral coordinate, weighted by broadened emission.

    Every geometry contributes its NEMO radiative strength times its normalized
    Gaussian line shape. The wavelength Jacobian cancels in the weighted mean.
    Chunking limits temporary allocations; no geometry sampling is used.
    """
    transitions = [c for c in breakdown if c.endswith("->S0")]
    if len(transitions) != 1:
        raise ValueError("Select an emission spectrum with one radiative transition to S0.")
    transition = transitions[0]
    chi_col = "chi_" + transition.split("->")[0].lower()
    required = ["eng", "sigma", chi_col, transition]
    if not all(c in breakdown for c in required):
        raise ValueError("Emission breakdown lacks energies, widths, or susceptibilities.")
    values = breakdown[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 1] <= 0).any() or (values[:, 3] < 0).any():
        raise ValueError("Emission weights must be finite and nonnegative, with positive widths.")
    energy = spectrum.Energy.to_numpy(dtype=float)
    energy = energy[np.isfinite(energy) & (energy > 0)]
    numerator, denominator = np.zeros(len(energy)), np.zeros(len(energy))
    for start in range(0, len(values), 128):
        center, width, chi, rate = values[start:start+128].T
        weight = (rate / width)[:, None] * np.exp(-.5*((energy[None, :]-center[:, None])/width[:, None])**2)
        denominator += weight.sum(axis=0)
        numerator += (weight * chi[:, None]).sum(axis=0)
    density = denominator * (energy**2 / HC if wavelength else 1.)
    mean = np.divide(numerator, denominator, out=np.full(len(energy), np.nan), where=denominator > 0)
    # Do not report means in negligible tails where there is effectively no light.
    meaningful = (density > 0) & (density >= .001*density.max()) if len(density) else np.array([], dtype=bool)
    mean[~meaningful] = np.nan
    return pd.DataFrame({"Wavelength (nm)" if wavelength else "Energy (eV)": HC/energy if wavelength else energy,
                         "Weighted χ (eV)": mean}).sort_values(
                             "Wavelength (nm)" if wavelength else "Energy (eV)").reset_index(drop=True)


def _format_rate_components(rate, error_rate):
    """Uncertainty-aware rounding supplied for the dashboard rate table."""
    if not np.isfinite(rate) or rate <= 1e-99:
        return "0", "0", 0
    exp = int(np.floor(np.log10(rate)))
    r = rate / 10 ** exp
    if not np.isfinite(error_rate) or error_rate <= 0:
        decimals = max(0, 1 - int(np.floor(np.log10(r))))
        return f"{r:.{decimals}f}", "0", exp
    e = error_rate / 10 ** exp
    order = int(np.floor(np.log10(e)))
    digits = 2 if e / 10 ** order < 3 or e / r > .5 else 1
    decimals = max(0, -order + digits - 1)
    if decimals == 0:
        decimals = 1
    return f"{r:.{decimals}f}", f"{e:.{decimals}f}", exp


def format_number(rate, error_rate, unit="s^-1"):
    rate, error, exponent = _format_rate_components(rate, error_rate)
    if exponent:
        return f"({rate} ± {error}) x 10^{exponent} {unit}".strip()
    return f"{rate} ± {error} {unit}".strip()


def relaxation_targets(rates):
    """Unmodelled higher states relax to the lowest state of the same spin."""
    parsed = rates.Transition.str.extract(r"^([ST]\d+)(->|~>)([ST]\d+)$")
    outgoing = set(parsed.loc[rates.Rate.to_numpy() > 0, 0])
    missing = (set(parsed[2]) - outgoing) - {"S0"}
    targets = {}
    for state in sorted(missing):
        target = state[0] + "1"
        if state == target or target not in outgoing:
            raise ValueError(f"Cannot model {state} relaxation: upload a {target} ensemble with outgoing rates.")
        targets[state] = target
    return targets


def relaxation_yields(rates, initial):
    """Infinite-time channel yields of the same kinetic network.

    Solve integrated state occupancy (equivalently an absorbing Markov chain),
    independent of the plotted time window. Internal rows retain local branching
    percentages; rows ending at S0 contain integrated final channel yields.
    """
    result = rates.copy().reset_index(drop=True)
    if result.empty:
        return result
    targets = relaxation_targets(rates)
    parsed = result.Transition.str.extract(r"^([ST]\d+)(->|~>)([ST]\d+)$")
    if parsed.isna().any(axis=None):
        raise ValueError("Unrecognized transition label.")
    values = result.Rate.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Yields require finite, nonnegative rates.")
    source = parsed[0].to_numpy()
    destination = parsed[2].map(lambda s: targets.get(s, s)).to_numpy()
    totals = result.Rate.groupby(parsed[0]).transform("sum").to_numpy()
    branching = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    result["Prob"] = 100 * branching
    initial = targets.get(initial, initial)
    reachable = {initial}
    while True:
        new = {d for s, d, k in zip(source, destination, values) if s in reachable and k > 0 and d != "S0"}
        if new <= reachable:
            break
        reachable |= new
    states = sorted(reachable)
    index = {s: i for i, s in enumerate(states)}
    q = np.zeros((len(states), len(states)))
    for s, d, p in zip(source, destination, branching):
        if s in index and d in index:
            q[index[s], index[d]] += p
    p0 = np.zeros(len(states))
    p0[index[initial]] = 1
    try:
        visits = np.linalg.solve(np.eye(len(states))-q.T, p0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The reachable kinetic network has no complete decay path to S0.") from exc
    terminal = parsed[2].eq("S0").to_numpy()
    final = np.array([visits[index[s]]*p if s in index else 0. for s,p in zip(source,branching)])
    if not np.isclose(final[terminal].sum(), 1., rtol=1e-6, atol=1e-9):
        raise ValueError("Integrated decay yields do not sum to 100%; check missing decay channels.")
    result.loc[terminal, "Prob"] = 100*np.maximum(final[terminal], 0)
    return result


def kinetics(rates, initial, log_start=-12, log_end=1, points=260):
    """Integrate populations and individual S0 fluxes to 99.9999% completion.

    A small absorbing-state matrix handles every terminal channel separately.
    Stochastic scaling/squaring keeps very stiff recycling networks stable;
    cost is bounded by the sample count and at most 1024 matrix squarings.
    The requested time window changes sampling, never final channel yields.
    """
    corrected = relaxation_yields(rates, initial)
    parsed = rates.Transition.str.extract(r"^([ST]\d+)(->|~>)([ST]\d+)$")
    states = sorted(set(parsed[0]) | set(parsed[2]))
    if initial not in states:
        raise ValueError("Initial state is absent from the rate network.")
    targets = relaxation_targets(rates)
    terminal = parsed[2].eq("S0").to_numpy()
    terminal_rows = np.flatnonzero(terminal)
    excited = [s for s in states if s != "S0"]
    index = {s: i for i, s in enumerate(excited)}
    size = len(excited) + len(terminal_rows)
    matrix = np.zeros((size, size))
    radiative = np.zeros(len(excited))
    sinks = {row: len(excited)+i for i, row in enumerate(terminal_rows)}
    for row, (source, kind, dest) in enumerate(parsed.itertuples(index=False, name=None)):
        rate = float(rates.Rate.iloc[row])
        i = index[source]
        j = sinks[row] if terminal[row] else index[targets.get(dest, dest)]
        matrix[i, i] -= rate
        matrix[j, i] += rate
        if kind == "->" and dest == "S0":
            radiative[i] += rate
    scale = float(np.max(-np.diag(matrix)))
    if scale <= 0:
        raise ValueError("The kinetic network has no decay to S0.")
    scaled = matrix / scale
    p0 = np.zeros(size)
    p0[index[targets.get(initial, initial)]] = 1.

    def population(time):
        # exp(scaled * step) has a moderate norm, even for extreme stiffness.
        log_step = np.log2(time) + np.log2(scale)
        squarings = max(0, int(np.ceil(log_step)))
        if squarings > 1024:
            raise ValueError("Kinetic timescales exceed numerical limits.")
        propagator = expm(scaled * 2.**(log_step-squarings))
        propagator = np.maximum(propagator, 0.)
        propagator /= propagator.sum(axis=0)
        for _ in range(squarings):
            propagator = propagator @ propagator
            # Only remove floating-point column-sum drift, preserving the
            # relative excited populations and each cumulative channel flux.
            propagator /= propagator.sum(axis=0)
        return propagator @ p0

    # Search by decades, starting at the fastest timescale. No ODE stepping
    # across trillions of fast recycling events, nor an arbitrary 1 s endpoint.
    horizon = 1. / scale
    for _ in range(310):
        final = population(horizon)
        if final[:len(excited)].sum() <= 1e-6:
            break
        horizon *= 10.
        if not np.isfinite(horizon):
            raise ValueError("Kinetics cannot reach S0 within numerical timescales.")
    else:
        raise ValueError("Kinetics failed to reach 99.9999% S0 within its work limit.")
    exact = corrected.loc[terminal, "Prob"].to_numpy()/100
    if not np.allclose(final[len(excited):], exact, rtol=1e-5, atol=2e-6):
        raise ValueError("Integrated kinetic fluxes disagree with the final-yield solution.")
    end = np.log10(horizon)
    start = min(float(log_start), end-8)
    t = np.r_[0., np.logspace(start, end, min(600, max(40, int(points))))]
    populations = np.array([p0] + [population(time) for time in t[1:]])
    frame = pd.DataFrame(0., index=np.arange(len(t)), columns=states)
    frame[excited] = populations[:, :len(excited)] * 100
    frame["S0"] = populations[:, len(excited):].sum(axis=1) * 100
    frame.insert(0, "Time (s)", t)
    frame["PL (s⁻¹)"] = populations[:, :len(excited)] @ radiative
    frame.attrs["terminal_yields_percent"] = final[len(excited):] * 100
    frame.attrs["terminal_rows"] = terminal_rows
    frame.attrs["ground_percent"] = float(frame.S0.iloc[-1])
    frame.attrs["convergence_time"] = float(horizon)
    return frame


def forster_radius(absorption, emission, rate, rate_error, kappa2):
    """Notebook overlap expression (Å); zero-safe first-order uncertainty."""
    a = absorption.sort_values("Energy")
    d = emission.sort_values("Energy")
    low = max(a.Energy.min(), d.Energy.min(), 1e-9)
    high = min(a.Energy.max(), d.Energy.max())
    if high <= low or rate <= 0 or kappa2 == 0:
        return 0., 0.
    x = np.linspace(low, high, 1000)
    ya = np.interp(x, a.Energy, a.Total)
    yd = np.interp(x, d.Energy, d.Diffrate)
    ea = np.interp(x, a.Energy, a.Error)
    ed = np.interp(x, d.Energy, d.Error)
    overlap = np.trapz(ya * yd / x**4, x)
    if overlap <= 0:
        return 0., 0.
    uncertainty = np.sqrt(np.trapz(((ea * yd)**2 + (ed * ya)**2) / x**8, x))
    const = 6.582119569e-16**3 * 9 * (299792458e10)**4 * kappa2 / (8 * np.pi * rate)
    radius = (const * overlap)**(1 / 6)
    error = radius / 6 * np.hypot(uncertainty / overlap, rate_error / rate)
    return float(radius), float(error)


def example_uploads():
    """Small deterministic, synthetic twocalc inputs, never experimental data."""
    rng = np.random.default_rng(28)
    uploads, names = [], []
    for name, shift in [("Aurora", 0.0), ("Solstice", -0.42)]:
        for initial in ["S0", "S1", "T1"]:
            n = 48
            data = {"ensemble": [initial] * n, "geometry": np.arange(1, n + 1),
                    "kbT": np.full(n, 0.02585), "gamma_s0": np.full(n, 0.04)}
            for spin, base in [("s", 2.75), ("t", 2.48)]:
                for j in range(1, 4):
                    state = f"{spin}{j}"
                    data["e_" + state] = rng.normal(base + shift + (j - 1) * 0.55, .07, n)
                    data["chi_" + state] = rng.uniform(.08, .18, n)
                    data["gamma_" + state] = rng.uniform(.02, .06, n)
                    if initial.lower().startswith(spin):
                        if initial != "S0":
                            data["osce_" + state] = np.full(n, .18 if spin == "s" else 1e-5)
                        if j > int(initial[1:]):
                            data["osc_" + state] = rng.uniform(.1, .3, n)
            if initial != "S0":
                spin, other = ("s", "t") if initial[0] == "S" else ("t", "s")
                for i in range(1, 4):
                    if spin == "t":
                        data[f"soc_t{i}_s0"] = np.full(n, 2e-5)
                    for j in range(1, 4):
                        data[f"soc_{spin}{i}_{other}{j}"] = rng.uniform(.0001, .0003, n)
            content = pd.DataFrame(data).to_csv(index=False).encode()
            uploads.append(validate_upload(f"{name}_{initial}.lx", content))
            names.append(name)
    return uploads, names
