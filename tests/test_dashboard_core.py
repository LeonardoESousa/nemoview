from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from nemoview.dashboard_core import (Workspace, example_uploads, forster_radius,
    group_uploads, kinetics, spectrum_color, spectrum_display, validate_upload)


@pytest.fixture
def workspace():
    uploads, names = example_uploads()
    return Workspace(group_uploads(uploads, names))


def test_grouping_requires_explicit_names_and_unique_states():
    uploads, names = example_uploads()
    with pytest.raises(ValueError, match="every file"):
        group_uploads(uploads, [""] * len(uploads))
    with pytest.raises(ValueError, match="two S0"):
        group_uploads([uploads[0], uploads[0]], ["A", "A"])
    groups = group_uploads(uploads, names)
    assert set(groups) == {"Aurora", "Solstice"}
    assert set(groups["Aurora"].states) == {"S0", "S1", "T1"}


def test_invalid_uploads():
    with pytest.raises(ValueError, match="twocalc"):
        validate_upload("old.lx", b"ensemble,geometry,kbT,d_s1\nS1,1,.026,.1\n")
    uploads, _ = example_uploads()
    frame = pd.read_csv(BytesIO(uploads[0].content))
    frame.loc[0, "gamma_s0"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_upload("bad.lx", frame.to_csv(index=False).encode())


def test_actual_molecule_absorption_emission_and_rates(workspace):
    ws = workspace
    mol = ws.molecules["Aurora"]
    actual, breakdown = ws.calculate("Aurora", "absorption", "S0", 2.38, 1.4)
    expected = mol.absorption("S0", (2.38, 1.4))
    pd.testing.assert_frame_equal(actual, expected)
    emi_rates, emission, _ = ws.calculate("Aurora", "emission", "S1", 2.38, 1.4)
    pd.testing.assert_frame_equal(emission, mol.emission("S1", (2.38, 1.4)))
    assert np.isfinite(emission.to_numpy()).all()
    assert emission.rate > 0
    rates = ws.calculate("Aurora", "rates", "S1", 2.38, 1.4)
    assert set(mol.states) == {"S0", "S1", "T1"}
    assert rates.loc[rates.Transition.str.endswith("S0"), "Prob"].sum() == pytest.approx(100)
    assert "Geometry" in breakdown


def test_temperature_restore_and_cache(workspace):
    before = {s: e.data.kbT.copy() for s,e in workspace.molecules["Aurora"].ensembles.items()}
    a = workspace.calculate("Aurora", "rates", "S1", 2.38, 1.4, temperature=400.)
    assert workspace.calculate("Aurora", "rates", "S1", 2.38, 1.4, temperature=400.) is a
    for s, data in before.items():
        pd.testing.assert_series_equal(data, workspace.molecules["Aurora"].ensemble(s).data.kbT)
    with pytest.raises(ValueError, match="ε"):
        workspace.calculate("Aurora", "rates", "S1", 1., 1.5)


def test_wavelength_preserves_integral_and_original():
    x = np.linspace(1, 5, 10000)
    y = np.exp(-((x-3)/.2)**2)
    frame = pd.DataFrame({"Energy": x, "Diffrate": y, "Error": y*.1})
    original = frame.copy()
    nm = spectrum_display(frame, True, True)
    assert np.trapz(nm.Diffrate, nm.Energy) == pytest.approx(np.trapz(y,x), rel=1e-6)
    pd.testing.assert_frame_equal(frame, original)


def test_color_and_dark_spectrum():
    x = np.arange(380, 781, dtype=float)
    rgb = spectrum_color(x, np.exp(-((x-630)/10)**2))
    assert int(rgb[1:3],16) > int(rgb[3:5],16)
    assert spectrum_color(x, np.zeros_like(x)) is None
    assert spectrum_color([800,900], [1,1]) is None


def test_kinetics_matches_analytic_decay_and_conserves_mass():
    rates = pd.DataFrame({"Transition": ["S1->S0"], "Rate": [1e8]})
    result = kinetics(rates, "S1", -12, -6)
    assert np.allclose(result.S1, 100*np.exp(-1e8*result["Time (s)"]), atol=1e-8)
    assert np.allclose(result.S0 + result.S1, 100)
    assert np.allclose(result["PL (s⁻¹)"], result.S1/100*1e8)
    with pytest.raises(ValueError, match="T1 ensemble"):
        kinetics(pd.DataFrame({"Transition": ["S1~>T1"], "Rate": [1e8]}), "S1", -12, -6)



def test_instant_relaxation_preserves_decay_and_yields():
    from nemoview.dashboard_core import relaxation_yields, format_number
    rates = pd.DataFrame({"Transition": ["S1->S0", "S1~>S2"],
                          "Rate": [1e7, 1e8], "Prob": [100/11, 1000/11]})
    pop = kinetics(rates, "S1", -12, -5)
    assert (pop.S2 == 0).all()
    assert np.allclose(pop.S1, 100*np.exp(-1e7*pop["Time (s)"]))
    assert np.allclose(pop[["S0", "S1", "S2"]].sum(axis=1), 100)
    corrected = relaxation_yields(rates, "S1")
    assert corrected.Prob.iloc[0] == pytest.approx(100)
    assert format_number(1.234e8, 1.23e7, "") == "(1.23 ± 0.12) x 10^8"


def test_forster_zero_and_positive_overlap(workspace):
    a, _ = workspace.calculate("Aurora", "absorption", "S0", 2.38, 1.4)
    _, e, _ = workspace.calculate("Aurora", "emission", "S1", 2.38, 1.4)
    assert forster_radius(a,e,e.rate,e.error,0) == (0,0)
    r, err = forster_radius(a,e,e.rate,e.error,2/3)
    assert r > 0 and np.isfinite(err)
    r2, _ = forster_radius(a,e,e.rate,e.error,4*(2/3))
    assert r2/r == pytest.approx(4**(1/6))
