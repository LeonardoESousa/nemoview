import numpy as np
import pandas as pd
from nemoview.dashboard_core import emission_weighted_susceptibility


def test_weighted_mean_and_axis_conversion():
    spectrum = pd.DataFrame({"Energy": np.linspace(1, 4, 500)})
    breakdown = pd.DataFrame({"eng": [2., 2.], "sigma": [.2, .2],
                              "chi_s1": [.1, .7], "S1->S0": [1., 3.]})
    for wavelength in (False, True):
        result = emission_weighted_susceptibility(spectrum, breakdown, wavelength)
        assert np.allclose(result["Weighted χ (eV)"].dropna(), .55)


def test_local_emission_contributions_and_dark_tails():
    energy = np.linspace(1, 4, 500)
    spectrum = pd.DataFrame({"Energy": energy})
    breakdown = pd.DataFrame({"eng": [1.8, 3.2], "sigma": [.1, .1],
                              "chi_s1": [.1, .7], "S1->S0": [1., 3.]})
    result = emission_weighted_susceptibility(spectrum, breakdown, False)
    assert result.loc[abs(energy-1.8).argmin(), "Weighted χ (eV)"] < .11
    assert result.loc[abs(energy-3.2).argmin(), "Weighted χ (eV)"] > .69
    assert pd.isna(result.iloc[0]["Weighted χ (eV)"])
