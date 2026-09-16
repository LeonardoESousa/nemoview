import numpy as np
import pandas as pd
from unittest.mock import patch
from matplotlib.figure import Figure
from nemoview.dashboard_core import kinetics, relaxation_targets, relaxation_yields
from nemoview.energy_landscape import render_energy_landscape


def network():
    return pd.DataFrame({"Transition": ["S1->S0", "S1~>T2", "T1->S0"],
        "Rate": [75., 25., 10.], "Error": [1., 1., 1.], "Prob": [100.,25.,0.],
        "AvgDE+L": [3., -.2, 2.5]})


def test_triplet_relaxation_and_phosphorescence_yield():
    rates = network()
    assert relaxation_targets(rates) == {"T2": "T1"}
    corrected = relaxation_yields(rates, "S1")
    assert np.isclose(corrected.loc[corrected.Transition == "T1->S0", "Prob"].iloc[0], 25)
    assert np.isclose(corrected.loc[corrected.Transition == "S1->S0", "Prob"].iloc[0], 75)
    pop = kinetics(corrected, "S1", -5, 1)
    assert (pop.T2 == 0).all()
    assert pop.T1.max() > 0
    assert np.allclose(pop[["S0", "S1", "T1", "T2"]].sum(axis=1),100)
    assert np.isclose(pop.S0.iloc[-1],100)


def test_diagram_cutoff_uses_corrected_yields():
    # The triplet channel has a corrected 25% yield after T2 relaxation.
    original = Figure.savefig
    def inspect(fig, *args, **kwargs):
        legend = fig.axes[0].get_legend()
        assert any("T1 → S0" in t.get_text() for t in legend.texts)
        assert any(p.get_linestyle() == ":" for p in fig.axes[0].patches)
        return original(fig, *args, **kwargs)
    with patch.object(Figure, "savefig", inspect):
        assert render_energy_landscape(relaxation_yields(network(), "S1"),cutoff=5,labels=True).startswith(b"\x89PNG")
