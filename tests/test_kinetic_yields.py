import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from matplotlib.figure import Figure
from nemoview.dashboard_core import kinetics, relaxation_yields
from nemoview.energy_landscape import render_energy_landscape


def recycling():
    return pd.DataFrame({'Transition':['S1->S0','S1~>T2','T1->S0','T1~>S1'],
        'Rate':[4.53e5,1.067e7,69.,6.7e7], 'Error':[1.]*4,
        'AvgDE+L':[1.41,.53,1.41,.26]})


def test_recycling_recovers_low_branching_final_decay():
    rates = recycling()
    exact = relaxation_yields(rates, 'S1')
    assert rates.Rate.iloc[0]/rates.Rate.iloc[:2].sum() < .05
    assert exact.Prob.iloc[0] > 99.9
    pop = kinetics(rates, 'S1', -12, -10)
    assert pop.S0.iloc[-1] >= 99.9999
    assert pop['Time (s)'].iloc[-1] > 1e-10
    assert np.allclose(pop[['S0','S1','T1','T2']].sum(axis=1),100,atol=1e-8)
    np.testing.assert_allclose(pop.attrs['terminal_yields_percent'],exact.Prob.iloc[[0,2]],atol=2e-4)
    def inspect(fig, *args, **kwargs):
        text = [t.get_text() for t in fig.axes[0].get_legend().texts]
        assert any('S1 → S0' in t for t in text)
        assert not any('T1 → S0' in t for t in text)
        assert not any('TRANSITION-DERIVED' in t.get_text() for t in fig.axes[0].texts)
    with patch.object(Figure,'savefig',inspect):
        render_energy_landscape(exact,cutoff=5,labels=True)


def test_long_lived_tail_and_separate_parallel_terminal_channels():
    rates = pd.DataFrame({'Transition':['S1->S0','S1~>T2','T1->S0','T1~>S0'],
                          'Rate':[1e10,1e10,3e-7,1e-7]})
    pop = kinetics(rates,'S1',-14,-6)
    assert pop.S0.iloc[-1] >= 99.9999
    assert len(pop) <= 601
    assert pop['Time (s)'].iloc[-1] > 1e7
    np.testing.assert_allclose(pop.attrs['terminal_yields_percent'],[50,37.5,12.5],atol=2e-4)
    assert (pop.T2==0).all()


def test_nonabsorbing_network_is_reported():
    rates = pd.DataFrame({'Transition':['S1~>T1','T1~>S1'], 'Rate':[1e8,1e8]})
    with pytest.raises(ValueError,match='decay path'):
        kinetics(rates,'S1')


def test_initial_state_changes_yields():
    rates = pd.DataFrame({'Transition':['S1->S0','T1->S0'],'Rate':[1e8,1e-3]})
    for state, expected in [('S1',[100,0]),('T1',[0,100])]:
        result = kinetics(rates,state)
        np.testing.assert_allclose(result.attrs['terminal_yields_percent'],expected,atol=2e-4)
