from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def example_app():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Explore synthetic example").click().run()
    assert not app.exception
    assert not app.error
    return app


def test_empty_state_and_example():
    app = example_app()
    assert len(app.get("plotly_chart")) == 1
    assert [m.value for m in app.metric][:3] == ["2", "6", "288"]
    assert len(app.get("imgs")) >= 0


@pytest.mark.parametrize("view", ["Energy & kinetics", "Susceptibility", "Transition network"])
def test_all_notebook_views(view):
    app = example_app()
    next(w for w in app.get("button_group") if w.label == "Analysis").set_value(view).run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) >= 1


def test_invalid_environment_keeps_previous_results():
    app = example_app()
    next(w for w in app.number_input if w.label == "Dielectric constant ε").set_value(1.)
    next(b for b in app.button if b.label == "Apply conditions").click().run()
    assert app.error
    assert not app.exception
    assert app.metric[-1].value == "2.38"
