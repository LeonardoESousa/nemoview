import requests
import pkg_resources
import ipywidgets as widgets


def eps_nr(kw,eps0=1, nr0=1):
    eps = widgets.BoundedFloatText(
        value=eps0,
        min=1,
        max=100.0,
        step=0.1,
        description="$\\epsilon$",
        tooltip="Dielectric constant",
        disabled=False,
    )

    nr = widgets.BoundedFloatText(
        value=nr0,
        step=0.1,
        min=1,
        max=10.0,
        description="$n_r$",
        tooltip="Refractive index",
        disabled=False,
    )
    kw['eps'] = eps
    kw['nr'] = nr
    return eps, nr, kw

def kappa(kw):
    kappa2 = widgets.BoundedFloatText(
        value=0.66,
        min=0,
        max=4,
        step=0.1,
        tooltip='Orientation factor for Förster radius calculation',
        description='$\\kappa^2$',
        disabled=False
    )
    kw['kappa'] = kappa2
    return kappa2, kw

def maxsusc(kw):
    susc = widgets.BoundedFloatText(
        value=1.0,
        min=0.1,
        max=5.0,
        step=0.1,
        tooltip='Maximum susceptibility for network plot',
        description='Max Susc.:',
        disabled=False
    )
    kw['maxsusc'] = susc
    return susc, kw

def wave(kw):
    wavelength = widgets.Checkbox(
    value=False,
    description='Wavelength (nm)',
    tooltip='Plot spectra in wavelength (nm)',
    disabled=False,
    indent=True
    )
    kw['wave'] = wavelength
    return wavelength, kw


def vial(kw):
    vial = widgets.Checkbox(
    value=False,
    description='Show vial',
    tooltip='Show color vial in the plot',
    disabled=False,
    indent=True
    )
    kw['vial'] = vial
    return vial, kw

def net(kw):
    network = widgets.Checkbox(
    value=False,
    description='Network',
    tooltip='Map susceptibility to spectrum',
    disabled=False,
    indent=True
    )
    kw['net'] = network
    return network, kw

def decomp(kw):
    decomposing = widgets.Checkbox(
    value=False,
    description='Decompose Absorption',
    tooltip='Decompose absorption spectra into states',
    disabled=False,
    indent=True
    )
    kw['decomp'] = decomposing
    return decomposing, kw

def nstates(kw):
    n_states = widgets.BoundedFloatText(
    value=-1,
    min=-1,
    max=100,
    step=1,
    description='# of states:',
    tooltip='Number of states to include in absorption spectrum (-1 for all)',
    disabled=False
    )
    kw['nstates'] = n_states
    return n_states, kw

def maxx(kw):
    maxx = widgets.BoundedFloatText(
    value=0,
    min=0,
    max=10000,
    step=0.1,
    description='Max. X:',
    tooltip='Right limit of plot',
    disabled=False
    )
    kw['maxx'] = maxx
    return maxx, kw

def ensemble(kw):
    ens = widgets.Checkbox(
    value=False,
    description='Ensemble Avg',
    tooltip='Toggle to go from weighted average to ensemble average',
    disabled=False,
    indent=False
    )
    kw['ens'] = ens
    return ens, kw

def cutoff(kw):
    cut = widgets.FloatSlider(
    value=10,
        min=0,
        max=100,
        step=5,
        description='Cutoff (%):',
        disabled=False,
        tooltip='Cutoff yield for displaying rates (0-100)',
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.0f',
    )
    kw['cutoff'] = cut
    return cut, kw

def legend(kw):
    lege = widgets.Checkbox(
    value=False,
    description='Display rates',
    tooltip='Display rates in the diagram',
    disabled=False,
    indent=False
    )
    kw['legend'] = lege
    return lege, kw

def gran_slider(kw):
    slider = widgets.FloatSlider(
    value=-2,
    max=-1,
    min=-3,
    step=1,
    description='Bin $10^x$ (eV)',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='.0f'
    )
    kw['gran'] = slider
    return slider, kw

def check_for_updates(package_name):
    try:
        # Get the currently installed version
        installed_version = pkg_resources.get_distribution(package_name).version
        
        # Fetch the latest version from PyPI
        response = requests.get(f'https://pypi.org/pypi/{package_name}/json')
        response.raise_for_status()
        latest_version = response.json()['info']['version']

        # Compare versions
        if installed_version != latest_version:
            text = f"ATTENTION: Update available! {package_name} {installed_version} -> {latest_version}\n"
            text += f"Run `pip install --upgrade {package_name}` to update."
            n2 = widgets.HTML(value = f'<b>{text}<b>')
            return n2
        else:
            return None
    except Exception as e:
        print(f"An error occurred while checking for updates: {e}")