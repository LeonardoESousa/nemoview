import warnings
import os
import nemo.tools
import nemo.analysis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from scipy.linalg import expm
import pandas as pd
from IPython.display import display
# pylint: disable=unbalanced-tuple-unpacking

THECOLOR = "black"
cmap = plt.get_cmap("cividis")


def set_fontsize(ax):
    fig_size = ax.get_figure().get_size_inches()
    # define font size based dynamically on figure size
    fontsize = max(fig_size[0] * 100 / 72, 14)
    return fontsize


def check(ax, xmin, xmax):
    x = sorted([xmin, xmax])
    y = None
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xs = list(sorted(vert[:, 0]))
            if xs == x and 0 not in vert[:, 1]:
                y = vert[1, 1]
        except (IndexError, AttributeError):
            pass
    return y


def fill(ax, xmin, xmax, y, text):
    fontsize = set_fontsize(ax)
    newy = check(ax, xmin, xmax)
    try:
        ax.fill_between([xmin, xmax], y, newy, alpha=0.5, hatch="x", color=cmap(0.5))
        txt_x = xmin + (xmax - xmin) / 2
        for txt in ax.texts:
            if txt.get_position()[0] == txt_x and txt.get_position()[1] != -0.4:
                txt.set_visible(False)
        ax.text(
            x=txt_x,
            y= 0.95 * min(newy, y),
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )
    except TypeError :
        ax.text(
            x=xmin + (xmax - xmin) / 2,
            y=0.95 * y,
            s=text,
            ha="center",
            va="top",
            color=THECOLOR,
            fontsize=fontsize,
        )

def format_number(rate, error_rate, unit="s^-1"):
    # Check if the rate is zero
    if rate <= 1e-99:
        return f"0 ± 0 {unit}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = np.floor(np.nan_to_num(np.log10(rate)))

    # Adjust exponent to ensure the first number is >= 1
    if rate / 10**exp < 1:
        exp -= 1

    # Determine the number of significant figures for rate and error_rate
    rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure
    error_rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure

    # Format the string without using LaTeX
    if exp != 0:
        formatted_rate = f"{rate/10**exp:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate/10**exp:.{error_rate_sig_figs}f}"
        formatted_string = f"({formatted_rate} ± {formatted_error_rate}) x 10^{int(exp)} {unit}"
    else:
        formatted_rate = f"{rate:.{rate_sig_figs}f}"
        formatted_error_rate = f"{error_rate:.{error_rate_sig_figs}f}"
        formatted_string = f"{formatted_rate} ± {formatted_error_rate} {unit}"

    return formatted_string

def format_rate(rate, error_rate, unit="$s^{-1}$"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = np.floor(np.nan_to_num(np.log10(rate)))

    # Adjust exponent to ensure the first number is >= 1
    if rate / 10**exp < 1:
        exp -= 1

    # Determine the number of significant figures for rate and error_rate
    rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure
    error_rate_sig_figs = max(0, -int(np.floor(np.log10(error_rate / 10**exp))))  # Ensure at least 1 significant figure

    if exp != 0:
        formatted_string = f"${rate/10**exp:.{rate_sig_figs}f}\\pm{error_rate/10**exp:.{error_rate_sig_figs}f}\\times10^{{{exp:.0f}}}$ " + unit
    else:
        formatted_string = f"${rate:.{rate_sig_figs}f}\\pm{error_rate:.{error_rate_sig_figs}f}$ " + unit

    return formatted_string

#def format_rate(rate, error_rate, unit="$s^{-1}$"):
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        exp = np.round((np.nan_to_num(np.log10(rate))),0)
#    if exp < -100:
#        exp = -100
#    if exp != 0:
#        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}\\times10^{{{exp:.0f}}}$ "+unit
#    else:
#        formatted_string = f"${rate/10**exp:.1f}\\pm{error_rate/10**exp:.1f}$ "+unit
#    return formatted_string


class State:
    def __init__(self):
        self.smin, self.smax = 0, 3.3
        self.tmin, self.tmax = 3.7, 7
        #singlets = [i for i in alvos if "S" in i and "0" not in i]
        #triplets = [i for i in alvos if "T" in i and "0" not in i]
        self.size = 0.5

    def color(self, state):
        num = int(state[1:])
        if "S" in state:
            return cmap(0.1 + (num - 1) * 0.4 / 3)
        else:
            return cmap(0.9 - (num - 1) * 0.4 / 3)

    def x(self, state):
        factor = 0.1
        num = int(state[1:])
        if "S" in state:
            xmin = self.smax - num * (1 + factor) * self.size
            return xmin, xmin + self.size
        else:
            xmin = self.tmin + (num - 1) * (1 + factor) * self.size
            return xmin, xmin + self.size

    def arrow(self, state, alvo):
        if "T" in state and "T" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + 3 * self.size / 4, 1
        elif "T" in state and "S" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + 3 * self.size / 4, 1
        elif "S" in state and "T" in alvo:
            x_i, x_f, factor = self.x(state)[1], self.x(alvo)[0] + self.size / 4, -1
        else:
            x_i, x_f, factor = self.x(state)[0], self.x(alvo)[0], 1
        return x_i, x_f, factor


def relu(x):
    return np.maximum(0.3, x)


def plot_transitions(data, ax, cutoff):
    cutoff = cutoff / 100
    fontsize = set_fontsize(ax)
    lw = fontsize / 2
    rates = data["Rate"].to_numpy()
    error = data["Error"].to_numpy()
    transitions = data["Transition"].to_numpy()
    weights = data["Prob"].to_numpy()/100
    #weights /= np.sum(weights)
    #weights = np.round(weights, 4)
    energies = data["AvgDE+L"].to_numpy()
    energies[1:] += energies[0]
    base = energies[0]
    state = transitions[0].split(">")[0][:-1]
    num = int(state[1:])
    alvos = [i.split(">")[1] for i in transitions]
    trans = [i.split(">")[0][-1] for i in transitions]
    S = State()
    ##Makes S0 lines
    xmin, xmax = S.x(state)
    fill(ax, xmin, xmax, base, f"{state[0]}$_{num}$")
    ax.hlines(y=base, xmin=xmin, xmax=xmax, lw=lw, color=S.color(state))
    ax.hlines(y=0, xmin=xmin, xmax=xmax, lw=lw, color=S.color(state))
    ax.text(
        x=xmin + abs(xmax - xmin) / 2,
        y=-0.4,
        s=f"S$_{0}$",
        ha="center",
        va="center",
        color=THECOLOR,
        fontsize=fontsize,
    )
    ##
    for i, _ in enumerate(energies):
        style = f"Fancy, tail_width={lw}, head_width={lw*3}, head_length={lw*2}"
        kw = dict(
            arrowstyle=style,
            color=S.color(state),
            zorder=10,
            mutation_scale=weights[i],#relu(weights[i]),
        )
        if np.round(weights[i], 2) > cutoff:
            if alvos[i] == "S0":
                xmin, xmax = S.x(state)
                if trans[i] == "-":
                    a3 = patches.FancyArrowPatch(
                        (xmin, base),
                        (xmin, 0),
                        **kw,
                        label=f"{state[0]}$_{state[1]}\\:\\to\\:${alvos[i][0]}$_{alvos[i][1:]}$: "
                        + format_rate(rates[i], error[i]),
                    )
                else:
                    a3 = patches.FancyArrowPatch(
                        (xmax, base),
                        (xmax, 0),
                        connectionstyle=f"arc3,rad={-0.1}",
                        **kw,
                        label=f"{state[0]}$_{state[1]}\\leadsto${alvos[i][0]}$_{alvos[i][1:]}$: "
                        + format_rate(rates[i], error[i]),
                    )
                ax.add_patch(a3)
            else:
                xmin, xmax = S.x(alvos[i])
                _ = check(ax, xmin, xmax)
                fill(ax, xmin, xmax, energies[i], f"{alvos[i][0]}$_{alvos[i][1:]}$")
                ax.hlines(
                    y=energies[i], xmin=xmin, xmax=xmax, lw=lw, color=S.color(state)
                )
                fx, tx, curve = S.arrow(state, alvos[i])
                a3 = patches.FancyArrowPatch(
                    (fx, base),
                    (tx, energies[i]),
                    connectionstyle=f"arc3,rad={curve*0.5}",
                    **kw,
                    label=f"{state[0]}$_{state[1]}\\leadsto${alvos[i][0]}$_{alvos[i][1:]}$: "
                    + format_rate(rates[i], error[i]),
                )
                ax.add_patch(a3)


def write_energies(ax):
    fontsize = set_fontsize(ax)
    xmin = np.inf
    xmax = -np.inf
    yleft, yright = [], []
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xmin = min(xmin, min(vert[:, 0]))
            xmax = max(xmax, max(vert[:, 0]))
        except AttributeError:
            pass
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            y = vert[0, 1]
            if (
                max(vert[:, 0]) - xmin < xmax - min(vert[:, 0])
                and y not in yleft
                and y != 0
            ):
                yleft.append(y)
            elif (
                max(vert[:, 0]) - xmin > xmax - min(vert[:, 0])
                and y not in yright
                and y != 0
            ):
                yright.append(y)
        except AttributeError:
            pass
    dleft, dright = [100], [100]
    for y in sorted(yleft):
        if min(np.abs([y - i for i in dleft])) > fontsize*0.18/13:
            ax.text(
                x=0.98 * xmin,
                y=y,
                s=f"{y:.2f} eV",
                ha="right",
                va="center",
                fontsize=fontsize,
                color=THECOLOR,
            )
            dleft.append(y)
    for y in sorted(yright):
        if min(np.abs([y - i for i in dright])) > fontsize*0.18/13:
            ax.text(
                x=1.02 * xmax,
                y=y,
                s=f"{y:.2f} eV",
                ha="left",
                va="center",
                fontsize=fontsize,
                color=THECOLOR,
            )
            dright.append(y)
    ax.set_xlim([0.9 * xmin, 1.1 * xmax])


def make_diagram(files, dielec, cutoff=0.01):
    _, ax = plt.subplots()
    ax.set_xticklabels([])
    plt.axis("off")
    for file in files:
        data, _ = nemo.analysis.rates(file.split("_")[1], dielec, data=file)
        data.rename(columns=lambda x: x.split("(")[0], inplace=True)
        plot_transitions(data, ax, cutoff)
    # medium = plt.legend(handles=[],title=f'Medium:\n$\epsilon ={dielec[0]}$\n$n={dielec[1]}$',title_fontsize=10, loc='best',frameon=False)
    # ax.add_artist(medium)
    # leg = plt.legend(loc='best',fontsize=10,title=f'$\epsilon ={dielec[0]}$ $n={dielec[1]}$',title_fontsize=10)
    # for item in leg.legendHandles:
    #    item.set_visible(False)
    write_energies(ax)
    # arquivo = nemo.tools.naming('diagram.png')
    # plt.savefig(arquivo,facecolor='white',dpi=300)#, transparent=True)
    return ax


#################################################################################################################################
##PREVENTS OVERWRITING#########################################
def naming(arquivo, folder="."):
    new_arquivo = arquivo
    if arquivo in os.listdir(folder):
        duplo = True
        vers = 2
        while duplo:
            new_arquivo = str(vers) + arquivo
            if new_arquivo in os.listdir(folder):
                vers += 1
            else:
                duplo = False
    return new_arquivo


###############################################################


def spectrum(dx, gran):
    num = int((max(dx) - min(dx)) / gran)
    if num == 0:
        bins = 1
    else:
        bins = np.linspace(min(dx), max(dx), num)
    hist, bins = np.histogram(dx, bins=bins, density=True)
    bins = bins[:-1] + (bins[1:] - bins[:-1]) / 2
    return hist, bins


def drift(data):
    t = data["Time"].to_numpy(dtype=float)
    dx = data["DeltaX"].to_numpy()
    dy = data["DeltaY"].to_numpy()
    dz = data["DeltaZ"].to_numpy()
    mux = np.mean(dx / t)
    muy = np.mean(dy / t)
    muz = np.mean(dz / t)
    return np.array([mux, muy, muz])


def get_peak(y, x, err):
    if err is None:
        max_idx = y.argmax()
        peak = x[max_idx]
        return f'{peak:.2f}', f'{1239.8/peak:.0f}'
    else:
        # fix seed
        np.random.seed(0)
        #sample from gaussian with mean y and std err
        yy = np.random.normal(y, err, size=(1000, len(y)))
        #find x value of maxima
        maxima_x = x[np.argmax(yy, axis=1)]
        peak = np.mean(maxima_x)
        err_p = np.std(maxima_x)
        min_p = 1239.8/(peak+err_p)
        mean_p = 1239.8/(peak)
        max_p = 1239.8/(peak-err_p)
        return f'{peak:.2f} ± {err_p:.2f}', f'{mean_p:.0f} [{min_p:.0f},{max_p:.0f}]'

def vertical_tanh(x, a, b):
    return (a - b) / 2 * np.tanh(3 * (x - 1)) + (a + b) / 2


def network_spectrum(breakdown, ax, initial, process, wave):
    ax1, ax2 = ax
    # get x limits of ax2
    xmin, xmax = ax1.get_xlim()
    x2min, x2max = ax2.get_xlim()
    func = vertical_tanh
    x = np.linspace(-1, 1.5, 100)
    color_map = plt.get_cmap("coolwarm")
    # make list of colors from 0 to 1
    if process == "emi":
        transition = initial + "->S0"
        width = breakdown[transition.upper()].to_numpy()
        d_initial = breakdown["chi_" + initial.lower()].to_numpy()
        d_final = breakdown["eng"].to_numpy()
    else:
        transitions = [col for col in breakdown.columns if "->" in col]
        width = breakdown[transitions].to_numpy().flatten()
        width /= np.max(width)
        d_initial = (
            breakdown[[col for col in breakdown.columns if "chi_" in col]]
            .to_numpy()
            .flatten()
        )
        d_final = (
            breakdown[[col for col in breakdown.columns if "eng_" in col]]
            .to_numpy()
            .flatten()
        )
    width /= np.max(width)
    if wave:
        d_final = 1239.8 / d_final
    scale = (x2max - x2min) / (xmax - xmin)
    d_final = (d_final - xmin) * scale + x2min
    for i in range(breakdown.shape[0]):
        if width[i] > 0.01:
            y = func(x, d_initial[i], d_final[i])
            ax2.plot(y, x, lw=2, alpha=width[i], color=color_map(d_initial[i] / x2max))


# define function that equals a for x=-5 and b for x=5 using tanh
def left_tanh(x, a, b):
    return (b - a) / 2 * np.tanh(3 * x) + (a + b) / 2


def right_tanh(x, a, b):
    return (a - b) / 2 * np.tanh(3 * x) + (a + b) / 2


def plot_network(breakdown, ax, side, transition):
    scheme = {
        "left": {"color": "#4477AA", "func": left_tanh},
        "right": {"color": "#EE6677", "func": right_tanh},
    }
    color = scheme[side]["color"]
    func = scheme[side]["func"]
    initial = transition.split("~>")[0]
    final = transition.split("~>")[1]
    width = breakdown[transition.upper()].to_numpy()
    width /= np.max(width)
    d_initial = breakdown["chi_" + initial.lower()].to_numpy()
    d_final = breakdown["chi_" + final.lower()].to_numpy()
    x = np.linspace(-1, 1, 100)
    for i in range(breakdown.shape[0]):
        if width[i] > 0.01:
            y = func(x, d_initial[i], d_final[i])
            if width[i] == 1:
                ax.plot(
                    x,
                    y,
                    lw=2,
                    alpha=width[i],
                    color=color,
                    label=f"{initial[0].upper()}$_{{{initial[1:]}}}\\leadsto$ {final[0].upper()}$_{{{final[1:]}}}$",
                )
            else:
                ax.plot(x, y, lw=2, alpha=width[i], color=color)
    # hist, bins = np.histogram(width,bins=100)
    # ax22.plot((bins[1:]+bins[:-1])/2,hist/np.sum(hist),color=color)

##CALCULATES FORSTER RADIUS####################################
def radius(acceptor, donor, kappa2):
    acceptor = acceptor.to_numpy()
    xa = acceptor[:, 0]
    ya = acceptor[:, -2]
    dya = acceptor[:, -1]

    xd = donor["Energy"].to_numpy()
    yd = donor["Diffrate"].to_numpy()
    dyd = donor["Error"].to_numpy()

    # Speed of light
    c = 299792458  # m/s

    # Finds the edges of interpolation
    minA = min(xa)
    minD = min(xd)
    maxA = max(xa)
    maxD = max(xd)
    MIN = max(minA, minD)
    MAX = min(maxA, maxD)

    if MIN > MAX:
        return 0, 0
    X = np.linspace(MIN, MAX, 1000)
    f1 = interp1d(xa, ya, kind="cubic")
    f2 = interp1d(xd, yd, kind="cubic")
    f3 = interp1d(xa, dya, kind="cubic")
    f4 = interp1d(xd, dyd, kind="cubic")

    YA = f1(X)
    YD = f2(X)
    DYA = f3(X)
    DYD = f4(X)

    # Calculates the overlap
    Overlap = YA * YD / (X**4)

    # Overlap error
    OverError = Overlap * np.sqrt((DYA / YA) ** 2 + (DYD / YD) ** 2)

    # Integrates overlap
    IntOver = np.trapz(Overlap, X)

    # Integrated Overlap Error
    DeltaOver = np.sqrt(np.trapz((OverError**2), X))

    # Gets lifetime
    emi_rate, emi_error = donor.rate, donor.error
    tau = 1 / emi_rate
    delta_tau = (1/emi_rate)*(emi_error/emi_rate)
    
    # Calculates radius sixth power
    c *= 1e10
    const = (nemo.parser.HBAR_EV**3) * (9 * (c**4) * kappa2 * tau) / (8 * np.pi)
    radius6 = const * IntOver

    # Relative error in radius6
    delta = np.sqrt((DeltaOver / IntOver) ** 2 + (delta_tau / tau) ** 2)

    # Calculates radius
    forster_radius = radius6 ** (1 / 6)

    # Error in radius
    error_forster_radius = forster_radius * delta / 6
    return forster_radius, error_forster_radius

def make_matrix(df2):
    trans = df2['Transition'].to_list()
    initials = [i.split('>')[0][:-1] for i in trans]
    finals = [i for i in trans if 'S0' in i]
    initials = list(set(initials))
    labels = initials + finals
    df = df2.copy()
    #keep only Transition and Rate columns
    df = df[['Transition','Rate']]
    final = []
    initial = []
    # Iterate over the dataframe
    for i in range(0, len(df)):
        transition = df.at[i, 'Transition']
        target_state = transition.split('>')[-1]  # Get the target state
        initial_state = transition.split('>')[0][:-1]
        if target_state not in labels and transition not in labels:
            # Add the rate to the preceding row
            df.at[ifin, 'Rate'] += df.at[i, 'Rate']
            # Mark the current row for removal
            df.at[i, 'Remove'] = True
        elif target_state in labels:
            ifin = i
            final.append(target_state)
            initial.append(initial_state)
        else:
            ifin = i
            final.append(transition)    
            initial.append(initial_state)
    
    # Remove the marked rows
    df = df[df['Remove'] != True].drop(columns=['Remove'])
    df['Initial'] = initial
    df['Final'] = final

    M = np.zeros((len(labels),len(labels)))
    for ini in labels:
        for fin in labels:
            try:
                rate = df['Rate'][(df.Initial == ini) & (df.Final == fin)].to_numpy()[0]
                if 'S0' in fin and 'S0' not in ini:
                    fin2 = fin.split('>')[0][:-1]
                else:
                    fin2 = fin
                if ini == fin2:
                    M[labels.index(ini),labels.index(fin2)] += -rate
                    M[labels.index(fin),labels.index(ini)] += rate
                else:
                    #print(ini, fin2, rate)
                    M[labels.index(ini),labels.index(ini)] += -rate
                    M[labels.index(fin),labels.index(ini)] += rate          
            except IndexError:
                pass
    M = pd.DataFrame(M)
    M.columns = labels
    M.index = labels
    return M, df

def kinetics(total_rates, initial, debug=False):
    M, df = make_matrix(total_rates)
    if debug:
        #format as .2e
        M = M.applymap(lambda x: f'{x:.2e}')
        df['Rate'] = df['Rate'].apply(lambda x: f'{x:.2e}')
        display(df)
        display(M)
    # get index of initial state
    states = M.columns.to_list()
    #count elements that contain 'S0'
    num = sum('S0' not in i for i in states)
    rows = M.index.to_list()
    #take numbers from M without column and row names
    M = M.to_numpy(float)
    dpop = np.zeros((M.shape[0],1))
    dpop[states.index(initial),0] = 100 # Initial population
    pop = dpop
    time = [0] # Initial time
    deltat = 1e-1/np.max(np.abs(M)) # Time step (s)
    while  np.sum(dpop[num:,0]) < 99.0:
        dpop = np.matmul(expm(M*deltat),dpop)
        dpop = (dpop / np.sum(dpop)) * 100
        pop = np.hstack((pop,dpop))
        time.append(time[-1]+deltat)
        deltat = max(0.01*(time[-1]+deltat),deltat)
        # To check progress
        #print(f'Computing... {np.sum(dpop[2:,0]):.1f}%',end="\r", flush=True)
    time = np.array(time)
    #make dataframe with time and populations
    pop = pd.DataFrame(pop)
    pop.index = rows
    pop.columns = time
    return time, pop

###############################################################

def compile(dielec, datas,ensemble_average=False):
    for data in datas:
        rates = data.rate(dielec, ensemble_average=ensemble_average)
        try:
            total_rates = pd.concat([total_rates, rates], axis=0, ignore_index=True)
        except NameError:
            total_rates = rates
    total_rates.rename(columns=lambda x: x.split('(')[0], inplace=True)        
    return total_rates

def trpl(time, pop):
    states = [i for i in pop.index.to_list() if '->S0' in i]
    emission = pop.loc[states].sum().to_numpy()
    #compute derivative of emission
    emission_derivative = np.diff(emission)/np.diff(time)
    y_data = max(emission)*emission_derivative/max(emission_derivative)
    x_data = time[:-1] + (time[1:] - time[:-1])/2
    return x_data, y_data
