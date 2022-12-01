import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from nemo.tools import naming
import nemo.analysis
import ipywidgets as widgets
import warnings
plt.style.use(['labplot','labplot_note'])

thecolor = 'black'
cmap = plt.get_cmap('cividis')
def check(ax, xmin,xmax):
    x = sorted([xmin,xmax])
    y = None
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xs = list(sorted(vert[:,0]))
            if xs == x and 0 not in vert[:,1]:
                y = vert[1,1]
        except:
            pass   
    return y 

def fill(ax,xmin,xmax,y,text):
    newy = check(ax,xmin,xmax)
    try:
        ax.fill_between([xmin,xmax],y,newy,alpha=0.5,hatch='x',color=cmap(0.5))
        txt_x = xmin+(xmax-xmin)/2
        for txt in ax.texts:
            if txt.get_position()[0] == txt_x and txt.get_position()[1] != 0:
                txt.set_visible(False)
        #ax.text(x=txt_x,y=(newy+y)/2, s=text,ha='center',va='center_baseline',color=thecolor)
        ax.text(x=txt_x,y=0.95*min(newy,y), s=text,ha='center',va='top',color=thecolor)
    except:
        ax.text(x=xmin+(xmax-xmin)/2,y=0.95*y, s=text,ha='center',va='top',color=thecolor)#,backgroundcolor='white')
        
        
def format_rate(r,dr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = int(np.nan_to_num(np.log10(r)))
    if exp < -100:
        exp = -100    
    if exp != 0:
        s=f'${r/10**exp:.1f}\pm{dr/10**exp:.1f}\\times10^{exp}$ $s^{{-1}}$'
        #s=f'${r/10**exp:.1f}\\times10^{exp}$ $s^{{^-1}}$'
    else:
        s=f'${r/10**exp:.1f}\pm{dr/10**exp:.1f}$ $s^{{-1}}$'
        #s=f'${r/10**exp:.1f}$ $s^{{^-1}}$'
    return s

class State():
    def __init__(self,alvos):
        self.smin, self.smax = 0,3.3
        self.tmin, self.tmax = 3.7,7
        singlets = [i for i in alvos if 'S' in i and '0' not in i]
        triplets = [i for i in alvos if 'T' in i and '0' not in i]
        div      = max(len(set(singlets)),len(set(triplets)))
        initial = np.linspace(self.tmin,self.tmax,div)
        self.size = initial[1] - initial[0]

    def color(self,state):
        num  = int(state[1:])
        if 'S' in state:
            return cmap(0.1+(num-1)*0.4/3)
        else:
            return cmap(0.9-(num-1)*0.4/3)  

    def x(self,state):
        factor = 0.1
        num  = int(state[1:])
        if 'S' in state:
            xmin = self.smax - num*(1+factor)*self.size
            return xmin, xmin + self.size
        else:
            xmin = self.tmin + (num-1)*(1+factor)*self.size 
            return xmin, xmin + self.size           

    def arrow(self,state,alvo,delta):
        #delta = np.sign(delta)*np.heaviside(delta-0.01,0)
        if 'T' in state and 'T' in alvo:
             xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+3*self.size/4, 1
        elif 'T' in state and 'S' in alvo:
            xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+3*self.size/4, 1
            #if delta > 0:
            #    xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+3*self.size/4, -1
            #elif delta < 0:
            #    xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+3*self.size/4, 1
            #else:
            #    xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+3*self.size/4, 0
        elif 'S' in state and 'T' in alvo:
            xi, xf, factor = self.x(state)[1], self.x(alvo)[0]+self.size/4, -1
            #if delta > 0:
            #    xi, xf, factor = self.x(state)[0], self.x(alvo)[0], -1
            #elif delta < 0:
            #    xi, xf, factor = self.x(state)[0], self.x(alvo)[0], -1  
            #else:
            #    xi, xf, factor = self.x(state)[0], self.x(alvo)[0], 0   
        else:
            xi, xf, factor = self.x(state)[0], self.x(alvo)[0], 1
        return  xi, xf, factor    

def plot_transitions(data,ax,cutoff):
    lw = 5
    #sigmas      = data['AvgSigma'].values
    rates       = data['Rate'].values
    error       = data['Error'].values 
    transitions = data['Transition'].values
    weights     = data['Prob'].values
    weights    /= np.sum(weights) 
    weights     = np.round(weights,4)
    energies    = data['AvgDE+L'].values
    energies[1:]+= energies[0]
    base        = energies[0]
    state       = transitions[0].split('>')[0][:-1]
    num         = int(state[1:])
    alvos       = [i.split('>')[1] for i in transitions]
    trans       = [i.split('>')[0][-1] for i in transitions]
    S = State(alvos)
    ##Makes S0 lines
    xmin, xmax = S.x(state)
    fill(ax,xmin,xmax,base,f'{state[0]}$_{num}$')
    ax.hlines(y=base,xmin=xmin,xmax=xmax,lw=lw,color=S.color(state))  #####   
    ax.hlines(y=0,xmin=xmin,xmax=xmax,lw=lw,color=S.color(state))     
    ax.text(x=xmin+abs(xmax-xmin)/2,y=0, s=f'S$_{0}$',ha='center',va='center',color=thecolor,backgroundcolor='white')
    ##
    for i in range(len(energies)):
        style = f"Fancy, tail_width=4, head_width=12, head_length=8"#, tail_width={4*weights[i]}, head_width={12*weights[i]}, head_length={8*weights[i]}"
        kw = dict(arrowstyle=style, color=S.color(state),zorder=10,mutation_scale=weights[i])
        if np.round(weights[i],2) > cutoff or (alvos[i] == 'S0' and np.round(weights[i],2) > 0):
            if alvos[i] == 'S0':          
                xmin, xmax = S.x(state)
                if trans[i] == '-':      
                    a3 = patches.FancyArrowPatch((xmin, base), (xmin, 0),**kw,label=f'{state[0]}$_{state[1]}\\: \\to \\:${alvos[i][0]}$_{alvos[i][1:]}$: '+format_rate(rates[i],error[i]))
                else:
                    a3 = patches.FancyArrowPatch((xmax, base), (xmax, 0),connectionstyle=f"arc3,rad={-0.1}",**kw,label=f'{state[0]}$_{state[1]}\\leadsto${alvos[i][0]}$_{alvos[i][1:]}$: '+format_rate(rates[i],error[i]))
                ax.add_patch(a3)
                #ax.text(x=1.01*xmin+abs(xmax-xmin)/2,y=base/3, s=f'{state[0]}$_{state[1]}\\to$S$_0$:\n '+format_rate(rates[i],error[i]),ha='center',va='center',fontsize=10)#,backgroundcolor='white')
            else:
                xmin, xmax = S.x(alvos[i])
                newy = check(ax,xmin,xmax)
                fill(ax,xmin,xmax,energies[i],f'{alvos[i][0]}$_{alvos[i][1:]}$')
                ax.hlines(y=energies[i],xmin=xmin,xmax=xmax,lw=lw,color=S.color(state))
                fx, tx, curve = S.arrow(state,alvos[i], energies[i]-base)
                a3 = patches.FancyArrowPatch((fx, base), (tx, energies[i]),connectionstyle=f"arc3,rad={curve*0.5}",**kw,label=f'{state[0]}$_{state[1]}\\leadsto${alvos[i][0]}$_{alvos[i][1:]}$: '+format_rate(rates[i],error[i]))
                ax.add_patch(a3)
                #pos = a3.get_path().vertices
                #ax.text(x=smax+(tmin-smax)/2,y=np.max(pos[:,1]), s=format_rate(dataS[i,1],dataS[i,2]),ha='center',va='center',fontsize=12)#,backgroundcolor='white')
                           
def write_energies(ax):
    xmin = np.inf
    xmax = -np.inf
    yleft, yright = [], []
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            xmin = min(xmin,min(vert[:,0]))
            xmax = max(xmax,max(vert[:,0]))
        except:
            pass
    for elem in ax.get_children():
        try:
            vert = elem.get_paths()[0].vertices
            y = vert[0,1]
            if max(vert[:,0]) - xmin < xmax -min(vert[:,0]) and y not in yleft and y != 0:
                yleft.append(y) 
            elif max(vert[:,0]) - xmin > xmax -min(vert[:,0]) and y not in yright and y != 0:
                yright.append(y)
        except:
            pass 
    dleft, dright = [100], [100]
    for y in sorted(yleft):
        if min(np.abs([y-i for i in dleft])) > 0.12:    
            ax.text(x=0.98*xmin,y=y, s=f'{y:.2f} eV',ha='right',va='center',fontsize=10,color=thecolor)#,backgroundcolor='white')    
            dleft.append(y)
    for y in sorted(yright):    
        if min(np.abs([y-i for i in dright])) > 0.12:
            ax.text(x=1.02*xmax,y=y, s=f'{y:.2f} eV',ha='left',va='center',fontsize=10,color=thecolor)#,backgroundcolor='white')          
            dright.append(y)
    ax.set_xlim([0.9*xmin,1.1*xmax])

def make_diagram(files,dielec,cutoff=0.01):
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    plt.axis('off')
    for file in files:
        data, emi = nemo.analysis.rates(file.split('_')[1],dielec,data=file)
        data.rename(columns=lambda x: x.split('(')[0], inplace=True)
        plot_transitions(data,ax,cutoff)    
    #medium = plt.legend(handles=[],title=f'Medium:\n$\epsilon ={dielec[0]}$\n$n={dielec[1]}$',title_fontsize=10, loc='best',frameon=False)
    #ax.add_artist(medium)
    #leg = plt.legend(loc='best',fontsize=10,title=f'$\epsilon ={dielec[0]}$ $n={dielec[1]}$',title_fontsize=10)
    #for item in leg.legendHandles:
    #    item.set_visible(False)
    write_energies(ax)
    arquivo = naming('diagram.png')
    #plt.savefig(arquivo,facecolor='white',dpi=300)#, transparent=True)
    return ax
     
#dielec = (3.8,1.5)     
#make_diagram(['Ensemble_s1_.lx','Ensemble_t1_.lx','Ensemble_t2_.lx'],dielec,cutoff=0.05)                             


#################################################################################################################################
    
def trpl(times,bin_num=10):
    num = int((max(times)-min(times))/bin_num)
    hist, bins = np.histogram(times, bins=np.linspace(min(times),max(times),num),density=True)    
    bins = bins[:-1] +(bins[1:] - bins[:-1])/2
    return  hist, bins

    
def spectrum(dx,gran):
    num = int((max(dx)-min(dx))/gran)
    if num == 0:
        bins = 1
    else:
        bins = np.linspace(min(dx),max(dx),num)    
    hist, bins = np.histogram(dx,bins=bins,density=True)
    bins = bins[:-1] + (bins[1:] - bins[:-1])/2    
    return hist,bins
    
def drift(data):
    t  = data['Time'].to_numpy(dtype=float)
    dx = data['DeltaX'].to_numpy()
    dy = data['DeltaY'].to_numpy()
    dz = data['DeltaZ'].to_numpy()
    mux = np.mean(dx/t)
    muy = np.mean(dy/t)
    muz = np.mean(dz/t)
    return np.array([mux,muy,muz])

def get_peak(hist,bins):
    ind   = np.where(hist == max(hist))[0][0]
    peak  = bins[ind]
    return peak


def eps_nr(eps0=1,nr0=1):
    eps = widgets.BoundedFloatText(
        value=eps0,
        min=1,
        max=100.0,
        step=0.1,
        description='$\epsilon$',
        disabled=False
    )
    
    nr = widgets.BoundedFloatText(
        value=nr0,
        step=0.1,
        min=1,
        max=10.0,
        description='$n_r$',
        disabled=False
    ) 
    return eps,nr
    
    
    
   

