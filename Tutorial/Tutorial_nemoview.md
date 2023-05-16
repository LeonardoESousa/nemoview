# Introduction

In the following, we will learn how to use nemoview package (Leonardo Evaristo de Sousa and Piotr de Silva, Journal of Chemical Theory and Computation 2021 17 (9), 5816-5824 DOI: 10.1021/acs.jctc.1c00476)

This is the second part of the tutorial of NEMO where the results have been previously generated following the first tutorial.

The results file EnsembleS0_.lx, EnsembleS1_.lx and EnsembleT1_.lx generated in the first tutorial will be needed here.

In this  will visualize the results using **nemoview**.


## Creation of the visualization environment on your work computer

Download **Visualization code studio** (https://code.visualstudio.com/) and install it on your machine.

Open Visual Studio Code and create a virtual environment 

```    
    python -m venv view and activate this environment : & “.view\Scripts\activate”
Note, to deactivate the environment, run the following command: 
        & “.view\Scripts\deactivate”
```

Import nemoview from github (i.e **Voilà** package)
```
    pip install git+https://github.com/LeonardoESousa/NEMO

    git clone https://github.com/LeonardoESousa/nemoview
    
    cd .\nemoview\

    pip install .
```
Import and install labplot
```   
    pip install git+https://github.com/LeonardoESousa/labplot
```

to open nemoview :
```
$nemoview    
```

Copy the files **Ensemble_S0_.lx**, **Ensemble_S1_.lx**, **Ensemble_T1_.lx** to your work directory on your computer.

Open  *visual studio code* and run a new terminal. Activate the virtual environment and run the following command :
    nemoview

A google chrome window will open. Click on the button **Open** and select the three ensemble files. 
In the three widgets "molecule:" right the same thing for the system to understand the same system is studied. Press *Read file*


A window composed of four sections will appear, **Diagram** that depicts the photophysics parameters, **Spectra** represents the absorption or emission spectra of the compound, **Susceptibility** represents the solvent susceptibility of the compound and  **Network** represents the initial and final susceptibility of a considered transition.

### Diagram

Organization of the energy levels for the S$_1$ (blue) and T$_1$ (yellow) ensembles. The arrow represents the direction of the inter-system-crossing.
One can modulate this vizualisation through the following parameters :
    $\epsilon$ : variation of the dielectric constant of the solvent
    $n_r$ : variation of the refractive index of the solvent
    Cutoff : modulate the number of rows displayed under the diagram. The higher the cutoff, the less row will be displayed.
Below the diagram is deplayed a Table :
    Transition : Transition considered between the two states

    Rate : Conversion rates between the two states

    Error : Error on the rate
    
    Prob : Probability of the transition
    
    AvgDE+L : Average energy difference between the two states
    
    AvgSOC : Average spin-orbit coupling
    
    AvgSigma : Average broadening 
    
    AvgConc : Average of the fraction of the ensemble that contribute to the ensemble rate.
    
### Spectra

Display the absorption, fluorescence and emission spectra of the compound. Once againm the dielectric constant and the refractive index of the solvent can be tuned.
To select or deselect a spectrum, click on the corresponding checkbox while pressing "ctrl" key.

### Susceptibility

For a better visualization, set the curseur **Bin 10^x (eV)** to -3.

The solvent properties can be extrapolated with the modification of the refractive index and the dielectric constant.

The Figure **a** represents the electronic susceptibility of each electronic state. 

The smaller the x value is, the more localized this state is, as can be visualized through the Figure **b** that depicts a strongly Localized excited state for these two states.

### Network

represents the initial and final susceptibility of a considered transition.

