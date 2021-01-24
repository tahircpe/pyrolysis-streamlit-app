"""
Plot the kinetic reactions of biomass pyrolysis for the Ranzi 2014 kinetic
scheme for biomass pyrolysis.

Reference:
Ranzi, Corbetta, Pierucci, 2014. Chemical Engineering Science, 110, pp 2-12.
"""

import numpy as np
import pandas as pd

# Parameters
# ------------------------------------------------------------------------------

# T = 773  # temperature for rate constants, K

# weight percent (%) cellulose, hemicellulose, lignin for beech wood
# wtcell = 48
# wthemi = 28
# wtlig = 24

# dt = 0.001                              # time step, delta t
# tmax = 4                                # max time, s
# t = np.linspace(0, tmax, num=int(tmax/dt))   # time vector
# nt = len(t)                             # total number of time steps

# Functions for Ranzi 2014 Kinetic Scheme
# ------------------------------------------------------------------------------

def ranzicell(wood, wt, T, dt, nt):
    """
    Cellulose reactions CELL from Ranzi 2014 paper for biomass pyrolysis.

    Parameters
    ----------
    wood = wood concentration, kg/m^3
    wt = weight percent wood as cellulose, %
    T = temperature, K
    dt = time step, s
    nt = total number of time steps

    Returns
    -------
    main = mass concentration of main group, (-)
    prod = mass concentration of product group, (-)
    """

    # vector for initial wood concentration, kg/m^3
    pw = np.ones(nt)*wood

    # vectors to store main product concentrations, kg/m^3
    cell = pw*(wt/100)      # initial cellulose conc. in wood
    g1 = np.zeros(nt)       # G1
    cella = np.zeros(nt)    # CELLA
    lvg = np.zeros(nt)      # LVG
    g4 = np.zeros(nt)       # G4

    R = 1.987   # universal gas constant, kcal/kmol*K

    # reaction rate constant for each reaction, 1/s
    # A = pre-factor (1/s) and E = activation energy (kcal/kmol)
    K1 = 4e7 * np.exp(-31000 / (R * T))         # CELL -> G1
    K2 = 4e13 * np.exp(-45000 / (R * T))        # CELL -> CELLA
    K3 = 1.8 * T * np.exp(-10000 / (R * T))     # CELLA -> LVG
    K4 = 0.5e9 * np.exp(-29000 / (R * T))       # CELLA -> G4

    # sum of moles in each group, mol
    sumg1 = 11      # sum of G1
    sumg4 = 4.08    # sum of G4

    # calculate concentrations for main groups, kg/m^3
    for i in range(1, nt):
        r1 = K1 * cell[i-1]     # CELL -> G1
        r2 = K2 * cell[i-1]     # CELL -> CELLA
        r3 = K3 * cella[i-1]    # CELLA -> LVG
        r4 = K4 * cella[i-1]    # CELLA -> G4
        cell[i] = cell[i-1] - (r1+r2)*dt            # CELL
        g1[i] = g1[i-1] + r1*dt                     # G1
        cella[i] = cella[i-1] + r2*dt - (r3+r4)*dt  # CELLA
        lvg[i] = lvg[i-1] + r3*dt                   # LVG
        g4[i] = g4[i-1] + r4*dt                     # G4

    # store main groups in array
    main = np.array([cell, g1, cella, lvg, g4])

    # total group concentration per total moles in that group, (kg/m^3) / mol
    fg1 = g1/sumg1  # fraction of G1
    fg4 = g4/sumg4  # fraction of G4

    # array to store product concentrations as a density, kg/m^3
    prod = np.zeros([21, nt])
    prod[0] = 0.16*fg4              # CO
    prod[1] = 0.21*fg4              # CO2
    prod[2] = 0.4*fg4               # CH2O
    prod[3] = 0.02*fg4              # HCOOH
    prod[5] = 0.1*fg4               # CH4
    prod[6] = 0.2*fg4               # Glyox
    prod[8] = 0.1*fg4               # C2H4O
    prod[9] = 0.8*fg4               # HAA
    prod[11] = 0.3*fg4              # C3H6O
    prod[14] = 0.25*fg4             # HMFU
    prod[15] = lvg                  # LVG
    prod[18] = 0.1*fg4              # H2
    prod[19] = 5*fg1 + 0.83*fg4     # H2O
    prod[20] = 6*fg1 + 0.61*fg4     # Char

    # return arrays of main groups and products as mass fraction, (-)
    return main/wood, prod/wood


def ranzihemi(wood, wt, T, dt, nt):
    """
    Hemicellulose reactions HCE from Ranzi 2014 paper for biomass pyrolysis.

    Parameters
    ----------
    wood = wood density, kg/m^3
    wt = weight percent of hemicellulose, %
    T = temperature, K
    dt = time step, s
    nt = total number of time steps

    Returns
    -------
    main/wood = mass fraction of main group, (-)
    prod/wood = mass fraction of product group, (-)
    """

    # vector for initial wood concentration, kg/m^3
    pw = np.ones(nt)*wood

    # vectors to store main product concentrations, kg/m^3
    hce = pw*(wt/100)   # initial hemicellulose conc. in wood
    g1 = np.zeros(nt)   # G1
    g2 = np.zeros(nt)   # G2
    g3 = np.zeros(nt)   # G3
    g4 = np.zeros(nt)   # G4
    xyl = np.zeros(nt)  # Xylan

    R = 1.987   # universal gas constant, kcal/kmol*K

    # reaction rate constant for each reaction, 1/s
    # A = pre-factor (1/s) and E = activation energy (kcal/kmol)
    K1 = 0.33e10 * np.exp(-31000 / (R * T))     # HCE -> G1
    K2 = 0.33e10 * np.exp(-33000 / (R * T))     # HCE2 -> G2
    K3 = 0.05 * T * np.exp(-8000 / (R * T))     # HCE1 -> G3
    K4 = 1e9 * np.exp(-32000 / (R * T))         # HCE1 -> G4
    K5 = 0.9 * T * np.exp(-11000 / (R * T))     # HCE1 -> Xylan

    # sum of moles in each group, mol
    sumg2 = 4.625   # sum of G2
    sumg3 = 4.875   # sum of G3
    sumg4 = 4.775   # sum of G4

    # calculate concentrations for main groups, kg/m^3
    # where HCE1 as 0.4*g1/(0.4+0.6) and HCE2 as 0.6*g1/(0.4+0.6)
    for i in range(1, nt):
        r1 = K1 * hce[i-1]      # HCE -> G1
        r2 = K2 * 0.6*g1[i-1]   # HCE2 -> G2
        r3 = K3 * 0.4*g1[i-1]   # HCE1 -> G3
        r4 = K4 * 0.4*g1[i-1]   # HCE1 -> G4
        r5 = K5 * 0.4*g1[i-1]   # HCE1 -> Xylan
        hce[i] = hce[i-1] - r1*dt                   # HCE
        g1[i] = g1[i-1] + r1*dt - (r2+r3+r4+r5)*dt  # G1
        g2[i] = g2[i-1] + r2*dt                     # G2
        g3[i] = g3[i-1] + r3*dt                     # G3
        g4[i] = g4[i-1] + r4*dt                     # G4
        xyl[i] = xyl[i-1] + r5*dt                   # Xylan

    # store main groups in array
    main = np.array([hce, g1, g2, g3, g4, xyl])

    # total group concentration per total moles in that group, (kg/m^3)/mol
    fg2 = g2/sumg2  # fraction of G2
    fg3 = g3/sumg3  # fraction of G3
    fg4 = g4/sumg4  # fraction of G4

    # array to store product concentrations as a density, kg/m^3
    prod = np.zeros([21, nt])
    prod[0] = 0.175*fg2 + (0.3 + 0.15)*fg3 + 0.5*fg4                # CO
    prod[1] = (0.275+0.4)*fg2 + (0.5+0.25)*fg3 + (0.5+0.275)*fg4    # CO2
    prod[2] = (0.5+0.925)*fg2 + 1.7*fg3 + (0.8+0.4)*fg4             # CH2O
    prod[3] = 0.025*fg2 + 0.05*fg3 + 0.025*fg4                      # HCOOH
    prod[4] = 0.3*fg2 + (0.1+0.45)*fg4                              # CH3OH
    prod[5] = 0.25*fg2 + 0.625*fg3 + 0.325*fg4                      # CH4
    prod[7] = 0.275*fg2 + 0.375*fg3 + 0.25*fg4                      # C2H4
    prod[9] = 0.2*fg2                                               # HAA
    prod[10] = 0.1*fg2 + 0.125*fg4                                  # C2H5OH
    prod[12] = xyl                                                  # Xylan
    prod[18] = 0.125*fg4                                            # H2
    prod[19] = 0.2*fg2 + 0.25*fg3 + 0.025*fg4                       # H2O
    prod[20] = 1*fg2 + 0.675*fg3 + 0.875*fg4                        # Char

    # return arrays of main groups and products as mass fraction, (-)
    return main/wood, prod/wood


def ranziligc(wood, wt, T, dt, nt):
    """
    Lignin carbon rich reactions LIG-C from Ranzi 2014 paper for biomass pyrolysis.

    Parameters
    ----------
    wood = wood density, kg/m^3
    wt = weight percent of lignin-c, %
    T = temperature, K
    dt = time step, s
    nt = total number of time steps

    Returns
    -------
    main/wood = mass fraction of main group, (-)
    prod/wood = mass fraction of product group, (-)
    """

    # vector for initial wood concentration, kg/m^3
    pw = np.ones(nt)*wood

    # vectors to store main product concentrations, kg/m^3
    ligc = pw*(wt/100/3)    # initial lignin in wood, assume 1/3 of total lignin
    g1 = np.zeros(nt)
    g2 = np.zeros(nt)

    R = 1.987       # universal gas constant, kcal/kmol*K

    # reaction rate constant for each reaction, 1/s
    # A = pre-factor (1/s) and E = activation energy (kcal/kmol)
    K1 = 1.33e15 * np.exp(-48500 / (R * T))     # LIG-C -> G1
    K2 = 1.6e6 * np.exp(-31500 / (R * T))       # LIG-CC -> G2

    # sum of moles in each group, mol
    sumg1 = 9.49    # sum of G1
    sumg2 = 11.35   # sum of G2

    # calculate concentrations for main groups, kg/m^3
    for i in range(1, nt):
        r1 = K1 * ligc[i-1]             # LIG-C -> G1
        r2 = K2 * 0.35*g1[i-1]/sumg1    # LIG-CC -> G2
        ligc[i] = ligc[i-1] - r1*dt         # LIG-C
        g1[i] = g1[i-1] + r1*dt - r2*dt     # G1
        g2[i] = g2[i-1] + r2*dt             # G2

    # store main groups in array
    main = np.array([ligc, g1, g2])

    # total group concentration per total moles in that group, (kg/m^3)/mol
    fg1 = g1/sumg1  # fraction of G1
    fg2 = g2/sumg2  # fraction of G2

    # array to store product concentrations as a density, kg/m^3
    prod = np.zeros([21, nt])
    prod[0] = 0.32*fg1 + (0.4 + 0.4)*fg2    # CO
    prod[2] = (0.3 + 0.7)*fg1 + 1*fg2       # CH2O
    prod[5] = 0.495*fg1 + 0.65*fg2          # CH4
    prod[7] = 0.41*fg1 + 0.6*fg2            # C2H4
    prod[9] = 0.35*fg2                      # HAA
    prod[13] = 0.08*fg1 + 0.2*fg2           # Phenol
    prod[16] = 0.1*fg1 + 0.3*fg2            # Coumaryl
    prod[19] = 1*fg1 + 0.7*fg2              # H2O
    prod[20] = 5.735*fg1 + 6.75*fg2         # Char

    # return arrays of main groups and products as mass fractions, (-)
    return main/wood, prod/wood


def ranziligh(wood, wt, T, dt, nt):
    """
    Lignin hydrogen rich reactions LIG-H from Ranzi 2014 paper for biomass pyrolysis.

    Parameters
    ----------
    wood = wood density, kg/m^3
    wt = weight percent of lignin-h, %
    T = temperature, K
    dt = time step, s
    nt = total number of time steps

    Returns
    -------
    main/wood = mass fraction of main group, (-)
    prod/wood = mass fraction of product group, (-)
    """

    # vector for initial wood concentration, kg/m^3
    pw = np.ones(nt)*wood

    # vectors to store main product concentrations, kg/m^3
    ligh = pw*(wt/100/3)    # initial lignin in wood, assume 1/3 of total lignin
    g1 = np.zeros(nt)       # G1
    g2 = np.zeros(nt)       # G2
    g3 = np.zeros(nt)       # G3
    g4 = np.zeros(nt)       # G4
    g5 = np.zeros(nt)       # G4
    fe2macr = np.zeros(nt)  # FE2MACR

    R = 1.987       # universal gas constant, kcal/kmol*K

    # reaction rate constant for each reaction, 1/s
    # A = pre-factor (1/s) and E = activation energy (kcal/kmol)
    K1 = 0.67e13 * np.exp(-37500 / (R * T))     # LIG-H -> G1
    K2 = 33 * np.exp(-15000 / (R * T))          # LIG-OH -> G2
    K3 = 0.5e8 * np.exp(-30000 / (R * T))       # LIG-OH -> LIG
    K4 = 0.083 * T * np.exp(-8000 / (R * T))    # LIG -> G4
    K5 = 0.4e9 * np.exp(-30000 / (R * T))       # LIG -> G5
    K6 = 2.4 * T * np.exp(-12000 / (R * T))     # LIG -> FE2MACR

    # sum of moles in each group, mol
    sumg1 = 2       # sum of G1
    sumg2 = 20.7    # sum of G2
    sumg3 = 9.85    # sum of G3
    sumg4 = 11.1    # sum of G4
    sumg5 = 10.7    # sum of G5

    # calculate concentrations for main groups, kg/m^3
    for i in range(1, nt):
        r1 = K1 * ligh[i-1]             # LIG-H -> G1
        r2 = K2 * 1*g1[i-1]/sumg1       # LIG-OH -> G2
        r3 = K3 * 1*g1[i-1]/sumg1       # LIG-OH -> LIG
        r4 = K4 * 1*g3[i-1]/sumg3       # LIG -> G4
        r5 = K5 * 1*g3[i-1]/sumg3       # LIG -> G5
        r6 = K6 * 1*g3[i-1]/sumg3       # LIG -> FE2MACR
        ligh[i] = ligh[i-1] - r1*dt                 # LIG-H
        g1[i] = g1[i-1] + r1*dt - (r2+r3)*dt        # G1
        g2[i] = g2[i-1] + r2*dt                     # G2
        g3[i] = g3[i-1] + r3*dt - (r4+r5+r6)*dt     # G3
        g4[i] = g4[i-1] + r4*dt                     # G4
        g5[i] = g5[i-1] + r5*dt                     # G5
        fe2macr[i] = fe2macr[i-1] + r6*dt           # FE2MACR

    # store main groups in array
    main = np.array([ligh, g1, g2, g3, g4, g5, fe2macr])

    # total group concentration per total moles in that group, (kg/m^3)/mol
    fg1 = g1/sumg1  # fraction of G1
    fg2 = g2/sumg2  # fraction of G2
    fg3 = g3/sumg3  # fraction of G3
    fg4 = g4/sumg4  # fraction of G4
    fg5 = g5/sumg5  # fraction of G5

    # array to store product concentrations as a density, kg/m^3
    prod = np.zeros([21, nt])
    prod[0] = (0.5 + 1.6)*fg2 + (0.3 + 1)*fg3 + (0.4 + 0.2)*fg4 + (1 + 0.45)*fg5 # CO
    prod[1] = 0.05*fg3                                              # CO2
    prod[2] = 3.9*fg2 + 0.6*fg3 + (2 + 0.4)*fg4 + (0.2 + 0.5)*fg5   # CH2O
    prod[3] = 0.05*fg3 + 0.05*fg5                                   # HCOOH
    prod[4] = 0.5*fg2 + (0.5 + 0.5)*fg3 + 0.4*fg4 + 0.4*fg5         # CH3OH
    prod[5] = (0.1 + 1.65)*fg2 + (0.1 + 0.35)*fg3 + (0.2 + 0.4)*fg4 + (0.2 + 0.4)*fg5 # CH4
    prod[6] = 0                                                     # Glyox
    prod[7] = 0.3*fg2 + 0.2*fg3 + 0.5*fg4 + 0.65*fg5                # C2H4
    prod[8] = 0.2*fg5                                               # C2H4O
    prod[9] = 0                                                     # HAA
    prod[10] = 0                                                    # C2H5OH
    prod[11] = 1*fg1 + 0.2*fg5                                      # C3H6O
    prod[12] = 0                                                    # Xylan
    prod[13] = 0                                                    # Phenol
    prod[14] = 0                                                    # HMFU
    prod[15] = 0                                                    # LVG
    prod[16] = 0                                                    # Coumaryl
    prod[17] = fe2macr                                              # FE2MACR
    prod[18] = 0.5*fg2 + 0.15*fg3                                   # H2
    prod[19] = 1.5*fg2 + 0.9*fg3 + 0.6*fg4 + 0.95*fg5               # H2O
    prod[20] = 10.15*fg2 + 4.15*fg3 + 6*fg4 + 5.5*fg5               # Char

    # return arrays of main groups and products as mass fractions, (-)
    return main/wood, prod/wood


def ranziligo(wood, wt, T, dt, nt):
    """
    Lignin oxygen rich reactions LIG-O from Ranzi 2014 paper for biomass pyrolysis.

    Parameters
    ----------
    wood = wood density, kg/m^3
    wt = weight percent of lignin-h, %
    T = temperature, K
    dt = time step, s
    nt = total number of time steps

    Returns
    -------
    main/wood = mass fraction of main group, (-)
    prod/wood = mass fraction of product group, (-)
    """

    # vector for initial wood concentration, kg/m^3
    pw = np.ones(nt)*wood

    # vectors to store main product concentrations, kg/m^3
    ligo = pw*(wt/100/3)    # initial lignin in wood, assume 1/3 of total lignin
    g1 = np.zeros(nt)       # G1
    g2 = np.zeros(nt)       # G2
    g3 = np.zeros(nt)       # G3
    g4 = np.zeros(nt)       # G4
    g5 = np.zeros(nt)       # G4
    fe2macr = np.zeros(nt)  # FE2MACR

    R = 1.987       # universal gas constant, kcal/kmol*K

    # reaction rate constant for each reaction, 1/s
    # A = pre-factor (1/s) and E = activation energy (kcal/kmol)
    K1 = 0.33e9 * np.exp(-25500 / (R * T))      # LIG-O -> G1
    K2 = 33 * np.exp(-15000 / (R * T))          # LIG-OH -> G2
    K3 = 0.5e8 * np.exp(-30000 / (R * T))       # LIG-OH -> LIG
    K4 = 0.083 * T * np.exp(-8000 / (R * T))    # LIG -> G4
    K5 = 0.4e9 * np.exp(-30000 / (R * T))       # LIG -> G5
    K6 = 2.4 * T * np.exp(-12000 / (R * T))     # LIG -> FE2MACR

    # sum of moles in each group, mol
    sumg1 = 2       # sum of G1
    sumg2 = 20.7    # sum of G2
    sumg3 = 9.85    # sum of G3
    sumg4 = 11.1    # sum of G4
    sumg5 = 10.7    # sum of G5

    # calculate concentrations for main groups, kg/m^3
    for i in range(1, nt):
        r1 = K1 * ligo[i-1]             # LIG-O -> G1
        r2 = K2 * 1*g1[i-1]/sumg1       # LIG-OH -> G2
        r3 = K3 * 1*g1[i-1]/sumg1       # LIG-OH -> LIG
        r4 = K4 * 1*g3[i-1]/sumg3       # LIG -> G4
        r5 = K5 * 1*g3[i-1]/sumg3       # LIG -> G5
        r6 = K6 * 1*g3[i-1]/sumg3       # LIG -> FE2MACR
        ligo[i] = ligo[i-1] - r1*dt                 # LIG-H
        g1[i] = g1[i-1] + r1*dt - (r2+r3)*dt        # G1
        g2[i] = g2[i-1] + r2*dt                     # G2
        g3[i] = g3[i-1] + r3*dt - (r4+r5+r6)*dt     # G3
        g4[i] = g4[i-1] + r4*dt                     # G4
        g5[i] = g5[i-1] + r5*dt                     # G5
        fe2macr[i] = fe2macr[i-1] + r6*dt           # FE2MACR

    # store main groups in array
    main = np.array([ligo, g1, g2, g3, g4, g5, fe2macr])

    # total group concentration per total moles in that group, (kg/m^3)/mol
    fg1 = g1/sumg1  # fraction of G1
    fg2 = g2/sumg2  # fraction of G2
    fg3 = g3/sumg3  # fraction of G3
    fg4 = g4/sumg4  # fraction of G4
    fg5 = g5/sumg5  # fraction of G5

    # array to store product concentrations as a density, kg/m^3
    prod = np.zeros([21, nt])
    prod[0] = (0.5 + 1.6)*fg2 + (0.3 + 1)*fg3 + (0.4 + 0.2)*fg4 + (1 + 0.45)*fg5 # CO
    prod[1] = 1*fg1 + 0.05*fg3                                      # CO2
    prod[2] = 3.9*fg2 + 0.6*fg3 + (2 + 0.4)*fg4 + (0.2 + 0.5)*fg5   # CH2O
    prod[3] = 0.05*fg3 + 0.05*fg5                                   # HCOOH
    prod[4] = 0.5*fg2 + (0.5 + 0.5)*fg3 + 0.4*fg4 + 0.4*fg5         # CH3OH
    prod[5] = (0.1 + 1.65)*fg2 + (0.1 + 0.35)*fg3 + (0.2 + 0.4)*fg4 + (0.2 + 0.4)*fg5 # CH4
    prod[6] = 0                                                     # Glyox
    prod[7] = 0.3*fg2 + 0.2*fg3 + 0.5*fg4 + 0.65*fg5                # C2H4
    prod[8] = 0.2*fg5                                               # C2H4O
    prod[9] = 0                                                     # HAA
    prod[10] = 0                                                    # C2H5OH
    prod[11] = 0.2*fg5                                              # C3H6O
    prod[12] = 0                                                    # Xylan
    prod[13] = 0                                                    # Phenol
    prod[14] = 0                                                    # HMFU
    prod[15] = 0                                                    # LVG
    prod[16] = 0                                                    # Coumaryl
    prod[17] = fe2macr                                              # FE2MACR
    prod[18] = 0.5*fg2 + 0.15*fg3                                   # H2
    prod[19] = 1.5*fg2 + 0.9*fg3 + 0.6*fg4 + 0.95*fg5               # H2O
    prod[20] = 10.15*fg2 + 4.15*fg3 + 6*fg4 + 5.5*fg5               # Char

    # return arrays of main groups and products as mass fractions, (-)
    return main/wood, prod/wood

# Products from Kinetic Scheme
# ------------------------------------------------------------------------------

def run_ranzi_2014(wtcell, wthemi, wtlig, temp, tmax):
    step = 0.001                                  # time step, delta t
    # tmax = 4                                    # max time, s
    t = np.linspace(0, tmax, num=int(tmax/step))  # time vector
    tot_step = len(t)                             # total number of time steps
    
    # arrays for Ranzi main groups and products as mass fractions, (-)
    pmcell, pcell = ranzicell(1, wt=wtcell, T=temp, dt=step, nt=tot_step)    # cellulose
    pmhemi, phemi = ranzihemi(1, wt=wthemi, T=temp, dt=step, nt=tot_step)    # hemicellulose
    pmligc, pligc = ranziligc(1, wt=wtlig, T=temp, dt=step, nt=tot_step)     # lignin-c
    pmligh, pligh = ranziligh(1, wt=wtlig, T=temp, dt=step, nt=tot_step)     # lignin-h
    pmligo, pligo = ranziligo(1, wt=wtlig, T=temp, dt=step, nt=tot_step)     # lignin-o

    # main cellulose groups as mass fraction, (-)
    cell = pmcell[0]
    g1cell = pmcell[1]
    cella = pmcell[2]
    lvg = pmcell[3]
    g4cell = pmcell[4]
    tcell = cell + g1cell + cella + lvg + g4cell            # total cellulose

    cell_main = {'Time (s)': t, 'cell': cell, 'g1cell': g1cell, 'cella': cella, 'lvg': lvg, 'g4cell': g4cell, 'tcell': tcell}

    df_cell=pd.DataFrame(data=cell_main).set_index('Time (s)')

    # main hemicellulose groups as mass fraction, (-)
    hemi = pmhemi[0]
    g1hemi = pmhemi[1]
    g2hemi = pmhemi[2]
    g3hemi = pmhemi[3]
    g4hemi = pmhemi[4]
    xyl = pmhemi[5]
    themi = hemi + g1hemi + g2hemi + g3hemi + g4hemi + xyl  # total hemicellulose

    hemi_main = {'Time (s)': t, 'hemi': hemi, 'g1hemi': g1hemi, 'g2hemi': g2hemi, 'g3hemi': g3hemi, 'g4hemi': g4hemi, 'xyl': xyl, 'themi': themi}
    df_hemi=pd.DataFrame(data=hemi_main).set_index('Time (s)')

    # main lignin-c groups as mass fraction, (-)
    ligc = pmligc[0]
    g1ligc = pmligc[1]
    g2ligc = pmligc[2]
    tligc = ligc + g1ligc + g2ligc                                       # total lignin-c

    ligc_main = {'Time (s)': t, 'ligc': ligc, 'g1ligc': g1ligc, 'g2ligc': g2ligc, 'tligc': tligc}
    df_ligc=pd.DataFrame(data=ligc_main).set_index('Time (s)')

    # main lignin-h groups as mass fraction, (-)
    ligh = pmligh[0]
    g1ligh = pmligh[1]
    g2ligh = pmligh[2]
    g3ligh = pmligh[3]
    g4ligh = pmligh[4]
    g5ligh = pmligh[5]
    fe2macr1 = pmligh[6]
    tligh = ligh + g1ligh + g2ligh + g3ligh + g4ligh + g5ligh + fe2macr1 # lignin-h

    ligh_main = {'Time (s)': t, 'ligh': ligh, 'g1ligh': g1ligh, 'g2ligh': g2ligh, 'g3ligh': g3ligh, 'g4ligh': g4ligh, 'g5ligh': 
    g5ligh, 'fe2marc1': fe2macr1, 'tligh': tligh}
    df_ligh = pd.DataFrame(data=ligh_main).set_index('Time (s)')

    # main lignin-o groups as mass fraction, (-)
    ligo = pmligo[0]
    g1ligo = pmligo[1]
    g2ligo = pmligo[2]
    g3ligo = pmligo[3]
    g4ligo = pmligo[4]
    g5ligo = pmligo[5]
    fe2macr2 = pmligo[6]
    tligo = ligo + g1ligo + g2ligo + g3ligo + g4ligo + g5ligo + fe2macr2 # lignin-o
    ligo_main = {'Time (s)': t, 'ligo': ligo, 'g1ligo':  g1ligo, 'g2ligo': g2ligo , 'g3ligo': g3ligo , 'g4ligo': g4ligo , 'g5ligo': g5ligo , 'fe2macr2': fe2macr2, 'tligo': tligo}

    df_ligo = pd.DataFrame(data=ligo_main).set_index('Time (s)')

    # Gas, Tar, Char from Cellulose, Hemicellulose, Lignin Reactions
    # ------------------------------------------------------------------------------

    # chemical species as mass fraction, (-)
    co = pcell[0] + phemi[0] + pligc[0] + pligh[0] + pligo[0]       # CO
    co2 = pcell[1] + phemi[1] + pligc[1] + pligh[1] + pligo[1]      # CO2
    ch2o = pcell[2] + phemi[2] + pligc[2] + pligh[2] + pligo[2]     # CH2O
    hcooh = pcell[3] + phemi[3] + pligc[3] + pligh[3] + pligo[3]    # HCOOH
    ch3oh = pcell[4] + phemi[4] + pligc[4] + pligh[4] + pligo[4]    # CH3OH
    ch4 = pcell[5] + phemi[5] + pligc[5] + pligh[5] + pligo[5]      # CH4
    glyox = pcell[6] + phemi[6] + pligc[6] + pligh[6] + pligo[6]    # Glyox (C2H2O2)
    c2h4 = pcell[7] + phemi[7] + pligc[7] + pligh[7] + pligo[7]     # C2H4
    c2h4o = pcell[8] + phemi[8] + pligc[8] + pligh[8] + pligo[8]    # C2H4O
    haa = pcell[9] + phemi[9] + pligc[9] + pligh[9] + pligo[9]      # HAA (C2H4O2)
    c2h5oh = pcell[10] + phemi[10] + pligc[10] + pligh[10] + pligo[10]  # C2H5OH
    c3h6o = pcell[11] + phemi[11] + pligc[11] + pligh[11] + pligo[11]   # C3H6O
    xyl = pcell[12] + phemi[12] + pligc[12] + pligh[12] + pligo[12]     # Xylose (C5H10O5)
    c6h6o = pcell[13] + phemi[13] + pligc[13] + pligh[13] + pligo[13]   # C6H6O
    hmfu = pcell[14] + phemi[14] + pligc[14] + pligh[14] + pligo[14]    # HMFU (C6H6O3)
    lvg = pcell[15] + phemi[15] + pligc[15] + pligh[15] + pligo[15]     # LVG (C6H10O2)
    coum = pcell[16] + phemi[16] + pligc[16] + pligh[16] + pligo[16]    # p-Coumaryl (C9H10O2)
    fe2macr = pcell[17] + phemi[17] + pligc[17] + pligh[17] + pligo[17] # FE2MACR (C11H12O4)
    h2 = pcell[18] + phemi[18] + pligc[18] + pligh[18] + pligo[18]      # H2
    h2o = pcell[19] + phemi[19] + pligc[19] + pligh[19] + pligo[19]     # H2O
    char = pcell[20] + phemi[20] + pligc[20] + pligh[20] + pligo[20]    # Char
 
    # groups for gas and tar as mass fraction, (-)
    gas = co + co2 + ch4 + c2h4 + h2
    tar = ch2o + hcooh + ch3oh + glyox + c2h4o + haa + c2h5oh + c3h6o + xyl + c6h6o + hmfu + lvg + coum + fe2macr

    gas_products = {'Time (s)': t, 'co': co, 'co2': co2, 'ch4': ch4 , 'c2h4': c2h4, 'h2': h2, 'total': gas}
    df_gasprod = pd.DataFrame(data=gas_products).set_index('Time (s)')
   
    tar_products = {'Time (s)': t, 'ch2o': ch2o , 'hcooh': hcooh , 'ch3oh': ch3oh , 'glyox': glyox , 'c2h4o': c2h4o , 'haa': haa , 'c2h5oh': c2h5oh , 'c3h6o': c3h6o , 'xyl': xyl , 'c6h6o': c6h6o , 'hmfu': hmfu , 'lvg': lvg , 'coum': coum , 'fe2macr': fe2macr, 'total': tar}

    df_tarprod = pd.DataFrame(data=tar_products).set_index('Time (s)')

    # chemical species mass balance as mass fraction, (-)
    total = co + co2 + ch2o + hcooh + ch3oh + ch4 + glyox + c2h4 + c2h4o + haa + c2h5oh + c3h6o + xyl + c6h6o + hmfu + lvg + coum + fe2macr + h2 + h2o + char

    total_prod = {'Time (s)': t, 'Wood': cell+hemi+ligc+ligh+ligo, 'Gas': gas, 'Tar': tar, 'Char': char}

    df_total = pd.DataFrame(data=total_prod).set_index('Time (s)')

    return df_cell, df_hemi, df_ligc, df_ligh, df_ligo, df_gasprod, df_tarprod, df_total
