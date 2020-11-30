# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:12:20 2020

@author: ashis
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from classDefs import solutionPhysics, romSolution

###############################USER INPUT###################################################

paramsDictFOM = {
 
##FOM Paramters######## 
'nt': 300,
'dt': 0.07,
'x0': 0,
'xf': 100,
'nx': 256,
'Time-Scheme': 'Explicit' ,
'Number of Sub-Iterations': 20,
'Residual Tolerance': 1e-15,
 
##Parameters Space#####
'Parameters': [4.3, 0.021],


##Scale FOM###########
'Scale FOM': True,

##Output Directory####
'Output Directory': 'Models',

##Saving FOM##########
'Save FOM': True,
'Save RHS': True,

##DIEM################
'QDIEM Interpolation': True,
'FOM Path': 'solFOM_RHS.npy',
'Number of Sampling Points': 160
}


##ROM Parameters#######

paramsDictROM = {
 
##ROM Paramters######## 
'Calculate ROM': True,
'Encoder Jacobian Approximation': False,
'Models Path': 'Models',
'Centering Profile Path': 'centProf.npy',
'Subtracting Profile Path': 'normSubProf.npy',
'Dividing Profile Path': 'normFacProf.npy',
'Encoder Path': 'encoder_full.h5',
'Decoder Path': 'decoder_full.h5',
'Save ROM': True}


###############################ENF OF USER INPUT#############################################

calcROM = paramsDictROM['Calculate ROM']

#FOM
if not calcROM:

    print('Running FOM')
    
    #Initializing solution object
    sol = solutionPhysics(paramsDictFOM)
    
    for i in range(sol.nt - 1):
        
        print('Iteration : ' + str(i+1))
        sol.advanceSolution()
        sol.updateBoundary()
        sol.updateSnapshots(i+1)
    
    sol.postProcess()
    
#ROM    
else:
    
    print('Running ROM')


    #Initializing rom solution object
    rom = romSolution(paramsDictROM, paramsDictFOM)
    
    for i in range(rom.sol.nt - 1):
        
        print('Iteration : ' + str(i+1)) 
        rom.mapRHS()
        rom.advanceROMSol()
        rom.updateSnapshots(i+1)
        
    rom.postProcess()
    









