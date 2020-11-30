# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 19:51:30 2020

@author: ashis
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def calc_RAE(truth,pred):
    
    RAE = np.mean(np.abs(truth-pred))/np.max(np.abs(truth))
    
    return RAE

###################USER INPUT############################
nt = 300                  #FOM Time
dt = 0.07                 #Time-step
x0 = 0                    #Domain Start
xf = 250                  #Domain End
nx = 256                  #Spatial discretization


#Output Directory
outDir = 'Models' 

#Visualisations Directory
visDir = 'Visualisations'

#Time-inspances to plot at
timPlot = [50,100,150,200]

#Labels
labs = ['Truth', 'Dec Jac (Galerkin)', 'Dec Jac(LSPG)', 'Enc Jac(Galerkin)', 'Enc Jac(LSPG)', 'Enc Jac+DIEM']

#Line-Styles
linColor = ['k', 'aquamarine', 'violet', 'lawngreen', 'maroon', '--c']
lineStyl = ['solid', 'dotted', 'dashed', 'dotted', 'dashed', 'dotted']
###########################################################



x = np.linspace(x0,xf,nx)

#Loading Solutions
solFOM = np.load(os.path.join(outDir, 'solFOM.npy'))
solROM_dec = np.load(os.path.join(outDir, 'solROM_decJac.npy'))
solROM_enc = np.load(os.path.join(outDir, 'solROM_decJac_LSPG.npy'))
solROM_DIEM = np.load(os.path.join(outDir, 'solROM_encJac.npy'))
solROM_rand = np.load(os.path.join(outDir, 'solROM_encJac_LSPG.npy'))
#solROM_encDIEM = np.load(os.path.join(outDir, 'solROM_encJac_DIEM.npy'))

#Making output directory 
if not os.path.isdir(visDir): os.mkdir(visDir)


#Field Plots
print('Generating Field Plots')
c = 0

for t in timPlot:
    
    if c==0:
        f,axs = plt.subplots(1)    
        axs.plot(x, solFOM[t,:], color = linColor[0], linestyle = lineStyl[0], label=labs[0])
        axs.plot(x, solROM_dec[t,:], color = linColor[1], linestyle = lineStyl[1], label=labs[1])
        axs.plot(x, solROM_enc[t,:], color = linColor[2], linestyle = lineStyl[2], label=labs[2])
        axs.plot(x, solROM_DIEM[t,:], color = linColor[3], linestyle = lineStyl[3], label=labs[3])
        axs.plot(x, solROM_rand[t,:], color = linColor[4], linestyle = lineStyl[4], label=labs[4])
#        axs.plot(x, solROM_encDIEM[t,:], color = linColor[5], linestyle = lineStyl[5], label=labs[5])
        
    else:
        axs.plot(x, solFOM[t,:], color = linColor[0], linestyle = lineStyl[0])
        axs.plot(x, solROM_dec[t,:], color = linColor[1], linestyle = lineStyl[1])
        axs.plot(x, solROM_enc[t,:], color = linColor[2], linestyle = lineStyl[2])
        axs.plot(x, solROM_DIEM[t,:], color = linColor[3], linestyle = lineStyl[3])
        axs.plot(x, solROM_rand[t,:], color = linColor[4], linestyle = lineStyl[4])
#        axs.plot(x, solROM_encDIEM[t,:], color = linColor[5], linestyle = lineStyl[5])
        
    #axs.set_ylim([0,5])
    #axs.legend()
    c += 1
    #plt.show()
    #f.savefig(os.path.join(visDir, 'fig_'+str(t)+'.png'))

axs.set_ylim([0,5])
axs.legend(loc='upper right')    
plt.show
f.savefig(os.path.join(visDir, 'fig_fieldCombin''.png'))


#Error Plots
print('Generating Error Plots')

t = np.linspace(0, dt * nt, nt)

RAE_dec = np.zeros(nt)
RAE_enc = np.zeros(nt)
RAE_DIEM = np.zeros(nt)
RAE_rand = np.zeros(nt)
RAE_encDIEM = np.zeros(nt)


for i in range(nt):
    
    RAE_dec[i] = calc_RAE(solFOM[i,:], solROM_dec[i,:])
    RAE_enc[i] = calc_RAE(solFOM[i,:], solROM_enc[i,:])
    RAE_DIEM[i] = calc_RAE(solFOM[i,:], solROM_DIEM[i,:])
    RAE_rand[i] = calc_RAE(solFOM[i,:], solROM_rand[i,:])
#    RAE_encDIEM[i] = calc_RAE(solFOM[i,:], solROM_encDIEM[i,:])


f, axs = plt.subplots(1)
axs.semilogy(t, RAE_dec, color = linColor[1], linestyle = lineStyl[1], label=labs[1])
axs.semilogy(t, RAE_enc, color = linColor[2], linestyle = lineStyl[2], label=labs[2])
axs.semilogy(t, RAE_DIEM, color = linColor[3], linestyle = lineStyl[3], label=labs[3])
axs.semilogy(t, RAE_rand, color = linColor[4], linestyle = lineStyl[4], label=labs[4])
#axs.semilogy(t, RAE_encDIEM, color = linColor[5], linestyle = lineStyl[5], label=labs[5])

axs.legend()
plt.show()
f.savefig(os.path.join(visDir, 'Error.png'))

    







