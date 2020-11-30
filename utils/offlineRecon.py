# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:37:49 2020

@author: ashis
"""

import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

#Loading the datasets

solFOM_scaled = np.load('Models/solFOM_burgers_scaled.npy')
centProf = np.load('Models/centProf.npy')
normFacProf = np.load('Models/normFacProf.npy')
normSubProf = np.load('Models/normSubProf.npy')

#Loading models
encoder = load_model('Models/encoder_full.h5')
decoder = load_model('Models/decoder_full.h5')


#Ordering
solFOM_scaled = np.transpose(solFOM_scaled,axes=(2,0,1))

#Offline reconstruction
recon = encoder.predict(solFOM_scaled)
recon = decoder.predict(recon)
recon = encoder.predict(recon)
recon = decoder.predict(recon)
recon = encoder.predict(recon)
recon = decoder.predict(recon)
recon = encoder.predict(recon)
recon = decoder.predict(recon)
recon = encoder.predict(recon)
recon = decoder.predict(recon)

#Re-Ordering
solFOM_scaled = np.transpose(solFOM_scaled,axes=(1,2,0))
recon = np.transpose(recon,axes=(1,2,0))

#Decscaling
recon = (recon * normFacProf) + normSubProf
recon += centProf

#Descalinn FOM
solFOM =  (solFOM_scaled * normFacProf) + normSubProf
solFOM += centProf




#Plotting

instances = [0,44,200,299]


for i in instances:
    
    plt.plot(solFOM[:,:,i])
    plt.plot(recon[:,:,i])
    plt.show()
    
    
#Reshaping and saving 
recon = np.transpose(recon, axes=(2,0,1))
recon = recon.reshape((recon.shape[0],recon.shape[1]))
np.save('Models/solROM_encJac_DIEM.npy', recon)
    

