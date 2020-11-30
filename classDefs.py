# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 03:50:20 2020

@author: ashis
"""

import numpy as np
import os
from scipy.linalg import pinv, qr
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from numpy.linalg import svd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import ThresholdedReLU
import tensorflow as tf


class solutionPhysics:
    
    # Initializing full-order solution object
    def __init__(self, paramsDict):
        
        #FOM Parameters
        self.nt = paramsDict['nt']
        self.dt = paramsDict['dt']
        self.x0 = paramsDict['x0']
        self.xf = paramsDict['xf']
        self.nx = paramsDict['nx']
        self.x  = np.linspace(self.x0, self.xf, self.nx)
        self.dx = self.x[1] - self.x[0]
        self.timeScheme = str(paramsDict['Time-Scheme'])
        if self.timeScheme == 'Implicit':
            self.numSubIters = paramsDict['Number of Sub-Iterations']
            self.resTol = paramsDict['Residual Tolerance']
            
        #Parameter Space
        self.mu = paramsDict['Parameters']
        
        #FOM Scaling
        self.scaleFOM = paramsDict['Scale FOM']
        
        #Output Directory
        self.outDir = str(paramsDict['Output Directory'])
        
        #Saving FOM
        self.saveFOM = paramsDict['Save FOM']
        self.saveRHS = paramsDict['Save RHS']
        
        #DIEM
        self.calcDIEM = paramsDict['QDIEM Interpolation']
        self.numSamp = paramsDict['Number of Sampling Points']
        
        #Calculating Initial Condition
        self.u = np.ones(self.nx)
        self.u[0] = self.mu[0]
        
        #Ghost cells
        self.inlet = self.u[0]
        self.outlet = self.u[-1]
        
        #Flux at each volume
        self.RHS = np.zeros(self.nx)
        
        #Snapshot matrix
        self.solFOM = np.zeros((self.nt, self.nx))
        self.solRHS = np.zeros((self.nt - 1, self.nx))
        self.updateSnapshots(0)
        
        #Initializing QDIEM
        if self.calcDIEM:            
            try:
                self.solFOM_load = np.load(os.path.join(self.outDir, paramsDict['FOM Path']))
            
            except:
                raise ValueError('FOM/RHS Snapshots needed for DIEM!')    

            self.P, self.Pi = self.initDIEM(self.solFOM_load.copy())
        
        
    # Determining DIEM sampling point, basis
    def initDIEM(self, solFOM_load):
        
        #Re-Ordering Solution
        solFOM = np.squeeze(solFOM_load).T
    
        #SVD Decomposition
        U, s, VT = svd(solFOM)
        U = U[:,:self.numSamp]
        
        #QR Fac to determine sampling points
        Q, R, P = qr(U, pivoting=True)
        
        #Pre-computing oblique projection term
        Pi = U @ pinv(U[P,:])
        
        return P, Pi
    
    # Updating Ghost Cells
    def updateBoundary(self):
        
        self.inlet = self.u[0].copy()
        self.outlet = self.u[-1].copy()
        
    # Calculating full-order RHS using Godunov's scheme
    def calcRHS(self):
        
        uFull = np.zeros(self.nx + 2)
        uFull[0] = self.inlet.copy()
        uFull[1:self.nx+1] = self.u.copy()
        uFull[-1] = self.outlet.copy()        
        
        xFull = np.zeros(self.nx + 2)
        xFull[0] = self.x[0].copy() - self.dx
        xFull[1:self.nx+1] = self.x.copy()
        xFull[-1] = self.x[-1].copy()  + self.dx    
        
        #F_(j+1/2) using Godunov's scheme (all +ve values)
        fluxR = - 0.5 * (uFull[1:-1] ** 2) 
        
        #F_(j-1/2)
        fluxL = - 0.5 * (uFull[:-2] ** 2) 
        
        if not self.calcDIEM:    
            self.RHS = (1.0 / self.dx) * (fluxR - fluxL)
            self.RHS += + 0.02 * np.exp(self.mu[1] * xFull[1:-1])
        
        else:
            self.RHS = (1.0 / self.dx) * (fluxR - fluxL)
            xCurr = xFull[1:-1] 
            self.RHS += + 0.02 * np.exp(self.mu[1] * xCurr)
            self.RHS = self.RHS[self.P]
            self.RHS = self.Pi @ self.RHS

    # Advancing the full-order solution
    def advanceSolution(self):
        
        if self.timeScheme == 'Explicit':
            self.calcRHS()
            self.u += self.dt * self.RHS.copy()
        
        else:
            self.calcRHS()
            for i in range(self.numSubIters):
                res = self.advanceImplicit()
                print(res)
                # Residual Check
                if res <= self.resTol:
                    break

        #Enforcing Boundary Condition
        self.u[0] = self.mu[0]
        
    # Implicit formulation (Backward Euler)
    def advanceImplicit(self):
        
        rhsJacob = self.calcRHSJacob(self.u.copy())
        
        #Solving Linear System
        du = spsolve(rhsJacob, self.dt * self.RHS.copy())
        
        #Advancing Solution
        self.u += du
        
        #Calculating residue
        res = rhsJacob @ du - self.dt * self.RHS.copy()
        
        res = np.linalg.norm(res, ord=2)
        
        return res

    # RHS Jacobian for implicit
    def calcRHSJacob(self, u):
        
        dFluxdu = diags((-u / self.dx), 0) + diags((u[:-1] / self.dx), -1)
        I = diags((np.ones(u.shape[0])), 0)
        rhsJacob = (I - self.dt * dFluxdu)
        
        return rhsJacob
    

    # Updating Snapshot Matrix
    def updateSnapshots(self, index):
        
        self.solFOM[index,:] = self.u
        if self.saveRHS :
            self.solRHS[index - 1,:] = self.RHS

        
    # Saving snapshot array
    def saveFOMSnapshots(self):
        
        if not os.path.isdir(self.outDir): os.mkdir(self.outDir)

        if self.calcDIEM:
            np.save(os.path.join(self.outDir,'solFOM_DIEM.npy'), self.solFOM)
            
        else:
            np.save(os.path.join(self.outDir, 'solFOM.npy'), self.solFOM)
            
        if self.saveRHS:
            np.save(os.path.join(self.outDir,'solFOM_RHS.npy'), self.solRHS)
            
    # Data Post-processing for manifold training
    def scaleFOMSnapshots(self):

        if not os.path.isdir(self.outDir): os.mkdir(self.outDir)
        
        self.solFOM = np.expand_dims(self.solFOM, axis=2)
        self.solFOM = np.transpose(self.solFOM, axes=(1,2,0))
        
        #Extracting profiles
        centProf = self.solFOM[:,:,[0]]
        self.solFOM -= centProf
        
        onesProf = np.ones((self.solFOM.shape[0],self.solFOM.shape[1],1), dtype = np.float64)
        zeroProf = np.zeros((self.solFOM.shape[0],self.solFOM.shape[1],1), dtype = np.float64)

	    # normalize by  (X - min(X)) / (max(X) - min(X)) 
        minVals = np.amin(self.solFOM, axis=(0,2), keepdims=True)
        maxVals = np.amax(self.solFOM, axis=(0,2), keepdims=True)

        normSubProf = minVals * onesProf
        normFacProf = (maxVals - minVals) * onesProf

        self.solFOM = (self.solFOM - normSubProf) / (normFacProf)
        
        
        #Saving Profiles
        np.save(os.path.join(self.outDir, 'centProf.npy'), centProf)
        np.save(os.path.join(self.outDir, 'normSubProf.npy'), normSubProf)
        np.save(os.path.join(self.outDir, 'normFacProf.npy'), normFacProf)
        
        #Saving scaled solution
        np.save(os.path.join(self.outDir, 'solFOM_burgers_scaled.npy'), self.solFOM)
        
    
    def postProcess(self):
        
        if self.saveFOM:
            self.saveFOMSnapshots()
            
        if self.scaleFOM:
            self.scaleFOMSnapshots()
            
        
class romSolution:
    
    # Initializing ROM solution object
    def __init__(self, paramsDictROM, paramDictFOM):
        
        self.calcROM = paramsDictROM['Calculate ROM']
        self.calcEncJac = paramsDictROM['Encoder Jacobian Approximation']
        self.trueJacobs = paramsDictROM['True Jacobians']
        self.modPath = paramsDictROM['Models Path']
        self.centProfPath = paramsDictROM['Centering Profile Path']
        self.normSubPath = paramsDictROM['Subtracting Profile Path']
        self.normFacPath = paramsDictROM['Dividing Profile Path']
        self.encPath = paramsDictROM['Encoder Path']
        self.decPath = paramsDictROM['Decoder Path']
        self.saveROM = paramsDictROM['Save ROM']
        
        #Loading models and profiles
        self.centProf = np.load(os.path.join(self.modPath, self.centProfPath))
        self.normSubProf = np.load(os.path.join(self.modPath, self.normSubPath))
        self.normFacProf = np.load(os.path.join(self.modPath, self.normFacPath))
        self.encoder =  load_model(os.path.join(self.modPath, self.encPath))           
        self.decoder =  load_model(os.path.join(self.modPath, self.decPath)) #, custom_objects = {"ThresholdedReLU": ThresholdedReLU(theta=1e-3)})           
        
        #Initializing the full order solution
        self.sol = solutionPhysics(paramDictFOM)
        
        #Initializing the ROM
        self.initROM()     
        
        #Initializing ROM rhs
        self.rhsROM = np.zeros(self.code.shape)

        #Output File superscript
        self.romFileSup = ''
        if self.calcEncJac:
            self.romFileSup += '_encJac'
        else:
            self.romFileSup += '_decJac'
        
        if self.sol.calcDIEM:
            self.romFileSup += '_DIEM'

        
    def center(self, dataArr, deCenter = False):
        
        if deCenter:
            dataArr += np.squeeze(self.centProf)
            
        else:
            dataArr -= np.squeeze(self.centProf)
            
        return dataArr
    
    
    def normalize(self, dataArr, deNormalize = False):
        
        if deNormalize:
            dataArr = (dataArr * np.squeeze(self.normFacProf)) + np.squeeze(self.normSubProf)
            
        else:
            dataArr = (dataArr - np.squeeze(self.normSubProf)) / np.squeeze(self.normFacProf)
            
        return dataArr


    def standardize(self, dataArr, inverse = False):
        
        if inverse:
            dataArr = self.normalize(dataArr, deNormalize = True)
            dataArr = self.center(dataArr, deCenter = True)
            
        else:
            dataArr = self.center(dataArr)
            dataArr = self.normalize(dataArr)
            
        return dataArr
    
    
    # Order/Re-Order for network inference
    def order(self, u, inverse = False):
        
        if inverse:
            u = np.squeeze(u)
            
        else:
            u = np.expand_dims(u, axis=(0,2))
            
        return u

        
    # Encoding map
    def calcEncoding(self, u):
        
        u = self.standardize(u)
        u = self.order(u)
        code = self.encoder.predict(u)
        
        return code
        
    # Decoding map
    def calcDecoding(self, code):
        
        uTilde = self.decoder.predict(code)
        uTilde = self.order(uTilde, inverse=True)
        uTilde = self.standardize(uTilde, inverse = True)
        uTilde[0] = self.sol.mu[0]
        
        return uTilde

    # Analytical Decoder/Encoder Jacobian
    def calcAnalyticalTFJacobian(self, code, uTilde, dtype = np.float64):
    
    	if self.calcEncJac:
    		with tf.GradientTape() as g:
    			inputs = tf.Variable(uTilde, dtype=dtype)
    			outputs = self.encoder(inputs)
    
    		Jacob = np.squeeze(g.jacobian(outputs, inputs).numpy())
    		
    	else:
    		with tf.GradientTape() as g:
    			inputs = tf.Variable(code[None,:], dtype=dtype)
    			outputs = self.decoder(inputs)
    
    		# output of model is in CW order, Jacobian is thus CWK, reorder to WCK 
    		Jacob = np.squeeze(g.jacobian(outputs, inputs).numpy())
    
    	return Jacob
        
    # Initializing ROM solution
    def initROM(self):
        
        self.code = self.calcEncoding(self.sol.u.copy())
        self.sol.u = self.calcDecoding(self.code.copy())
        self.updateSnapshots(0)
         
    # Galerkin Projection of RHS to manifold tangent space
    def mapRHS(self):
        
        self.sol.calcRHS()
        
        if self.calcEncJac:            
            uTilde = self.standardize(self.sol.u.copy())
            uTilde = self.order(uTilde)
            encJac = self.calcAnalyticalTFJacobian(self.trueCodeCurr.copy(), self.trueVarCurr.copy())
            self.rhsROM = encJac @ self.sol.RHS
            
        else:    
            decJac = self.calcAnalyticalTFJacobian(self.trueCodeCurr.copy(), self.trueVarCurr.copy())                
            self.rhsROM = pinv(decJac) @ self.sol.RHS
        

    # Advancing solution in latent space
    def advanceROMSol(self):
        
        if self.sol.timeScheme == 'Explicit':
            self.code += self.sol.dt * self.rhsROM
            
        else:
            for i in range(self.sol.numSubIters):
                res = self.advanceImplicit()
                print(res)
                if res <= self.sol.resTol:
                    break
            
        self.sol.u = self.calcDecoding(self.code.copy())
        #Enforcing Boundary condition
        self.sol.u[0] = self.sol.mu[0]
        self.sol.updateBoundary()
    

    # Implicit formulation (Backward-Euler)
    def advanceImplicit(self):
        
        rhsJacob = self.calcRHSJacob(np.squeeze(self.code.copy()))
        
        #Solving Linear System
        dcode = spsolve(rhsJacob, self.sol.dt * self.rhsROM.copy())
        
        #Advancing Solution
        self.code += dcode
        
        #Calculating residue
        res = rhsJacob @ dcode - self.sol.dt * self.rhsROM.copy()
        
        res = np.linalg.norm(res, ord=2)
        
        return res
        
        
    def calcRHSJacob(self, u):
        
        dFluxdu = diags((-u / self.sol.dx), 0) + diags((u[:-1] / self.sol.dx), -1)
        I = diags((np.ones(u.shape[0])), 0)
        rhsJacob = (I - self.sol.dt * dFluxdu)
        
        return rhsJacob
 


    def updateSnapshots(self, index):
        
        self.sol.solFOM[index,:] = self.sol.u        


    def postProcess(self):
        
        if self.saveROM:
            np.save(os.path.join(self.modPath,'solROM'+self.romFileSup+'.npy'), self.sol.solFOM)


        