#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:50:46 2021

Script to plot the pareto curves 

"""

import pandas as pd
import numpy as np
import seaborn as sns  


MST = np.load("outputs/varying_cmt_d15_m50_mod0.npy")
cMNL = np.load("outputs/varying_mnlkm_k30_5_10_mod0.npy")   
   
dfMST = pd.DataFrame({'# Segments':MST.mean(axis = 0)[0,4,:], 
                      'In-sample NLL':-MST.mean(axis = 0)[0,2,:], 
                      'Out-of-sample NLL':-MST.mean(axis = 0)[0,0,:],
                      'In-sample MSE':MST.mean(axis = 0)[0,3,:], 
                      'Out-of-sample MSE': MST.mean(axis = 0)[0,1,:]})
    
dfcMNL = pd.DataFrame({'# Segments':cMNL.mean(axis = 0)[0,4,:], 
                      'In-sample NLL':cMNL.mean(axis = 0)[0,2,:], 
                      'Out-of-sample NLL':-cMNL.mean(axis = 0)[0,0,:],
                      'In-sample MSE':cMNL.mean(axis = 0)[0,3,:], 
                      'Out-of-sample MSE': cMNL.mean(axis = 0)[0,1,:]})    
dfcMNL['Model'] = "MNLKM"
dfMST['Model'] = "CMT"

dfcMNL = dfcMNL[dfcMNL["# Segments"]<=110]
                       
df = pd.concat([dfMST,dfcMNL],axis = 0)

sns.relplot(x="# Segments", y="Out-of-sample MSE", hue="Model", data=df)
sns.relplot(x="# Segments", y="In-sample MSE", hue="Model", data=df)   
sns.relplot(x="# Segments", y="Out-of-sample NLL", hue="Model", data=df)
sns.relplot(x="# Segments", y="In-sample NLL", hue="Model", data=df)               
            
            