#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:19:32 2020

"""
import pandas as pd
import numpy as np

path = 'data/'
df = pd.read_csv(path+'swissmetro.dat',sep='\t')

df["CAR_HE"] = 0
c_features = ["GROUP", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE",
              "MALE", "INCOME", "GA", "ORIGIN", "DEST"]
p_features = ["TRAIN_AV", "SM_AV", "CAR_AV",
              "TRAIN_TT", "SM_TT", "CAR_TT",
              "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]
target = "CHOICE"
df = df[df[target] > 0]
df.loc[:,target] = df.loc[:,target]-1
df = df.reset_index()
df = df.sample(frac=1).reset_index(drop=True)
already_dummies = ["FIRST", "MALE", "GA"]

df2 = df.copy()

c_features2  = c_features[:]

for c in c_features2:
    if c not in already_dummies:
        dummies = pd.get_dummies(df[c], prefix = c)
        df2 = pd.concat((df2,dummies), axis = 1)
        df2 = df2.drop(c, axis = 1)
        c_features2.remove(c)
        c_features2 = c_features2 + dummies.columns.tolist()

def prepare_data(df,train_indices,test_indices,validation_indices,c_features):
  '''
    Prepares a partition of the data into train, validation and test sets

  '''

  Y = df.loc[train_indices,target].values
  X = df.loc[train_indices,c_features].values
  P = df.loc[train_indices,p_features].values
  YT = df.loc[test_indices,target].values
  XT = df.loc[test_indices,c_features].values
  PT = df.loc[test_indices,p_features].values
  YV = df.loc[validation_indices,target].values
  XV = df.loc[validation_indices,c_features].values
  PV = df.loc[validation_indices,p_features].values

  return X,P,Y,XV,PV,YV,XT,PT,YT


n_valid = int(df.shape[0]*0.125);
n_test = int(df.shape[0]*0.125);

for i in range(10):
    n = df.shape[0]
    selected_indices = np.random.choice(range(n), size= n_valid + n_test, replace=False)
    test_indices = np.random.choice(selected_indices, size = n_test, replace=False)
    validation_indices = np.setdiff1d(selected_indices,test_indices)
    train_indices = np.setdiff1d(np.arange(n),selected_indices)

    X,P,Y,XV,PV,YV,XT,PT,YT = \
        prepare_data(df2, train_indices, test_indices,validation_indices,c_features2)

    np.save(path+'X_long_{}.npy'.format(i),X)
    np.save(path+'P_long_{}.npy'.format(i),P)
    np.save(path+'Y_long_{}.npy'.format(i),Y)

    np.save(path+'XV_long_{}.npy'.format(i),XV)
    np.save(path+'PV_long_{}.npy'.format(i),PV)
    np.save(path+'YV_long_{}.npy'.format(i),YV)

    np.save(path+'XT_long_{}.npy'.format(i),XT)
    np.save(path+'PT_long_{}.npy'.format(i),PT)
    np.save(path+'YT_long_{}.npy'.format(i),YT)

    X,P,Y,XV,PV,YV,XT,PT,YT = \
        prepare_data(df, train_indices, test_indices,validation_indices,c_features)

    np.save(path+'X{}.npy'.format(i),X)
    np.save(path+'P{}.npy'.format(i),P)
    np.save(path+'Y{}.npy'.format(i),Y)

    np.save(path+'XV{}.npy'.format(i),XV)
    np.save(path+'PV{}.npy'.format(i),PV)
    np.save(path+'YV{}.npy'.format(i),YV)

    np.save(path+'XT{}.npy'.format(i),XT)
    np.save(path+'PT{}.npy'.format(i),PT)
    np.save(path+'YT{}.npy'.format(i),YT)
