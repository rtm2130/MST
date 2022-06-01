#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model,backend
from functools import reduce
import pandas as pd
import numpy as np
import tensorboard
import datetime


p_features = ["TRAIN_AV", "SM_AV", "CAR_AV",
              "TRAIN_TT", "SM_TT", "CAR_TT",
              "TRAIN_CO", "SM_CO", "CAR_CO",
              "TRAIN_HE", "SM_HE", "CAR_HE"]

# size of mini-batch in gradient computation
batch_size = 32
total = 3
# m = 4
p_features = 4
c_features = 72
scores_np = np.zeros((10,4,5))

for i in range(10):
    x_train = np.load(f'data/X_long_{i}.npy')
    p_train = np.load(f'data/P_long_{i}.npy')
    p_train = p_train.reshape((p_train.shape[0],p_features,total),order = 'C')
    y_train= np.load(f'data/Y_long_{i}.npy')
    
    x_val = np.load(f'data/XV_long_{i}.npy')
    p_val = np.load(f'data/PV_long_{i}.npy')
    p_val = p_val.reshape((p_val.shape[0],p_features,total),order = 'C')
    y_val = np.load(f'data/YV_long_{i}.npy')
    
    x_test = np.load(f'data/XT_long_{i}.npy')
    p_test = np.load(f'data/PT_long_{i}.npy')
    p_test = p_test.reshape((p_test.shape[0],p_features,total),order = 'C')
    y_test = np.load(f'data/YT_long_{i}.npy')
    
    X1 = np.concatenate( [x_train.reshape((x_train.shape[0],c_features,1)) for u in range(total)],axis = 2)
    x_train= np.concatenate([p_train,X1], axis = 1)
    
    X1 = np.concatenate( [x_val.reshape((x_val.shape[0],c_features,1)) for u in range(total)],axis = 2)
    x_val= np.concatenate([p_val,X1], axis = 1)
    
    X1 = np.concatenate( [x_test.reshape((x_test.shape[0],c_features,1)) for u in range(total)],axis = 2)
    x_test= np.concatenate([p_test,X1], axis = 1)
       
    
    x_train = x_train.astype(float)
    x_val = x_val.astype(float)
    x_test = x_test.astype(float)
        
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    

    # creates the data batch generation object 
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    
         
    # specifies the model class       
    class TasteNetv2(Model):
      def __init__(self,params):
        super(TasteNetv2, self).__init__()
        self.params = params
        regularizer = tf.keras.regularizers.L2(0.)
        #coefficients of NN for unobserved variables
    
        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0001, stddev=0.0002)
        self.d1 = [[Dense(params['hidden_eps'], activation='elu',
                          kernel_regularizer=regularizer,
                          kernel_initializer=initializer) for l in range(params['depth_eps'])]
                   for t in range(params['T'])] 
        #last layer for unobserved variables NN (no activation)
        self.d2 = [Dense(params['dimension_eps'],kernel_initializer=initializer,
                         kernel_regularizer=regularizer) for t in range(params['T'])]
    
        #coefficients of NN for utility function
        self.u1 = [Dense(params['hidden_utility'], activation='elu',kernel_initializer=initializer,
                         kernel_regularizer=regularizer) 
                       for l in range(params['depth_utility'])]        
        #last layer for utility function NN (no activation)        
        self.u2 = Dense(1,kernel_regularizer=regularizer, kernel_initializer=initializer)
    
      def call(self, x):
        y = []
        # loop over the different samples    
        for t in range(self.params['T']):
            # generates the epsilon unobservable variables
            list_eps = [tf.reshape(self.d2[t](reduce(lambda x,y: y(x),self.d1[t],x[:,-c_features:,i])),
                                   [-1,self.params['dimension_eps'],1]) 
                        for i in range(total)]    
            # formats the epsilon unobservable variables        
            eps = tf.transpose(tf.stack(list_eps, axis = 1)[:,:,:,0], [0,2,1])
            # computes the scalar between taste vector and observables
            x1 = tf.reshape(tf.math.reduce_sum(tf.keras.layers.Multiply()([x[:,:p_features,:],eps]),axis = 1),[-1,1,total])
            # computes the utility
            list_x1 = [tf.reshape(self.u2(reduce(lambda x,y: y(x),self.u1,tf.concat([x[:,:,i],x1[:,:,i]],1))),
                                  [-1,1,1]) 
                       for i in range(total)]
            # formats the utility        
            x1 = tf.stack(list_x1, axis = 1)[:,:,0,0]
            y.append(x1)  
        
        x = tf.math.reduce_sum(tf.stack([(1.0/(1.03*self.params['T']))*(0.01+tf.nn.softmax(y[t])) 
                                         for t in range(self.params['T'])],axis = 1), axis =1)
        return(x) 
    
    
    
    # # Select the parameter configuration
    params_list= [  
    # taste_net
    {
              'depth_eps': 3, # number of hidden layers in generating the unobserved variables
              'depth_utility': 0, # number of hidden layers in generating the utility function
              'dimension_eps':4, # dimension of unobserved variables
              'hidden_eps':15, # width of hidden layers generating the unobserved variables
              'hidden_utility':0, # width of hidden layers generating the utility function
              'T':1, # number of samples (heterogeneity)
              'name': 'taste_net1'
              }
    ]

    
    for p in params_list:
        loss_val = 10000
        test_loss_f = []
        loss_train = 0
        for r in [0.001,0.0001]:
            print("\n",p['name'],r)
            # Create an instance of the model
            model = TasteNetv2(p)          
            
            
            # designs the estimation process
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)            
            optimizer = tf.keras.optimizers.Adam(learning_rate=r)
       
            # picks the metrics
            train_loss = tf.keras.metrics.Mean(name='train_loss')
            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            train_mse =tf.keras.metrics.MeanSquaredError('train_mse')
            
            test_loss = tf.keras.metrics.Mean(name='test_loss')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
            test_mse =tf.keras.metrics.MeanSquaredError('test_mse')
            
            val_loss = tf.keras.metrics.Mean(name='val_loss')
            val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
            val_mse =tf.keras.metrics.MeanSquaredError('val_mse')               
            
            # computes the gradient function
            @tf.function
            def train_step(images, labels):
              with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
                regularization_loss=tf.add_n(model.losses)
                loss2 = tf.math.add(loss,regularization_loss)                                
                
              gradients = tape.gradient(loss2, model.trainable_variables)
              optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
              train_loss(loss)
              train_accuracy(labels, predictions)
              train_mse(tf.one_hot(labels,3), predictions,sample_weight = 1.0/batch_size)             
            
            # computes the test function
            @tf.function
            def test_step(images, labels):
              predictions = model(images, training=False)
              t_loss = loss_object(labels, predictions)
            
              test_loss(t_loss)
              test_accuracy(labels, predictions)
              test_mse(tf.one_hot(labels,3), predictions,sample_weight = 1.0/batch_size) 
              
            # computes the validation function
            @tf.function
            def val_step(images, labels):
              predictions = model(images, training=False)
              v_loss = loss_object(labels, predictions)
            
              val_loss(v_loss)
              val_accuracy(labels, predictions)  
              val_mse(tf.one_hot(labels,3), predictions,sample_weight = 1.0/batch_size)               
            
            # Training loop
            EPOCHS = 1001
            current_time = p['name'] +str(r)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            
            for epoch in range(EPOCHS):
              try:
                  # Reset the metrics at the start of the next epoch
                  train_loss.reset_states()
                  train_accuracy.reset_states()
                
                  del train_ds
                  del test_ds
                  train_ds = tf.data.Dataset.from_tensor_slices(
                        (x_train, y_train)).shuffle(1000+epoch).batch(batch_size)
                  test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(1000+epoch).batch(batch_size)
                  
                  for images, labels in train_ds:
                    train_step(images, labels)
                  with train_summary_writer.as_default():
                         tf.summary.scalar('loss', train_loss.result(), step=epoch)
                         tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                         tf.summary.scalar('mse', train_mse.result(), step=epoch)                                                     
                         
                  if (epoch % 10) == 0:
                      test_loss.reset_states()
                      test_accuracy.reset_states()
                      test_mse.reset_states()                         
                      val_loss.reset_states()
                      val_accuracy.reset_states()
                      val_mse.reset_states()                       
                      
                  for val_images, val_labels in val_ds:
                     val_step(val_images, val_labels)
                  with test_summary_writer.as_default():
                     tf.summary.scalar('loss', val_loss.result(), step=epoch)
                     tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch) 
                     tf.summary.scalar('mse', val_mse.result(), step=epoch)                     
                  for test_images, test_labels in test_ds:
                     test_step(test_images, test_labels)                          
            
              except:
                  print("Error")
                  raise 
              if (epoch % 1000) == 0:
                  print(
                    f'Epoch {epoch }, '
                    f'Loss: {train_loss.result()}, '
                    f'Accuracy: {train_accuracy.result() * 100}, '
                    f'Validation Loss: {val_loss.result()}, '                      
                    f'Test Loss: {test_loss.result()}, '
                    f'Test Accuracy: {test_accuracy.result() * 100},'
                    f'Test Mse: {test_mse.result() * 100},'                                 
                    f'Number of parameters: {np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])}'
                  )
               
              if  (epoch % 10) == 0 and (float(val_loss.result()) <  loss_val):
                      test_loss_f = [float(test_loss.result()),float(test_accuracy.result()),float(test_mse.result())*3]
                      loss_val = float(val_loss.result())
                      loss_train = float(train_loss.result())
            print("Test loss final:",p['name'],r, test_loss_f)
            print("Train/validation loss final:",p['name'],r,loss_train,loss_val)
        
        scores_np[i,params_list.index(p),0] = test_loss_f[0]
        scores_np[i,params_list.index(p),1] = test_loss_f[1]
        scores_np[i,params_list.index(p),2] = test_loss_f[2]        
        scores_np[i,params_list.index(p),3] = loss_val
        scores_np[i,params_list.index(p),4] = loss_train        
    
    
    
