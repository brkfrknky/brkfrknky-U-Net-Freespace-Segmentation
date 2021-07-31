# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:38:42 2021

@author: berke
"""
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker



def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
    plt.savefig("../data/graphs/" + "8000.png")
    plt.show()
    
    
