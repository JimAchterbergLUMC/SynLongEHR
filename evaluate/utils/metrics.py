from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error,accuracy_score,roc_auc_score
from matplotlib import pyplot as plt 
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp

def mape(true,pred):
    return mean_absolute_percentage_error(true,pred)

def mae(true,pred):
    return mean_absolute_error(true,pred)

def accuracy(true,pred):
    return accuracy_score(true,pred)

def auc(labels,pred_scores):
    return roc_auc_score(labels,pred_scores)

def ks_test(real_pred,syn_pred):
    return ks_2samp(data1=real_pred.flatten(),data2=syn_pred.flatten(),alternative='two-sided')

def descr_stats(data):
    stats = ['mean','std','min','max']
    list_ = []
    list_.append(data.mean(axis=0))
    list_.append(data.std(axis=0))
    list_.append(data.min(axis=0))
    list_.append(data.max(axis=0))
    return pd.DataFrame(list_,index=stats,columns=data.columns)

def relative_freq(data):
    proportions = pd.concat([data[col].value_counts(normalize=True) for col in data],axis=1)
    proportions.columns = data.columns
    return proportions

def rel_freq_matrix(data,columns,timestep_idx='seq_num'):
    rel_freq = data.groupby(timestep_idx)[columns].value_counts(normalize=True).rename('rel_freq').reset_index()
    rel_freq = rel_freq.pivot(index=columns,columns=timestep_idx,values='rel_freq')
    return rel_freq

def freq_matrix_plot(rel_freq,range=None):
    if range != None:
        vmin,vmax = range
    else:
        vmin = vmax = None
    plt.figure(figsize=(10, 6))
    sns.heatmap(rel_freq, annot=False, cmap='rocket_r', fmt=".2f", vmin=vmin, vmax=vmax)
    plt.xlabel('Step')
    plt.ylabel('Category')
    return plt

def GoF_kdeplot(pred,y_test):
    plt.figure(figsize=(10, 6))
    range = (0,1)
    plt.xlim(0,1)
    plt.ylim(0,6)
    sns.kdeplot(pred[y_test==0],palette=['blue'],clip=range,alpha=.5,fill=True,label='Real')
    sns.kdeplot(pred[y_test==1],palette=['red'],clip=range,alpha=.5,fill=True,label='Synthetic')
    plt.xlabel('Classification score')
    plt.ylabel('Density')
    plt.legend()
    return plt 

def plot_max_steps(r_tsteps,s_tsteps):
    _,bins,_ = plt.hist(s_tsteps,bins='auto',color='red',label='Synthetic',alpha=.5)
    plt.hist(r_tsteps,bins=bins,color='blue',label='Real',alpha=.5)
    plt.xlabel('Max steps')
    plt.ylabel('Frequency')
    plt.legend()
    return plt