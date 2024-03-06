#executes descriptive statistics and tsne 
#note that GoF test is part of utility.py, since it uses preprocessed data
#tSNE and descriptive statistics use the raw data

import pandas as pd
import numpy as np 
import os
from utils import preprocess,metrics,models
import pickle


def descriptive_statistics(real_df:pd.DataFrame,syn_df:pd.DataFrame,result_path:str):
    """
    Outputs descriptive statistics. Tables for static, heatmaps for dynamic data.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe. 
    result_path: Path to which we save results.
    returns: None, saves results to result directory.
    """

    static_features = ['age','gender','deceased','race']
    #get static feature dataframes
    real_df_static = preprocess.get_static(data=real_df,columns=static_features)
    syn_df_static = preprocess.get_static(data=syn_df,columns=static_features)

    # get descriptive statistics for static numerical variables
    real_stats = metrics.descr_stats(data=real_df_static[['age']])
    syn_stats = metrics.descr_stats(data=syn_df_static[['age']])
    filename = 'descr_stats_staticnumerical.csv'
    pd.concat([real_stats,syn_stats],axis=1).to_csv(os.path.join(result_path,filename))

    # #get relative frequencies for static categorical variables
    real_rel_freq = metrics.relative_freq(data=real_df_static[['gender','deceased','race']])
    syn_rel_freq = metrics.relative_freq(data=syn_df_static[['gender','deceased','race']])
    filename = 'descr_stats_staticcategorical.csv'
    pd.concat([real_rel_freq,syn_rel_freq],axis=1).to_csv(os.path.join(result_path,filename))

    #get matrix of relative frequencies at each step
    real_freqmatrix = metrics.rel_freq_matrix(data=real_df,columns='icd_code')
    syn_freqmatrix = metrics.rel_freq_matrix(data=syn_df,columns='icd_code')

    #plot frequencies as a heatmap
    for matrix,name in zip([real_freqmatrix,syn_freqmatrix],['Real','Synthetic']):
        plot = metrics.freq_matrix_plot(matrix,range=(0,.8))
        plot.title(f'{name} ICD chapter frequencies')
        filename = f'{name}_matrixplot.png'
        plot.savefig(os.path.join(result_path,filename))
        plot.show()
    


def steps(real_df:pd.DataFrame,syn_df:pd.DataFrame,subject_idx:str='subject_id'):
    """
    Outputs a plot of the amount of sequential steps across samples, to check if we accurately capture this in synthetic data.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe. 
    subject_idx: Subject identifier.
    returns: Plot of amount of steps per sample (real vs. synthetic).
    """

    r_steps = real_df.groupby(subject_idx).seq_num.max()
    s_steps = syn_df.groupby(subject_idx).seq_num.max()
    steps_plot = metrics.plot_max_steps(r_steps,s_steps)
    steps_plot.title('Maximum #steps per sample')
    return steps_plot




#executes the tsne step
def tsne_plot(real_df:pd.DataFrame,syn_df:pd.DataFrame,result_path:str):
    """
    Generates tSNE embeddings and creates a plot.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe.
    result_path: Path to which we save (intermediary) results.
    returns: tSNE plot.

    """

    #split data into static and dynamic
    df = pd.concat([real_df,syn_df],axis=0)
    static = preprocess.get_static(df,['age','gender','deceased','race'])
    static['age']= static['age'].astype(float)
    df['icd_code'],_ = pd.factorize(df['icd_code'])
    seq = preprocess.df_to_3d(df,cols=['icd_code'],padding=-1)

    # #find distance matrices
    static_distances = models.static_gower_matrix(static,cat_features=[False,True,True,True])
    dynamic_distances = models.dyn_gower_matrix(seq)

    # # take weighted sum of static and timevarying distances
    distance_matrix = ((len(static.columns))/len(df.columns))*static_distances + \
        (seq.shape[2]/len(df.columns))*dynamic_distances
    filename = 'distance_matrix.csv'
    with open(os.path.join(result_path,filename),'wb') as f:
        pickle.dump(distance_matrix,f)

    # #compute and plot tsne projections with synthetic/real labels as colors
    labels = np.concatenate((np.zeros(real_df.subject_id.nunique()),
                           np.ones(syn_df.subject_id.nunique())),axis=0)
    tsne_plot = models.tsne(distance_matrix,labels)
    #tsne_plot.title('tSNE plot of synthetic/real samples')
    return tsne_plot
    

if __name__ == '__main__':
    #set the result path
    syn_model = 'cpar'
    result_path = os.path.join('results',syn_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #load real and synthetic data
    #load_path = 'data'
    load_path = 'C:/Users/jlachterberg/Documents/data'
    real_file = os.path.join(load_path,'real.csv.gz')
    syn_file = os.path.join(load_path,f'{syn_model}.csv.gz')
    cols = ['subject_id','seq_num','icd_code','gender','age','deceased','race']
    real_df = pd.read_csv(real_file,sep=',',compression='gzip',usecols=cols)
    syn_df = pd.read_csv(syn_file,sep=',',compression='gzip',usecols=cols)

    #select only k subjects to test code quickly
    # k = 60
    # syn_df = syn_df[syn_df.subject_id.isin(np.random.choice(syn_df.subject_id.unique(),k))]
    # real_df = real_df[real_df.subject_id.isin(np.random.choice(real_df.subject_id.unique(),k))]
    
    descriptive_statistics(real_df,syn_df,result_path)

    steps_plot = steps(real_df,syn_df)
    filename =  'step_plot.png'
    steps_plot.savefig(os.path.join(result_path,filename))
    steps_plot.show()

    tsne_plot_ = tsne_plot(real_df,syn_df,result_path)
    filename = 'tsne.png'
    tsne_plot_.savefig(os.path.join(result_path,filename))
    tsne_plot_.show()
        
    