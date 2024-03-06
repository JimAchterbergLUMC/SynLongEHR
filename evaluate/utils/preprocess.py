import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



#numerically encode categoricals, one hot if necessary, separate static and sequential data
def preprocess_(real_df,syn_df):
        n_real = real_df.subject_id.nunique()
        #pool samples for encoding
        df = pd.concat([real_df,syn_df],axis=0)
        #encode categoricals
        for col in ['gender','deceased','race','icd_code']:
            df[col],_ = pd.factorize(df[col])
        #one hot encode
        for col in ['race','icd_code']:
            dummies = pd.get_dummies(df[col],prefix=col)
            df = pd.concat([df,dummies],axis=1)
            df = df.drop(col,axis=1)
        #separate static and sequential
        stat_columns = [x for x in df.columns if 'race' in x or x in ['age','gender','deceased']]
        dyn_columns = [x for x in df.columns if 'icd_code' in x]
        static = get_static(df,stat_columns)
        seq = df_to_3d(df,dyn_columns,padding=0)
        #unpool samples
        real = [static[:n_real],seq[:n_real]]
        syn = [static[n_real:],seq[n_real:]]
        real[0] = real[0].reset_index(drop=True)
        syn[0] = syn[0].reset_index(drop=True)
        return real,syn

def k_test_splits(real:list,syn:list,k:int=10):
        """
        Creates k test splits stratified on mortality, for real and synthetic data individually.

        real: Real data, list of pandas dataframe and 3D numpy array (static and sequential data)
        syn: Synthetic data, list of pandas dataframe and 3D numpy array (static and sequential data)
        returns: List of k lists of test split indices for both real and synthetic data
        """
        #find full data indices to select splits from
        idx_real = np.arange(0,real[0].shape[0]) 
        idx_syn = np.arange(0,syn[0].shape[0])
        real_deceased_ratio = real[0]['deceased'].sum()/real[0].shape[0]
        syn_deceased_ratio = syn[0]['deceased'].sum()/syn[0].shape[0]
        #set seed for random splitting
        np.random.seed(0)
        #create test splits by sampling deceased and non-deceased patients according to their ratio in original data
        test_split_real_deceased = [np.random.choice(idx_real[real[0]['deceased']==1],size=int(((1/k)*real[0].shape[0])*real_deceased_ratio),replace=False) for _ in range(k)]
        test_split_real = [np.random.choice(idx_real[real[0]['deceased']==0],size=int(((1/k)*real[0].shape[0])*(1-real_deceased_ratio)),replace=False) for _ in range(k)]
        test_split_syn_deceased = [np.random.choice(idx_syn[syn[0]['deceased']==1],size=int(((1/k)*syn[0].shape[0])*syn_deceased_ratio),replace=False) for _ in range(k)]
        test_split_syn = [np.random.choice(idx_syn[syn[0]['deceased']==0],size=int(((1/k)*syn[0].shape[0])*(1-syn_deceased_ratio)),replace=False) for _ in range(k)]
        test_split_real = [np.concatenate((arr_A, arr_B)) for arr_A, arr_B in zip(test_split_real, test_split_real_deceased)]
        test_split_syn = [np.concatenate((arr_A, arr_B)) for arr_A, arr_B in zip(test_split_syn, test_split_syn_deceased)]
        return test_split_real,test_split_syn


def scale(real:list,syn:list,tr_real:list,te_real:list,tr_syn:list,te_syn:list,feature:list=['age']):
        """
        Scales numerical features to [0,1] for each train/test set separately.
        Additionally, turns all data into float numpy arrays to input into ML models.

        real: Real data, list of pandas dataframe and 3D numpy array (static and sequential data).
        syn: Synthetic data, list of pandas dataframe and 3D numpy array (static and sequential data).
        tr_real: List of training indices of real data.
        te_real: List of test indices of real data.
        tr_syn: List of training indices of synthetic data.
        te_syn: List of test indices of synthetic data.
        feature: List of numerical features to scale.
        returns: List of real and synthetic data, ordered from train to test (2D and 3D float numpy array).
        """
        data_ = []
        for data,splits in zip([real,syn],[[tr_real,te_real],[tr_syn,te_syn]]):
            tr,te = splits 
            for i in [tr,te]:
                stat = data[0].loc[i]
                stat[feature] = Scaler().transform(stat[feature])
                stat = stat.to_numpy().astype(float)
                seq = data[1][i]
                seq = seq.astype(float)
                data_.append([stat,seq])
        return data_

#get static attribute table in which values are not repeated
def get_static(data,columns,subject_idx='subject_id'):
    return data.groupby(subject_idx)[columns].first()

#get 2d timevarying data to 3d numpy array (necessary when data is multi-column)
def df_to_3d(df,cols,subject_idx='subject_id',timestep_idx='seq_num',padding='-1',pad_to=None):
    #check if we pad to prespecified number of timesteps
    if pad_to == None:
        t = max(df[timestep_idx])
    else: 
        t = pad_to
    #check if we need an object np array or float
    if type(padding)==str:
        dtype=object
    else:
        dtype=int
    seq = np.full((df[subject_idx].nunique(),t,len(cols)),padding,dtype=dtype)
    for idx,(_,subject) in enumerate(df.groupby(subject_idx)[cols]):
        seq[idx,:subject.shape[0],:] = subject
    return seq



class Scaler():
    def __init__(self,method='zero-one'):
        super().__init__()
        self.method = method

    def transform(self,x):
        if self.method=='zero-one':
            self.min = x.min().values
            self.max = x.max().values
            return (x-self.min)/(self.max-self.min)
        
    def reverse_transform(self,x):
        if self.method=='zero-one':
            return x*(self.max-self.min) + self.min
        
    

def train_split(X,y,stratify=None,train_size=.7):
    #create train test split stratified on syn/real labels
    return train_test_split(X,y,stratify=stratify,train_size=train_size)



def trajectory_input_output(x,max_t):
    static = x[0]
    seq = x[1]
    timesteps = np.max(np.where(np.any(seq!=0,axis=2),np.arange(seq.shape[1]),-1),axis=1)
    seqs = []
    stat = []
    y = []
    for t in range(1,max_t-1):
        #filter on rows which have max timesteps>=t+1
        x_seq = seq[timesteps>=t+1]
        x_stat = static[timesteps>=t+1]
        #input data is data up until t=t from seq (padded to end) and the corresponding static data
        stat.append(x_stat)
        x_seq = x_seq[:,:t,:]
        pad_size = max_t-x_seq.shape[1]
        x_seq = np.pad(x_seq,((0,0),(0,pad_size),(0,0)),'constant',constant_values=0)
        seqs.append(x_seq)
        #output data is seq at t=t+1
        y.append(seq[timesteps>=t+1][:,t+1,:])
    stat = np.concatenate(stat)
    seqs = np.concatenate(seqs)
    x = [stat,seqs]
    y = np.concatenate(y)
    return x,y