import keras 
from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from gower import gower_matrix
from dtwParallel import dtw_functions
from scipy.spatial import distance

#base RNN without output layer
class RNN(keras.Model):
    def __init__(self,config):
        super().__init__()
        self.build_model(config)

    def build_model(self,config):
        input_attr = layers.Input(shape=config['input_shape_attr'])
        input_feat = layers.Input(shape=config['input_shape_feat'])
        #first layer consists of separate processing layers
        x_attr = layers.Dense(config['hidden_units'][0],activation=config['activation'])(input_attr)
        x_feat = layers.GRU(config['hidden_units'][0],activation=config['activation'])(input_feat)
        x = layers.Concatenate(axis=1)([x_attr,x_feat])
        #rest is just feedforward so can be done in a loop
        config['hidden_units'] = config['hidden_units'][1:]
        for units in config['hidden_units']:
            x = layers.Dropout(config['dropout_rate'])(x)
            x = layers.Dense(units,activation=config['activation'])(x)
        
        self.model = keras.Model(inputs=[input_attr,input_feat],outputs=x)

    def call(self, inputs):
        return self.model(inputs)
    

class GoF_RNN(keras.Model):
    def __init__(self,config):
        super().__init__()
        self.base_model = RNN(config)
        self.output_layer = layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.output_layer(x)
    
class mortality_RNN(keras.Model):
    def __init__(self,config):
        super().__init__()
        self.base_model = RNN(config)
        self.output_layer = layers.Dense(1,activation='sigmoid')

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.output_layer(x)
    
    
class trajectory_RNN(keras.Model):
    def __init__(self,config):
        super().__init__()
        self.base_model = RNN(config)
        self.output_layer = layers.Dense(config['output_units'],activation='softmax')

    def call(self, inputs):
        x = self.base_model(inputs)
        return self.output_layer(x)
    
class privacy_RNN(keras.Model):
    def __init__(self,config,labels):
        super().__init__()
        self.base_model = RNN(config)
        self.labels=labels
        self.race_size = sum(l.count('race') for l in self.labels)
        #initialize the output layers with names
        self.output_age = layers.Dense(1,activation='linear',name='output_1')
        self.output_gender = layers.Dense(1,activation='sigmoid',name='output_2')
        self.output_race = layers.Dense(self.race_size,activation='softmax',name='output_3')

    def call(self, inputs):
        x = self.base_model(inputs)
        #output layer is conditional on input labels
        output_layer = []
        if 'age' in self.labels:
            output_layer.append(self.output_age(x))
        if 'gender' in self.labels:
            output_layer.append(self.output_gender(x))
        if self.race_size>0:
            output_layer.append(self.output_race(x))
        return output_layer

    
class mortality_LR(LogisticRegression):
    def __init__(self, penalty='elasticnet', l1_ratio=0.5,):
        super().__init__(
            penalty=penalty,
            solver='saga',  # 'saga' solver supports both 'l1' and 'elasticnet' penalties
            l1_ratio=l1_ratio,
            max_iter=1000
        )
        
class mortality_RF(RandomForestClassifier):
    def __init__(self, n_estimators=100,max_depth=None):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    

#tSNE projection plot, color coded by label
def tsne(distance_matrix,labels):
    embeddings = TSNE(n_components=2,init='random',metric='precomputed').fit_transform(distance_matrix)
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings[:,0],embeddings[:,1],c=labels,cmap='bwr',alpha=.5)
    handles=[plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='blue', label='Real'),plt.Line2D([0], [0], marker='o', color='w', 
                    markerfacecolor='red', label='Synthetic')]
    plt.legend(handles=handles)
    return plt

#get gower matrix for 2d data
def static_gower_matrix(data,cat_features=None):
    return gower_matrix(data,cat_features=cat_features)

#get gower matrix for 3d sequences of mixed datatypes
def dyn_gower_matrix(data):
    class Input:
        def __init__(self):
            self.check_errors = False 
            self.type_dtw = "i"
            self.constrained_path_search = None
            self.MTS = True
            self.regular_flag = -1
            self.n_threads = -1
            self.local_dissimilarity = distance.hamming
            self.visualization = False
            self.output_file = False
            self.dtw_to_kernel = False
            self.sigma_kernel = None
            self.itakura_max_slope = None
            self.sakoe_chiba_radius = None
    input_obj = Input()
    #to see progress, we can import tqdm and use it in dtwParallel package -> @ dtw_functions.dtw_tensor_3d 
    timevarying_distance = dtw_functions.dtw_tensor_3d(data,data,input_obj)
    return timevarying_distance