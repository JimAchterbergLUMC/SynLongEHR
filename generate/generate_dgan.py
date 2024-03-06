import pickle
import os
import preprocess



def generate_dgan(n_samples):
    model_path = os.path.join('model',f'dgan_model.pkl')
    weight_path = os.path.join('model',f'dgan_weights.pt')
    #load instantiated model and input saved weights
    with open(model_path,'rb') as f:
        model = pickle.load(f)
    model = model.load(weight_path)
    #return samples
    samples = model.generate_dataframe(n_samples)
    return samples


if __name__=='__main__':
    #CHANGE WHEN RUNNING SCRIPT!!!
    n_samples = 100
    path = 'C:/Users/Jim/Documents/thesis_paper/data'

    #generate synthetic samples from trained models
    #also postprocess and save samples
    
    samples = generate_dgan(n_samples)
    samples = preprocess.postprocess_dgan(samples)

    save_path = os.path.join(path,'processed','generated','dgan')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = f'dgan.csv.gz'
    samples.to_csv(os.path.join(save_path,file),compression='gzip',sep=',')

    

    


    
    
    


    