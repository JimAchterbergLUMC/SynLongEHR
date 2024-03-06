import pickle
import os
import preprocess


def generate_cpar(n_samples,version):
    load_path = os.path.join('model',f'cpar.pkl')
    model = pickle.load(open(load_path,'rb'))
    samples = model.sample(num_sequences=n_samples,sequence_length=None)
    return samples


if __name__=='__main__':
    #CHANGE WHEN RUNNING SCRIPT!!!
    n_samples = 18245
    path = 'C:/Users/Jim/Documents/thesis_paper/data'

    #generate synthetic samples from trained models
    #also postprocess and save samples
    samples = generate_cpar(n_samples)
    samples = preprocess.postprocess_cpar(samples)
    
    save_path = os.path.join(path,'processed','generated','cpar')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = f'cpar.csv.gz'
    samples.to_csv(os.path.join(save_path,file),compression='gzip',sep=',')
    
    


    