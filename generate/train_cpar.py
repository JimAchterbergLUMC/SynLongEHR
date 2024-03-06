import os
from sdv.sequential import PARSynthesizer
import preprocess

def train_cpar(df,epochs=10,sample_size=1,cuda=False):
    model_dir = 'model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir,f'cpar.pkl')
    if os.path.exists(model_path):
        raise Exception(f'there already exists a trained model, terminating...')
    
    df = preprocess.preprocess_cpar(df)
    metadata = preprocess.get_metadata(df)
    
    context_columns = ['age','gender','deceased','race']

    model = PARSynthesizer(metadata=metadata,context_columns=context_columns,
                           epochs=epochs,sample_size=sample_size,
                           cuda=cuda,verbose=True,
                           enforce_min_max_values=True,enforce_rounding=False,
                           locales=None,segment_size=None)

    model.fit(df)
    model.save(model_path)

if __name__=='__main__':
    #CHANGE WHEN RUNNING SCRIPT!!!
    #also dont forget to change nnet architecture if necessary
    path = 'C:/Users/Jim/Documents/thesis_paper/data'
    params = {'EPOCHS':10,
              'SAMPLE_SIZE':1,
              'CUDA':False}

    #loading raw data
    load_path = path + '/raw' + '/hosp'
    patient_file = 'patients.csv.gz'
    diagnoses_file = 'diagnoses_icd.csv.gz'
    admissions_file = 'admissions.csv.gz'
    patients,diagnoses = preprocess.load_mimic_data(load_path,patient_file,diagnoses_file,
                                                    admissions_file,nrows=None)
    
    #preprocess and export data
    save_path = path + '/processed' + '/generated' + '/real' 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = 'real.csv.gz'
    df = preprocess.preprocess(patients,diagnoses)
    df.to_csv(os.path.join(save_path,file),sep=',',compression='gzip')
       
    #check datatypes   
    # #perform training job
    train_cpar(df,epochs=params['EPOCHS'],sample_size=params['SAMPLE_SIZE'],cuda=params['CUDA'])
    


