import os
from sdv.sequential import PARSynthesizer
import generate.utils.preprocess as preprocess
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
import pickle

def train_cpar(df,model_dir,epochs=10,sample_size=1,cuda=False):
    model_path = os.path.join(model_dir,f'cpar.pkl')
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

def train_dgan(df,model_dir,epochs=10,batch_size=32,sample_len=5,cuda=False):
    # #IF USING WINDOWS AS OS:
    # #dont forget to turn off multiprocessing in the torch DataLoader in dgan.py (line 652)
    # this is done by setting num_workers=0, and removing subsequent multiprocessing arguments
    # #this is giving bugs on Windows, as "fork" multiprocessing is not available on Windows, but is hardcoded in the package...
    weight_path = os.path.join(model_dir,f'dgan_weights.pt')
    model_path = os.path.join(model_dir,f'dgan_model.pkl')
    if os.path.exists(weight_path):
        raise Exception(f'there already exists a trained model, terminating...')
    
    #preprocess (padding, adding fake dynamic column, renumbering subjects)
    df = preprocess.preprocess_dgan(df)
    
    #instantiate the model (and export to later load in the weights again)
    config = DGANConfig(max_sequence_len=df.seq_num.max(),sample_len=sample_len,batch_size=batch_size,epochs=epochs,cuda=cuda)
    model = DGAN(config)
    with open(model_path,'wb') as f:
        pickle.dump(model,f)
    
    model.train_dataframe(df=df,
                          attribute_columns=['gender','age','deceased','race'],
                          feature_columns=['icd_code','final_flag','throwaway'],
                          example_id_column='subject_id',
                          time_column='seq_num',
                          discrete_columns=['icd_code','final_flag','gender','deceased','race'],
                          df_style='long')
    
    #save weights
    model.save(weight_path)

if __name__=='__main__':
    load_path = 'data'
    patient_file = 'patients.csv.gz'
    diagnoses_file = 'diagnoses_icd.csv.gz'
    admissions_file = 'admissions.csv.gz'
    patients,diagnoses = preprocess.load_mimic_data(load_path,patient_file,diagnoses_file,
                                                    admissions_file,nrows=None)
    
    #preprocess and export data
    save_path = os.path.join(load_path,'generated')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file = 'real.csv.gz'
    df = preprocess.preprocess(patients,diagnoses)
    df.to_csv(os.path.join(save_path,file),sep=',',compression='gzip')
         
    #perform training job and save model to directory
    model_dir = 'syn_model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_cpar(df=df,model_dir=model_dir,epochs=10,sample_size=1,cuda=False)
    train_dgan(df=df,model_dir=model_dir,epochs=10,batch_size=8,sample_len=5,cuda=False)