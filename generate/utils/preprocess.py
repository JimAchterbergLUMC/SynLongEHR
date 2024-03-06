import pandas as pd
import numpy as np
import os
from sdv.metadata import SingleTableMetadata
from gretel_synthetics.timeseries_dgan.config import OutputType


def load_mimic_data(load_path,patient_file,diagnoses_file,admissions_file,nrows=None):
    #loads compressed mimic file, while only selecting columns and rows of interest
    patient_file = os.path.join(load_path,patient_file)
    diagnoses_file = os.path.join(load_path,diagnoses_file)
    admissions_file = os.path.join(load_path,admissions_file)
    #setting filepaths and columns of interest
    diagnoses_cols = ['subject_id','hadm_id','seq_num','icd_code']
    patient_cols = ['subject_id','gender','anchor_age','dod']
    adm_cols = ['subject_id','race']
    #loading data
    diagnoses = pd.read_csv(diagnoses_file,usecols=diagnoses_cols,compression='gzip',sep=',',nrows=nrows)
    patients = pd.read_csv(patient_file,usecols=patient_cols,compression='gzip',sep=',',nrows=nrows)
    adm = pd.read_csv(admissions_file,usecols=adm_cols,compression='gzip',sep=',',nrows=nrows)
    #select only first admission for the race
    adm = adm.groupby('subject_id').first()
    adm = adm.reset_index()
    #merging additional attributes to patients
    patients = patients.merge(adm,on='subject_id',how='left')
    return patients,diagnoses

    
#----------------------------------------------------------------------------------------------------------------------
#MIMIC-IV preprocessing


def preprocess(patients,diagnoses):
    #ensure we are not working on view of dataframe
    patients = patients.copy()
    diagnoses = diagnoses.copy()
    #select only numerical icd codes, and get sections by grabbing first three numbers
    diagnoses = diagnoses[diagnoses.icd_code.str.contains(r'^\d+$')]
    diagnoses.icd_code = diagnoses.icd_code.str[0:3]
    diagnoses.icd_code = diagnoses.icd_code.astype(int)

    #filter diagnoses to only include first hospital admission
    #this can be a lot more efficient?
    first_admission = diagnoses.groupby('subject_id').hadm_id.min().reset_index()
    diagnoses = diagnoses.merge(first_admission.rename({'hadm_id':'first_hadm_id'},axis=1),on='subject_id')
    diagnoses['rank'] = (diagnoses['hadm_id']==diagnoses['first_hadm_id']).astype(int)
    diagnoses = diagnoses[diagnoses['rank']==1]
    diagnoses = diagnoses.drop(['hadm_id','first_hadm_id','rank'],axis=1)

    #select heart disease patients (icd section 410-414)
    hd_patients = diagnoses.subject_id[(diagnoses.icd_code>=410) & (diagnoses.icd_code<=414)]
    patients = patients[patients.subject_id.isin(hd_patients)]
    diagnoses = diagnoses[diagnoses.subject_id.isin(hd_patients)]
    #drop diagnoses duplicates if there are somehow still two values for same sequence number for a patient
    diagnoses = diagnoses.drop_duplicates(subset=['subject_id','seq_num'])
    #add deceased column
    patients.dod = patients.dod.notnull().astype(int).astype(bool)
    patients = patients.rename({'dod':'deceased','anchor_age':'age'},axis=1)   
    #load mapping table and map diagnostics codes to corresponding categories
    #note it does not include congenital anomalies(Cong), perinatal diseases (Perin), pregnancy disease (Preg)
    map = pd.read_csv('generate/utils/icd_mapping.csv',sep=',')
    def map_to_category(row):
        category = map.description_short[(row.icd_code>=map.start_section)&(row.icd_code<=map.end_section)]
        return category.values[0] if not category.empty else None
    diagnoses.icd_code = diagnoses.apply(map_to_category,axis=1)
    #remove nan rows
    diagnoses = diagnoses.dropna(axis=0)
    #map races to higher level categories
    def race_mapping(s):
        if isinstance(s,str):
            if 'HISPANIC' in s or 'SOUTH AMERICAN' in s:
                return 'hispanic'
            elif 'WHITE' in s or 'PORTUGUESE' in s:
                return 'white'
            elif 'BLACK' in s:
                return 'black'
            elif 'ASIAN' in s:
                return 'asian'
            elif 'NATIVE' in s:
                return 'native_american'
            elif 'MULTIPLE' in s:
                return 'multiple'
            else:
                return 'unknown'
        else:
            return 'unknown'
    patients.race = patients.race.apply(race_mapping)
    #renumber the timestep column to correct ascending non missing order
    diagnoses = diagnoses.sort_values(['subject_id','seq_num'])
    diagnoses.seq_num = diagnoses.groupby('subject_id').cumcount()+1
    #only keep subjects with at least 5 timesteps
    diagnoses = diagnoses.groupby('subject_id').filter(lambda x: x.seq_num.max()>=5)
    #for some reason we need to rearrange variables in cpar to have continuous first
    df = diagnoses.merge(patients,on='subject_id',how='left')
    df = df[['subject_id','seq_num','age','gender','deceased','race','icd_code']]
    return df


def get_metadata(df,numerical_cols=[],categorical_cols=[],prim_key='primary_key',subject_key='subject_id',sequence_key='seq_num'):
    #create metadata object necessary in SDV PAR model
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    #manually set dtypes of the columns, to ensure they are correct 
    metadata.update_column(column_name=prim_key,sdtype='id')
    metadata.update_column(column_name=subject_key,sdtype='id')
    metadata.update_column(column_name=sequence_key,sdtype='numerical')
    #if we specify categories as strings it can auto-detect correct sdtypes
    for col in numerical_cols:
        metadata.update_column(column_name=col,sdtype='numerical')
    for col in categorical_cols:
        metadata.update_column(column_name=col,sdtype='categorical')
    #set primary and sequence key
    metadata.set_primary_key(column_name=prim_key)
    metadata.set_sequence_key(column_name=subject_key)
    metadata.set_sequence_index(column_name=sequence_key)
    return metadata


#pads a longitudinal pandas dataframe
def pad_df(group,static_cols,dynamic_cols,pad_to,pad_with=0,timestep_idx='seq_num'):
        #find the missing timesteps
        existing_timesteps = group[timestep_idx].unique()
        missing_timesteps = list(set(range(1,pad_to+1))-set(existing_timesteps))
        if missing_timesteps:
            missing_rows = pd.DataFrame({timestep_idx: missing_timesteps})
            for col in group.columns:
                #pad with number if numeric or zero string if non numeric
                
                #turn the missing rows into: <PAD> for dynamic cols, first static value for static cols
                if col in static_cols:
                    missing_rows[col] = group[col].iloc[0]
                elif col in dynamic_cols:
                    missing_rows[col] = pad_with
            #concatenate rows to current subject
            group = pd.concat([group, missing_rows], ignore_index=True, sort=False)
        return group.sort_values(timestep_idx)

def preprocess_dgan(df,subject_idx='subject_id',timestep_idx='seq_num'):
        max_t = df[timestep_idx].max()
        max_t = int(np.ceil(max_t/5)*5)
        #add a fake continuous dynamic column (DGAN package issue)
        df['throwaway'] = 0
        #add a binary column, where a 1 indicates it is the final timestep 
        #all timesteps AFTER this first 1 are masked in the discriminator
        df['final_flag'] = 0
        df.loc[df.groupby(subject_idx)[timestep_idx].idxmax(), 'final_flag'] = 1
        #pad dynamic features 
        #we pad to a multiple of 5 in this use case to output batches of 5 timesteps
        df = df.groupby('subject_id').apply(pad_df,static_cols=['subject_id','age','gender','deceased','race'],
                                            dynamic_cols=['icd_code','throwaway','final_flag'],pad_to=max_t).reset_index(drop=True)
        #renumber subjects starting from 0 (DGAN package error)
        df[subject_idx] = pd.factorize(df[subject_idx])[0]
        return df

def preprocess_cpar(df,subject_idx='subject_id',timestep_idx='seq_num'):
    #create a primary key column, which is distinct for each row, and add to dataframe
    # -> combination of subject_id and seq_num, since we only use first hospital admission
    prim_key = pd.DataFrame(df[subject_idx].astype(str) + '_' + df[timestep_idx].astype(str),columns=['primary_key'])
    df = pd.concat((prim_key,df),axis=1)
    return df

def postprocess_dgan(df):
    df = df.sort_values(['subject_id','seq_num'])
    #remove padding (so infer real amount of timesteps)
    #check first prediction of end flag
    first_flag = df[df['final_flag'] == 1].groupby('subject_id')['seq_num'].min().reset_index()
    #merge to retain only rows before the first predicted end flag
    df = df.merge(first_flag, on='subject_id', suffixes=('', '_min'))
    df = df[df['seq_num'] <= df['seq_num_min']]
    #remove any rows still containing the padding icd code
    df = df[df['icd_code'!=0]]
    # Drop the helper columns
    df = df.drop(['seq_num_min','final_flag','throwaway'],axis=1)
    #round age column to integer
    df.age = df.age.round()
    return df

def postprocess_cpar(df):
    df = df.sort_values(['subject_id','seq_num'])
    #remove primary key
    df = df.drop('primary_key',axis=1)
    #renumber seqnum 
    df = df.sort_values(['subject_id','seq_num'])
    df.seq_num = df.groupby('subject_id').cumcount()+1
    return df

#randomly perturbes label with level% probability
def rd_perturbation(column,level):
    perturbed_labels = column.copy()
    mask = np.random.rand(len(column)) < level  
    unique_labels = column.unique()
    for idx, label in enumerate(perturbed_labels):
        if mask[idx]:
            perturbed_labels[idx] = np.random.choice(np.setdiff1d(unique_labels, [label]))
    return perturbed_labels

#adds random gaussian noise of level% of the real std
def rd_noise(column,level):
    return column + np.random.normal(loc=0,scale=level*np.std(column),size=len(column))