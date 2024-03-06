#script to create synthetic data by adding random noise / label perturbation 


import preprocess
import numpy as np

#import data
path = 'C:/Users/Jim/Documents/thesis_paper/data'
load_path = path + '/raw' + '/hosp'
patient_file = 'patients.csv.gz'
diagnoses_file = 'diagnoses_icd.csv.gz'
admissions_file = 'admissions.csv.gz'
patients,diagnoses = preprocess.load_mimic_data(load_path,patient_file,diagnoses_file,
                                                admissions_file,nrows=10000)

df = preprocess.preprocess(patients,diagnoses)

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
    
cat_cols = ['gender','deceased','race','icd_code']
cont_cols = ['age']

df_noisy = df.copy()
df_noisy[cat_cols] = df_noisy[cat_cols].apply(rd_perturbation,level=.1)
df_noisy[cont_cols] = df_noisy[cont_cols].apply(rd_noise,level=.1)


