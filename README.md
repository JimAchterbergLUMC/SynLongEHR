# Overview
This repository is based on the paper 'On the evaluation of synthetic longitudinal electronic health records' by Jim Achterberg, Marcel Haas, and Marco Spruit from Leiden University Medical Center. 
The paper provides a discussion on evaluation metrics for synthetic longitudinal Electronic Health Records (EHRs). To support the discussion, synthetic EHRs from MIMIC-IV are generated, to subsequently apply evaluation metrics. For further details, we refer to the paper.

# Features 
- Generate synthetic longitudinal data. Mixed numerical and categorical data types. Variable-length sequences.
- Evaluate the quality of synthetic longitudinal data.


# Usage

## Required
- Python 3.10
- Packages in requirements.txt

## Generate synthetic longitudinal EHRs from MIMIC-IV
- Put required MIMIC-IV tables in data folder (diagnoses_icd, patients, admissions)
- Create environment from requirements.txt file
- Run generate/train.py (select model and parameters of choice)
- Run generate/generate.py (select model of choice and number of samples to generate)
- Automatically puts generated synthetic data in data/generated folder
  
## Evaluate the quality of generated synthetic EHRs
- Run evaluate/fidelity.py (descriptive statistics, low-dimensional projections from tSNE or UMAP, number of sequence steps)
- Run evaluate/utility.py (classification-based GoF metric, mortality and diagnoses prediction task, attribute inference attack)
- Results are found under results/ folder
  
