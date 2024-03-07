Required:
- Python 3.10
- Packages in requirements.txt

Generating synthetic data:
- Put required MIMIC-IV tables in data folder (diagnoses_icd, patients, admissions)
- Create environment from requirements.txt file
- Run generate/train.py (select model and parameters of choice)
- Run generate/generate.py (select model of choice and number of samples to generate)
  
Evaluating synthetic data:
- Run fidelity.py (descriptive statistics, tSNE, number of sequence steps evaluation)
- Run utility.py (classification-based GoF metric, mortality and diagnoses prediction task, attribute inference attack)
- Results are found under results/ folder
  
