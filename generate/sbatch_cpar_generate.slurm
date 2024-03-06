#!/bin/bash
#
# set the job parameters
#SBATCH -J generate_cpar_v00
#SBATCH -n 16
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --mail-user=j.l.achterberg@lumc.nl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -t 23:59:00

# existing module cleanup
module purge > /dev/null 2>&1

# load necessary modules
module load system/python/3.10.2
module load library/cuda/11.6.1/gcc.8.3.1
module load library/cudnn/11.6/cudnn


# activate python venv (should be inside cwd)
source venv/bin/activate

# here we actually run the job 
python3 train_cpar.py
python3 generate_cpar.py


# deactivate the venv
deactivate

echo "Program finished with exit code $?"