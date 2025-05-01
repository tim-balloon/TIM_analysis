#!/bin/bash
###############################################################################
##                                      ##
##              Campus Cluster                 ##
##          Sample Python Job Batch Script             ##
##                                      ##
## SLURM Options (To view, run the following command below)         ##
##                                      ##
##   man sbatch                              ##
##                                      ##
###############################################################################
#
##SBATCH --time=00:10:00         # Job run time (hh:mm:ss)
#SBATCH --nodes=1            # Number of nodes
#SBATCH --mem=500GB
#SBATCH --ntasks-per-node=24       # Number of task (cores/ppn) per node
#SBATCH --job-name=python_job      # Name of batch job
#SBATCH --partition=caps         # Partition (queue)
#SBATCH --account=caps          # Bactch account to use
#SBATCH --output=python_job.o%j     # Name of batch job output file
##SBATCH --error=python_job.e%j     # Name of batch job error file
##SBATCH --mail-user=NetID@illinois.edu # Send email notifications
##SBATCH --mail-type=BEGIN,END      # Type of email notifications to send
#
###############################################################################
# Change to the directory from which the batch job was submitted
# Note: SLURM defaults to running jobs in the directory where
# they are submitted, no need for cd'ing to $SLURM_SUBMIT_DIR
# cd /u/mvancuyc/TIM_analysis/namap/
# Load Python/Anaconda module (Enable Python in batch job environment)
# by loading a Python or Anaconda modulefile (Only one modulfile should be
# loaded. To use the python modulefile below, uncomment that "module load ..."
# line and comment out the anaconda "module load ..." line.)
#module load python
# or
#module load anaconda3
# Run the python script(code)
python make_correlated_noisy_TOD.py params_strategy.par
# Or to run multiple instances of the python script(code)
#srun python3 my_code.py



