#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --job-name="us_rf_all_tp"
#SBATCH --output=output/us_rf_all_tp.txt
#SBATCH --mail-user=ktarlind@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Anaconda3/2022.10
python ../scripts/models/run_model_n_tp.py -model rf -region US -fundamentals_delay 90 --test_on_fridays --scale -win_pc 0.01 -parallel 1 --friday -drop_cols ggroup,gind,gsubind,exchange_code -rf_jobs 30 -trees 3000 -max_samples 0.1

uname -a

