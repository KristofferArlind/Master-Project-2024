#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --account=share-ie-idi
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --job-name="us_rf_all_cv_2"
#SBATCH --output=output/us_rf_all_cv_2.txt
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
python ../scripts/models/run_model.py -model rf -region US -fundamentals_delay 90 --test_on_fridays -test_date_csv time_periods/test_2020_to_2023.csv -train_date_csv time_periods/train_1965_to_2020.csv -val_method blocked --scale -win_pc 0.01 -cv_grid_values_csv scripts/models/cv_values/rf_2.csv -parallel 8 -n_trials 200 --friday -drop_cols ggroup,gind,gsubind,exchange_code

uname -a

