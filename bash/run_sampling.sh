#!/bin/bash
# to do
# check for task and run the correct python script
# before updating the status, check if the job ran successfuly
# OCULUS

# cd ..
set -f        # disable globbing
IFS=$'\n'     # set field separator to NL (only)

cd ..
cd Database
n=$(hostname)
if [ $n = mhshaker ]
then
    results=($(python3 DB_run_manager.py get sampling))
else
    module add singularity
    results=($(singularity run --bind /upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/:/upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/ /upb/scratch/departments/pc2/groups/hpc-prf-isys/mhshaker/s_python3.simg python3 DB_run_manager.py get sampling))
fi
cd ..

for ((idx=1; idx<${#results[@]}; ++idx)); do
    IFS=$'\t' read -r job_id runs <<< "${results[idx]}"

    if [ $n = mhshaker ]
    then
        python3 Sampling.py $job_id
        echo job_id $job_id Done
    else
        sbatch ./bash/noctua.sh $job_id
    fi
    
done
echo "All jobs are done"