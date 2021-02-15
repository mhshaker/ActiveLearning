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
results=($(python3 DB_run_manager.py get unc_test))
cd ..

n=$(hostname)
for ((idx=1; idx<${#results[@]}; ++idx)); do
    IFS=$'\t' read -r job_id runs <<< "${results[idx]}"
    if [ $n = fe1 ]
    then
        run oculus.sh $job_id $runs
    else
        python3 UNC_test.py $job_id
        echo job_id $job_id Done
    fi
done
echo "All jobs are done"