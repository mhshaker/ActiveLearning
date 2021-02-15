code of paper:
# How to measure uncertainty in uncertainty sampling for active learning

To run the code install the packages in the requirements.txt and run Sampling.py
The DB_insert.py is used to input multiple tasks (with different active sampling method or diffrent datasets) into the database to then be run using the run_sampling.sh which will query the parameters of each task using the database and run them on the Sampling.py
