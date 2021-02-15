
# ########### Do not print in this file, as the prints are used as input for the bash file ###########

import sys
import mysql.connector as db

mode = "get"
var = "unc"
job_id = 0

if len(sys.argv) > 1:
    mode   = sys.argv[1]
if len(sys.argv) > 2:
    var    = sys.argv[2]
if len(sys.argv) > 3:
    job_id = int(sys.argv[3])

mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
mycursor = mydb.cursor()

if(mode == "get"):
    mycursor.execute(f"SELECT id, runs FROM experiments Where status='pending' AND task='{var}'")  # remove  "AND id>=1582" after testing PCA experiment
    results = mycursor.fetchall()

    print("id\ttabruns")
    for res in results:
        print(f"{res[0]}\t{res[1]}")
elif mode == "set":
    mycursor.execute(f"UPDATE experiments SET status='{var}' Where id={job_id}")
    mydb.commit()
