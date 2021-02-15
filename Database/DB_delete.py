import mysql.connector as db
import os

update = False

mydb = db.connect(host="131.234.250.119", user="noctua", passwd="uncertainty", database="uncertainty")
mycursor = mydb.cursor()

if(update == False):
    sql = "DELETE FROM experiments WHERE id >= 2954"
else:
    # sql = "UPDATE experiments SET status='Error' Where status='running'"
    # sql = "UPDATE experiments SET status='pending' Where run_name='DF_sampling'"
    sql = "UPDATE experiments SET runs='400' Where run_name='DF_sampling'"
mycursor.execute(sql)
mydb.commit() # save
