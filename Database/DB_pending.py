import mysql.connector as db

mydb = db.connect(host="131.234.250.119", user="root", passwd="uncertainty", database="uncertainty")
mycursor = mydb.cursor()

# show pending jobs in the DB
mycursor.execute("SELECT id, task, run_name, dataset, runs, result_type, algo, prams FROM experiments Where status='pending'")
results = mycursor.fetchall()
print("[DB: pending jobs]")
for row in results:
	print(row)
