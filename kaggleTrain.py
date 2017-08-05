import sqlite3
conn = sqlite3.connect('example.db')

c = conn.cursor()

#number, string, number, string, string, number, string, string, number
for row in c.execute("SELECT funded_amnt, term, replace(int_rate, '%', '')*.01 as int_rate, emp_length, home_ownership, annual_inc, CASE WHEN loan_status in (', purpose, dti"):
    print(row)


# We can also close the connection if we are done with it.
# Just be sure any changes have been committed or they will be lost.
conn.close()