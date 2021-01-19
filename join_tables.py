# code reference: https://www.postgresqltutorial.com/postgresql-python/call-stored-procedures/

from config import db_password
import psycopg2
from pathlib import Path

proj_root_dir = Path(__file__).resolve().parents[0]
conn = None
try:
    conn = psycopg2.connect(
            database="covid_property_pandemic_one", 
            user='postgres', 
            password=db_password, 
            host='localhost', 
            port= '5432'
    )
    #Setting auto commit false
    conn.autocommit = True

    #Creating a cursor object using the cursor() method
    cur = conn.cursor()

    ## Code reference from https://stackoverflow.com/questions/17261061/execute-sql-schema-in-psycopg2-in-python
    # call a stored procedure
    cur.execute(open(f"{proj_root_dir}/data/queries/proj_tables.sql", "r").read())

    # commit the transaction
    conn.commit()

    # close the cursor
    cur.close()

    print('Table join complete')


except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
