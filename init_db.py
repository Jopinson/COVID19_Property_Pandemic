
import os, sys
from config import db_password 

# import the psycopg2 database adapter for PostgreSQL
from psycopg2 import connect, extensions, sql

# declare a new PostgreSQL connection object
conn = connect(
dbname = "postgres",
user = "postgres",
host = "127.0.0.1",
password = db_password
)

# object type: psycopg2.extensions.connection
print ("\ntype(conn):", type(conn))

# string for the new database name to be created
DB_NAME = "covid_property_pandemic_one"

# get the isolation leve for autocommit
autocommit = extensions.ISOLATION_LEVEL_AUTOCOMMIT
print ("ISOLATION_LEVEL_AUTOCOMMIT:", extensions.ISOLATION_LEVEL_AUTOCOMMIT)

"""
ISOLATION LEVELS for psycopg2
0 = READ UNCOMMITTED
1 = READ COMMITTED
2 = REPEATABLE READ
3 = SERIALIZABLE
4 = DEFAULT
"""

# set the isolation level for the connection's cursors
# will raise ActiveSqlTransaction exception otherwise
conn.set_isolation_level( autocommit )

# instantiate a cursor object from the connection
cursor = conn.cursor()

# use the execute() method to make a SQL request
#cursor.execute('CREATE DATABASE ' + str(DB_NAME))

# use the sql module instead to avoid SQL injection attacks
cursor.execute(sql.SQL(
"CREATE DATABASE {}"
).format(sql.Identifier( DB_NAME )))

# close the cursor to avoid memory leaks
cursor.close()

# close the connection to avoid memory leaks
conn.close()