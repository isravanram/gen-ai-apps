# Your database is stored in a local .db file (i.e., student2.db), 
# and your Python code is interacting with that file using SQLite — a lightweight, embedded SQL database engine.

import sqlite3

## connect to sqllite
connection=sqlite3.connect("influencer.db")

##create a cursor object to insert record,create table
cursor=connection.cursor()

## create the table
table_info="""
create table influencer(NAME VARCHAR(25),followers bigint,
domain VARCHAR(25),bio text)
"""

cursor.execute(table_info)

## Insert some more records
cursor.execute('''Insert Into influencer values('Krish',12220,'Food','We talk about food and its recipes')''')
cursor.execute('''Insert Into influencer values('Ravi',15000,'Fitness','We talk about fitness and its recipes')''')
cursor.execute('''Insert Into influencer values('Sita',20000,'Fashion','We talk about fashion and its trends')''')
cursor.execute('''Insert Into influencer values('Gita',18000,'Travel','We talk about travel and its experiences')''')

## Display all the records
print("The inserted records are")
data=cursor.execute('''Select * from influencer''')
for row in data:
    print(row)

## Commit your changes in the database
connection.commit()
connection.close()
