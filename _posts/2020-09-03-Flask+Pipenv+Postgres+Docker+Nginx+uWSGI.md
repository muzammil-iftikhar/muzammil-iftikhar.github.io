---
title:  "Flask+Pipenv+Postgres+Docker+Nginx+uWSGI"
date: 2020-09-03
categories: [Tutorial]
tags: [python,flask,pipenv,nginx,docker,docker-compose,uwsgi]
excerpt: "Learn how you can make your flaskapp production ready using Pipenv, uWSGI, Nginx and Docker"
author_profile: true
mathjax: true
---
As a Machine Learning Engineer, you would have to deploy your models to production. In this tutorial, we are going to learn how you can do that.
We will start with creating our virtual environment using Pipenv. Then we will create a basic FlaskApp. Then we will connect that FlaskApp to a Postgres database. Then we will launch our FlaskApp using uWSGI. Then we will put a Nginx at the front of our FlaskApp and finally we will dockerize this whole ecosystem. 

Highlights:
* Creating virtual environemnt for our project using Pipenv
* Creating a Flaskapp
* Creating postgres script in python to execute queries
* Creating Dockerfile for our Flaskapp
* Creating Dockerfile for our Postgres db
* Creating Nginx Dockerfile
* Creating docker-compose.yml file
* Starting flaskapp using uWSGI
* Running the whole ecosystem with docker-compose

This is going to be lengthy post. Let's get started.

### Creating a virtual environment using Pipenv

```python
pipenv install flask psycopg2 uwsgi
```

Please note that i am not going to use Flask-SQLAlchemy to connect to postgres db. Instead i will be using psycopg2 library, since i like writing SQL queries better than the ORM. Also, you can write SQL queries in SQLAlchemy as well but i like psycopg2 more.  
One more thing, if you are on windows, the above command will fail on uWSGI installation. In such case, you can just remove *uwsgi* from above command and we will install that later using Docker directly in our container.

```python
pipenv lock
```

### Creating FlaskApp

This is going to be a very simple Flaskapp that will take the name and location in input of the GET request from browser and store that in a postgres db.  
Then we will define another view to fetch the records from db based on name field. Create a *tutorial.py* file at the root directory.

```python
from database.db import execute_query
from flask import Flask

app = Flask(__name__)


@app.route('/<string:name>/<string:location>')
def index(name, location):
    return execute_query('insert', name, location)


@app.route('/<string:name>')
def fetch(name):
    return execute_query('select', name)
```

### Creating Postgres python script

Create a folder inside the root folder *database* and inside it create a file *db.py*

```python
import psycopg2

def execute_query(operation, name=None, location=None):

    connection = psycopg2.connect(
        user='postgres',
        password='admin',
        host='localhost',
        port='5432',
        database='flask'
    )
    cursor = connection.cursor()
    if operation == 'select':
        try:
            query = f"SELECT * FROM users where name='{name}';"
            cursor.execute(query)
            records = cursor.fetchone()
            cursor.close()
            connection.close()
            return f'{records}'
        except Exception as e:
            cursor.close()
            connection.close()
            return f'{e}'

    elif operation == 'insert':
        try:
            query = f"INSERT INTO users (name,location) VALUES ('{name}','{location}');"
            cursor.execute(query)
            connection.commit()
            cursor.close()
            connection.close()
            return f'User added successfully'
        except Exception as e:
            cursor.close()
            connection.close()
            return f'{e}'
```

### Creating Dockerfile for our Flaskapp

This will go inside the root directory

```python
FROM python:3.8

RUN pip3 install pipenv
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile
COPY . .
CMD [ "python","tutorial.py" ]
```

### Creating Dockerfile for our Postgres

If you look at the connection string in our postgres code above, we are connecting with database *flask* and then we are executing our queries in the table *users*. We have to manage that via our Dockerfile because we have to ensure that once our postgres container is up and running, it is fully loaded with all the requirements our main code needs.  
This will go inside the *database* directory

```python
FROM postgres:alpine

ENV POSTGRES_PASSWORD admin
ENV POSTGRES_DB flask

COPY init.sql /docker-entrypoint-initdb.d/
```

Create a init.sql file and paste the below code:

```python
CREATE TABLE IF NOT EXISTS users (
    id serial PRIMARY KEY,
    name varchar(50) UNIQUE NOT NULL,
    location varchar(50)
);
```

### Creating Nginx Dockerfile

Create a new folder *nginx* inside our main flask folder. Inside this folder, create a Dockerfile and a nginx.conf file

```python
FROM nginx

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/conf.d/
```

Create a nginx.conf file and paste the following:

```python
server {
    listen 80;

    location / {
        include uwsgi_params;
        uwsgi_pass flaskapp:8080;
    }
}
```

Our nginx will listen on port 80 and will forward the requests to our flaskapp on port 8080

### Creating a docker-compose file

Create a *docker-compose.yml* file at the root directory

```python
version: "3.5"

services:
  flaskapp:
    build: .
    image: muuzii/flaskapp
    container_name: flask_app
    restart: always
    environment:
      - APP_NAME=MyFlaskApp
    expose:
      - 8080

  postgres:
    build: ./database
    image: muuzii/postgresdb
    container_name: flask_db
    volumes:
      - pgdata:/var/lib/postgresql/data

  nginx:
    build: ./nginx
    container_name: flask_nginx
    restart: always
    ports:
      - "80:80"

volumes:
  pgdata:
    external: false
```

Volume is created to make the postgres data persistent i.e. we doesn't lose our data once we shutdown the containers.

### Starting Flaskapp using uWSGI

For this we have to make few changes in our original Dockerfile of our Flaskapp

```python
FROM python:3.8

RUN pip3 install pipenv
RUN pip3 install uwsgi
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile
COPY . .
CMD [ "uwsgi","app.ini" ]
```

Next, create a file at the root directory *app.ini* and paste the following into it:

```python
[uwsgi]
wsgi-file = tutorial.py
callable = app
socket = :8080
processes = 4
threads = 2
master = true
chmod-socket = 660
vacuum = true
die-on-term = true
```

After all this is done, goto the terminal and into the root path where *docker-compose.yml* file has been created. Run the following command:

```python
docker-compose up --build
```

And there you have a fully functional, production ready Flaskapp :)
