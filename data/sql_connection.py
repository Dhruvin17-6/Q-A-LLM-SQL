import yaml
from langchain_community.utilities import SQLDatabase
from pathlib import Path

# Load the configuration from app_config.yml
config_path = Path(__file__).parent / 'app_config.yml'
with open(config_path, 'r') as file:
    app_config = yaml.safe_load(file)

# Get the SQL connection URI from the configuration file
mysql_uri = app_config["directories"]["sqldb_directory"]

# Create the SQLDatabase object using the URI
db = SQLDatabase.from_uri(mysql_uri)

# Print database dialect and usable table names
print(db.dialect)
print(db.get_usable_table_names())

# Run a SQL query on the customer table
result = db.run("SELECT * FROM customer LIMIT 10;")
print(result)





# import yaml
# import os
# from langchain_community.utilities import SQLDatabase
# from pyprojroot import here

# # Load the YAML configuration
# config_file_path = str(here("configs/app_config.yml"))
# with open(config_file_path, 'r') as file:
#     config = yaml.safe_load(file)

# # Access the MySQL configuration
# mysql_config = config['mysql']
# mysql_username = mysql_config['username']
# mysql_password = mysql_config['password']
# mysql_host = mysql_config['host']
# mysql_database = mysql_config['database']

# # Construct the MySQL URI
# mysql_uri = f"mysql+mysqlconnector://{mysql_username}:{mysql_password}@{mysql_host}/{mysql_database}"

# # Connect to the MySQL database
# db = SQLDatabase.from_uri(mysql_uri)

# # Debugging prints
# print(f"Connected to MySQL using dialect: {db.dialect}")
# print("Usable table names:", db.get_usable_table_names())

# # Run a SQL query on the `customer` table
# result = db.run("SELECT * FROM customer LIMIT 10;")
# print(result)
