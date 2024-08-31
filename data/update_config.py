import yaml
from pyprojroot import here

# Define the new path for the SQL database
new_sqldb_directory = "new/path/to/your/sqldb.db"  # Change this to your desired path

# Path to the YAML config file
config_file_path = str(here("configs/app_config.yml"))

# Load the existing YAML configuration
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Update only the sqldb_directory with the new path
config['directories']['sqldb_directory'] = new_sqldb_directory

# Write the updated configuration back to the YAML file
with open(config_file_path, 'w') as file:
    yaml.safe_dump(config, file, default_flow_style=False)

print("The sqldb_directory has been updated to:", new_sqldb_directory)
