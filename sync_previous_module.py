"""
Description:
Note: Make sure that both the new and old module files are in same directory!

This script helps you sync your previous module works with current modules.
It takes 2 arguments, source_dir_name and destination_dir_name.
All the files which will be moved are specified in files_to_sync.txt as newline separated strings

Usage: python sync_previous_module.py <source_dir_name> <dest_dir_name>

Ex:  python sync_previous_module.py mle-module-0-sauravpanda24 mle-module-1-sauravpanda24
"""
import os
import shutil
import sys

if len(sys.argv) != 3:
    print(
        "Invalid argument count! Please pass source directory and destination directory after the file name"
    )
    sys.exit()

# Get the users path to evaluate the username and root directory
current_path = os.getcwd()
grandparent_path = "/".join(current_path.split("/")[:-1])

print("Looking for modules in : ", grandparent_path)

# List of files which we want to move
f = open("files_to_sync.txt", "r+")
files_to_move = f.read().splitlines()
f.close()

# get the source and destination from arguments
source = sys.argv[1]
dest = sys.argv[2]

# copy the files from source to destination
try:
    for file in files_to_move:
        print(f"Moving file : ", file)
        shutil.copy(
            os.path.join(grandparent_path, source, file),
            os.path.join(grandparent_path, dest, file),
        )
    print(f"Finished moving {len(files_to_move)} files")
except Exception as e:
    print(
        "Something went wrong! please check if the source and destination folders are present in same folder"
    )
