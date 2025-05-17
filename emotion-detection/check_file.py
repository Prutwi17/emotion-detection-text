import os

# Check if the train.txt file exists
file_path = "data/train.txt"
exists = os.path.exists(file_path)

print(f"Does '{file_path}' exist? --> {exists}")
