import os
import re

folder_path = './hh'

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a .wav file and matches the pattern
    if filename.endswith('.wav') and re.match(r'hh#\d{2} merged\.\d+\.wav', filename):
        # Extract the number part from the filename
        new_name = re.sub(r'hh#\d{2} merged\.(\d+)\.wav', r'\1.wav', filename)
        # Define the full path for the old and new file names
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        # Rename the file
        os.rename(old_file, new_file)

print("Files have been renamed.")