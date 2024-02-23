import os
import csv

# Function to get all file locations in a directory and its subdirectories
def get_file_locations(folder):
    file_locations = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.docx'):  # Check if file ends with ".docx"
                file_locations.append(os.path.join(root, file))
    return file_locations

# Function to split file locations by folders and save each folder name to CSV file
def save_folders_to_csv(file_locations, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FD_Name', 'State', 'Region', 'Member_State', 'File_Name'])
        for location in file_locations:
            folders = location.split('/')  # Split file path by '/'
            # Only keep the last 6 parts of the path
            folders = folders[-6:]
            if 'FDs' in folders:
                folders.remove('FDs')
            if 'Financial Diaries_3' in folders:
                folders.remove('Financial Diaries_3')
            if 'Data' in folders:
                folders.remove('Data')
            writer.writerow(folders)

# Folder to search for files
folder = 'Data'

# Get file locations
file_locations = get_file_locations(folder)

# Save file locations split by folders to a CSV file
csv_filename = 'file_location.csv'
save_folders_to_csv(file_locations, csv_filename)

print(f'Folder names saved to {csv_filename}')