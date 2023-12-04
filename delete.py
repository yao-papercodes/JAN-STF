import os
import shutil

def delete_pkl_files(path, exc_folders_list):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pkl"):
                file_path = os.path.join(root, file)
                # Gets the directory name of the file
                folder_name = os.path.basename(os.path.dirname(file_path))
                # Determines whether the directory name is in exc_folders_list
                if folder_name not in exc_folders_list:
                    os.remove(file_path)  # Delete files with '.pkl' extension

if __name__ == "__main__":
    folder_path = "./exp"  # Replace with your folder path
    exc_folders_list = [
        "2023-07-11-10:26:00", "2023-07-11-12:13:08", "2023-07-12-16:34:37"
    ]
    delete_pkl_files(folder_path, exc_folders_list)
