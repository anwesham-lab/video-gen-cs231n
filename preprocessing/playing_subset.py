import os
import shutil

def delete_subdirectories_without_substring(directory, substring):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            if substring not in name:
                path = os.path.join(root, name)
                print("Deleting directory:", path)
                # Remove the directory and its contents
                shutil.rmtree(path)
                # Alternatively, use the following line to only remove the empty directory
                # os.rmdir(path)

if __name__ == "__main__":
    directory = "data/UCF101/classified"
    substring = "Playing"

    delete_subdirectories_without_substring(directory, substring)