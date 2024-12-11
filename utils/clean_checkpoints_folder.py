"""
Sometimes, you mess up and the train model script fails, but an empty checkpoint folder
gets created anyways. This script cleans up empty directories in checkpoints. 
"""

import os

def find_and_delete_empty_dirs(base_dir):
    """
    Find and delete all empty directories within a specified base directory.

    Param:
        (str)   base_dir: The path to the base directory to search for empty directories.
    """
    if not os.path.isdir(base_dir):
        print(f"The path '{base_dir}' is not a valid directory.")
        return

    for dirpath, dirnames, filenames in os.walk(base_dir, topdown=False):
        # Check if the directory is empty
        if not dirnames and not filenames:
            print(f"Deleting empty directory: {dirpath}")
            try:
                os.rmdir(dirpath)
            except OSError as e:
                print(f"Error deleting directory {dirpath}: {e}")

if __name__ == "__main__":
    find_and_delete_empty_dirs(f"checkpoints")
