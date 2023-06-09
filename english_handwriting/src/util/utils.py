import os


def set_working_dir(current_file_path):
    # get the absolute path of the root dir

    # iterate over the path until the root dir is reached
    root_dir_name = 'english_handwriting'
    root_dir = current_file_path
    while os.path.basename(root_dir) != root_dir_name:
        root_dir = os.path.dirname(root_dir)

        if root_dir == '/':
            raise Exception("Root dir not found")

    # calculate the relative path from the current file to the root dir
    relative_path = os.path.relpath(current_file_path, start=root_dir)

    # count the number of directories in the relative path
    num_levels = relative_path.count(os.sep)

    # create the relative prefix
    relative_prefix = os.path.join(*(['..'] * num_levels))

    # change the working directory
    os.chdir(relative_prefix)



