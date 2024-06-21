import dill
import os


def save_dill(obj: object, path: str) -> None:
    """
    Save a Python object to a pickle file.

    Parameters
    ----------
    obj : object
        The Python object to save.
    path : str
        The path to save the object to.

    """
    with open(path, 'wb') as f:
        dill.dump(obj, f)


def load_dill(path: str) -> object:
    """
    Load a Python object from a pickle file.

    Parameters
    ----------
    path : str
        The path to the pickle file.

    Returns
    -------
    object
        The Python object loaded from the pickle file.

    """
    with open(path, 'rb') as f:
        return dill.load(f)


def create_dir_if_required(script_filepath: str, dir_name: str) -> str:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    script_filepath : str
        The path to the script that is calling this function.
    dir_name : str
        The name of the directory to create.

    Returns
    -------
    str
        The absolute path to the directory created.
    """
    cwd = os.path.dirname(os.path.realpath(script_filepath))
    dir_to_make = os.path.join(cwd, dir_name)

    if not os.path.exists(dir_to_make):
        os.makedirs(dir_to_make)

    return dir_to_make
