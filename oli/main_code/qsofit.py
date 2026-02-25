import numpy as np
from typing import Any
import os
import warnings
from astropy.io import fits

import sys
sys.path.append("/Users/o_thorp/Downloads/my_stuff/Uni/other/scholarships/denison_2026/oli")

from . import constants as const
from main_code.helpers import get_new_qso_filename

def edit_qsopar(
    file_name: str,
    value_name_to_edit: str,
    new_value: Any,
    col_name_and_val: tuple[str, Any] | None, # ("linename", "Ha_na"), ("maxsig", 0.001690)
    mask: np.ndarray | None = None,
    folder_name: str = "pyqsofit_code/data/",
    ext_num: int = 1,
    make_copy: bool = True,
    print_change: bool = True
) -> None:
    """
    Edits a value in the qsopar file.

    Parameters
    ----------
    file_name: str
        The name of the qsopar file.
    value_name_to_edit: str
        The name of the value to edit.
    new_value: Any
        The new value to set.
    col_name_and_val: tuple[str, Any] | None
        The column name and value to match.
    mask: np.ndarray | None
        The mask used to locate the data to edit. Should only be
        provided if col_name_and_val is None.
    folder_name: str
        The name of the folder containing the qsopar file.
    ext_num: int
        The extension number of the hdulist to edit in the qsopar file.
    make_copy: bool
        Whether to make a copy of the edited qsopar file or save changes
        to the original file.
    print_change: bool
        Whether to print the changes made.
    """
    try:
        with fits.open(folder_name + file_name, mode=('readonly' if make_copy else 'update')) as hdul:

            data = hdul[ext_num].data
            col_name, col_name_val = col_name_and_val if col_name_and_val is not None else (None, None)
            
            if col_name_and_val is not None:
                if mask is not None:
                    raise ValueError("Exactly one of col_name_and_val or mask must be None")
                mask = data[col_name] == col_name_val
            elif mask is None:
                raise ValueError("Exactly one of col_name_and_val or mask must be None")
            
            if np.any(mask):
                # Update 'ngauss' for that specific row (e.g., changing 1 to 2)
                old_val = data[value_name_to_edit][mask]
                if np.all(old_val == new_value):
                    if col_name_and_val is not None:
                        warn_msg = f"Value '{value_name_to_edit}' for {col_name} = {col_name_val} is already {new_value}. No changes made."
                    else:
                        warn_msg = f"Value '{value_name_to_edit}' is already {new_value}. No changes made."
                    warnings.warn(warn_msg)
                else:
                    data[value_name_to_edit][mask] = new_value
                    new_file_name = get_new_qso_filename(file_name, folder_name=folder_name)
                    if print_change:
                        if col_name_and_val is not None:
                            print(f"{value_name_to_edit} for {col_name} = {col_name_val} updated successfully from {old_val} to {new_value}.")
                        else:
                            print(f"{value_name_to_edit} updated successfully from {old_val} to {new_value}.")
                        if make_copy:
                            print(f"Changes saved to new file: {folder_name + new_file_name}")
                        else:
                            print(f"Changes saved to original file: {folder_name + file_name}")
                    if make_copy:
                        hdul.writeto(folder_name + new_file_name, overwrite=False)
            else:
                if col_name_and_val is not None:
                    warn_msg = f"{col_name} = {col_name_val} not found in the table."
                    warnings.warn(warn_msg)
                else:
                    warn_msg = "No values found in the table that match the mask."
                    warnings.warn(warn_msg)
    except FileNotFoundError:
        warn_msg = f"Error: The file '{folder_name + file_name}' was not found.\nCurrent working directory:\n{os.getcwd()}"
        warnings.warn(warn_msg)

def fits_is_equal(
    file_name_1: str,
    file_name_2: str = "qsopar0.fits",
    folder: str = "pyqsofit_code/data/",
    ext_num: int = 1,
) -> bool:
    with fits.open(folder + file_name_1, mode='readonly') as hdul1, fits.open(folder + file_name_2, mode='readonly') as hdul2:
        data1 = hdul1[ext_num].data
        data2 = hdul2[ext_num].data
        return np.all(data1 == data2)