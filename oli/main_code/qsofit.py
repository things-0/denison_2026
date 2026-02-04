import numpy as np
from typing import Any
import os
import warnings
from astropy.io import fits

from .constants import *
from .helpers import get_new_filename

def edit_qsopar(
    value_name_to_edit: str,
    new_value: Any,
    col_name_and_val: tuple[str, Any] | None, # ("linename", "Ha_na"), ("maxsig", 0.001690)
    mask: np.ndarray | None = None,
    filename: str = "data/qsofit/qsopar2.fits",
    ext_num: int = 1,
    make_copy: bool = True,
    print_change: bool = True
) -> None:
    try:
        with fits.open(filename, mode='update' if make_copy else 'readonly') as hdul:

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
                        warn_msg = f"\nValue '{value_name_to_edit}' for {col_name} = {col_name_val} is already {new_value}. No changes made."
                    else:
                        warn_msg = f"\nValue '{value_name_to_edit}' is already {new_value}. No changes made."
                    warnings.warn(warn_msg)
                else:
                    data[value_name_to_edit][mask] = new_value
                    new_filename = get_new_filename(filename)
                    if print_change:
                        if col_name_and_val is not None:
                            print(f"{value_name_to_edit} for {col_name} = {col_name_val} updated successfully from {old_val} to {new_value}.")
                        else:
                            print(f"{value_name_to_edit} updated successfully from {old_val} to {new_value}.")
                        if make_copy:
                            print(f"Changes saved to new file: {new_filename}")
                        else:
                            print(f"Changes saved to original file: {filename}")
                    if make_copy:
                        hdul.writeto(new_filename, overwrite=True)
            else:
                if col_name_and_val is not None:
                    warnings.warn(f"{col_name} = {col_name_val} not found in the table.")
                else:
                    warnings.warn("No values found in the table that match the mask.")
    except FileNotFoundError:
        warn_msg = f"\nError: The file '{filename}' was not found.\nCurrent working directory:\n{os.getcwd()}"
        warnings.warn(warn_msg)
