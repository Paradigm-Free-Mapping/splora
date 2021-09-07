"""I/O."""
from subprocess import run

import nibabel as nib
from nilearn import masking


def read_data(data_filename, mask_filename, mask_idxs=None):
    """Read data from filename and apply mask.

    Parameters
    ----------
    data_filename : str or path
        Path to data to be read.
    mask_filename : str or path
        Path to mask to be applied.

    Returns
    -------
    data_restruct : (T x S) array_like
        [description]
    data_header : nib.header
        Header of the input data.
    dims : list
        List with dimensions of data.
    mask_idxs : (S x) array_like
        Indexes to transform data back to 4D.
    """
    data_img = nib.load(data_filename)
    data_header = data_img.header
    data = data_img.get_fdata()

    mask = nib.load(mask_filename)
    data = masking.apply_mask(data_img, mask)

    return data, data_header, mask


def reshape_data(signal2d, mask):
    """Reshape data from 2D back to 4D.

    Parameters
    ----------
    signal2d : (T x S) array_like
        Data in 2D.
    mask : Nifti1Image
        Mask.

    Returns
    -------
    signal4d : (S x S x S x T) array_like
        Data in 4D.
    """
    signal4d = masking.unmask(signal2d, mask)
    return signal4d


def update_header(filename, command):
    """Update history of data to be read with 3dInfo.

    Parameters
    ----------
    filename : str or path
        Path to the file that is getting the header updated.
    command : str
        splora command to add to the header.
    """
    run(f"3dcopy {filename} {filename} -overwrite", shell=True)
    run(f'3dNotes -h "{command}" {filename}', shell=True)


def write_data(data, filename, mask, header, command):
    """Write data into NIFTI file.

    Parameters
    ----------
    data : (T x S)
        Data in 2D.
    filename : str or path
        Name of the output file.
    mask : Nifti1Image
        Mask.
    header : nib.header
        Header of the input data.
    command : str
        splora command to add to the header.
    """
    reshaped = reshape_data(data, mask)
    U_nib = nib.Nifti1Image(reshaped.get_fdata(), None, header=header)
    U_nib.to_filename(filename)
    update_header(filename, command)
