"""I/O."""
from subprocess import run

import nibabel as nib
import numpy as np


def read_data(data_filename, mask_filename):
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
    dims = data.shape

    mask = nib.load(mask_filename).get_fdata()
    data_masked = np.zeros((dims[0], dims[1], dims[2], dims[3]))

    # Masks data
    if len(mask.shape) < 4:
        for i in range(dims[-1]):
            data_masked[:, :, :, i] = np.squeeze(data[:, :, :, i]) * mask
    else:
        data_masked = data * mask

    # Initiates data_restruct to make loop faster
    data_restruct_temp = np.reshape(
        np.moveaxis(data_masked, -1, 0), (dims[-1], np.prod(data_img.shape[:-1]))
    )
    mask_idxs = np.unique(np.nonzero(data_restruct_temp)[1])
    data_restruct = data_restruct_temp[:, mask_idxs]

    return data_restruct, data_header, dims, mask_idxs


def reshape_data(signal2d, dims, mask_idxs):
    """Reshape data from 2D back to 4D.

    Parameters
    ----------
    signal2d : (T x S) array_like
        Data in 2D.
    dims : list
        List with dimensions of data.
    mask_idxs : (S x) array_like
        Indexes to transform data back to 4D.

    Returns
    -------
    signal4d : (S x S x S x T) array_like
        Data in 4D.
    """
    signal4d = np.zeros((dims[0] * dims[1] * dims[2], signal2d.shape[0]))
    idxs = 0

    # Merges signal on mask indices with blank image
    for i in range(signal2d.shape[0]):
        if len(mask_idxs.shape) > 3:
            idxs = np.where(mask_idxs[:, :, :, i] != 0)
        else:
            idxs = mask_idxs

        signal4d[idxs, i] = signal2d[i, :]

    # Reshapes matrix from 2D to 4D double
    signal4d = np.reshape(signal4d, (dims[0], dims[1], dims[2], signal2d.shape[0]))
    del signal2d, idxs, dims, mask_idxs
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


def write_data(data, filename, dims, idxs, header, command):
    """Write data into NIFTI file.

    Parameters
    ----------
    data : (T x S)
        Data in 2D.
    filename : str or path
        Name of the output file.
    dims : list
        List with dimensions of data.
    idxs : (S x) array_like
        Indexes to transform data back to 4D.
    header : nib.header
        Header of the input data.
    command : str
        splora command to add to the header.
    """
    reshaped = reshape_data(data, dims, idxs)
    U_nib = nib.Nifti1Image(reshaped, None, header=header)
    U_nib.to_filename(filename)
    update_header(filename, command)
