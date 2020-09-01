"""I/O."""
import nibabel as nib
import numpy as np


def read_data(data_filename, mask_filename):

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
    data_restruct_temp = np.reshape(np.moveaxis(data_masked, -1, 0), (dims[-1], np.prod(data_img.shape[:-1])))
    mask_idxs = np.unique(np.nonzero(data_restruct_temp)[1])
    data_restruct = data_restruct_temp[:, mask_idxs]

    return data_restruct, data_header, dims, mask_idxs


def reshape_data(signal2d, dims, mask_idxs):
    signal4d = np.zeros((dims[0] * dims[1] * dims[2], signal2d.shape[0]))

    # Merges signal on mask indices with blank image
    for i in range(dims[3]):
        if len(mask_idxs.shape) > 3:
            idxs = np.where(mask_idxs[:, :, :, i] != 0)
        else:
            idxs = mask_idxs

        signal4d[idxs, i] = signal2d[i, :]

    # Reshapes matrix from 2D to 4D double
    signal4d = np.reshape(signal4d, (dims[0], dims[1], dims[2], signal2d.shape[0]))
    del signal2d, idxs, dims, mask_idxs
    return(signal4d)
