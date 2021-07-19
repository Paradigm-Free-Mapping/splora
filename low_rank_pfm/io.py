"""I/O."""
import nibabel as nib
import numpy as np
from subprocess import run
from nilearn._utils import check_niimg
from nilearn.image import new_img_like


def read_data(data_filename, mask_filename):
    """
    Read files.

    Args:
        data_filename ([type]): [description]
        mask_filename ([type]): [description]

    Returns:
        [type]: [description]
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
    """
    Reshape data from 2D to 4D.

    Args:
        signal2d ([type]): [description]
        dims ([type]): [description]
        mask_idxs ([type]): [description]
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


def update_history(filename, command):
    """
    Update file history for 3dinfo.

    Args:
        filename ([type]): [description]
        command ([type]): [description]
    """
    # run(f"3dcopy {filename} {filename} -overwrite", shell=True)
    run(f'3dNotes -h "{command}" {filename}', shell=True)


def new_nii_like(ref_img, data, affine=None, copy_header=True):
    """
    Coerces `data` into NiftiImage format like `ref_img`
    Parameters
    ----------
    ref_img : :obj:`str` or img_like
        Reference image
    data : (S [x T]) array_like
        Data to be saved
    affine : (4 x 4) array_like, optional
        Transformation matrix to be used. Default: `ref_img.affine`
    copy_header : :obj:`bool`, optional
        Whether to copy header from `ref_img` to new image. Default: True
    Returns
    -------
    nii : :obj:`nibabel.nifti1.Nifti1Image`
        NiftiImage
    """

    ref_img = check_niimg(ref_img)
    newdata = data.reshape(ref_img.shape[:3] + data.shape[1:])
    if ".nii" not in ref_img.valid_exts:
        # this is rather ugly and may lose some information...
        nii = nib.Nifti1Image(newdata, affine=ref_img.affine, header=ref_img.header)
    else:
        # nilearn's `new_img_like` is a very nice function
        nii = new_img_like(ref_img, newdata, affine=affine, copy_header=copy_header)
    nii.set_data_dtype(data.dtype)

    return nii
