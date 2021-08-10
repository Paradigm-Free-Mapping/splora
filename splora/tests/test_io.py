import os

import nibabel as nib
import numpy as np

from splora import io
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()


def test_read_data():
    data_filename = os.path.join(data_dir, "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz")
    mask_filename = os.path.join(data_dir, "mask.nii.gz")
    data_check = np.load(os.path.join(data_dir, "data.npy"))
    data, header, mask = io.read_data(data_filename, mask_filename)

    assert np.allclose(data, data_check)
    assert isinstance(header, nib.Nifti1Header)
    assert isinstance(mask, nib.Nifti1Image)
    assert mask.shape == (41, 52, 28)


def test_reshape_data():
    data = np.load(os.path.join(data_dir, "data.npy"))
    mask = nib.load(os.path.join(data_dir, "mask.nii.gz"))
    data4d = io.reshape_data(data, mask)
    assert data4d.shape == (41, 52, 28, 160)


def test_write_data():
    data_file = os.path.join(data_dir, "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz")
    data = np.load(os.path.join(data_dir, "data.npy"))
    mask = nib.load(os.path.join(data_dir, "mask.nii.gz"))
    filename = os.path.join(data_dir, "test.nii.gz")
    header = nib.load(data_file).header
    command = "test"
    io.write_data(data, filename, mask, header, command)
    assert "test.nii.gz" in os.listdir(data_dir)
    os.remove(filename)