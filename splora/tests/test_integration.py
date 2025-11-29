import os
import shutil

import pytest

from splora import splora
from splora.tests.utils import get_test_data_path

data_dir = get_test_data_path()


def test_integration_single_echo(skip_integration):
    if skip_integration:
        pytest.skip("Skipping five-echo integration test")

    single_echo_files = [
        "test_lambda.nii.gz",
        "call.sh",
        "test_eigenmap_1.nii.gz",
        "test_innovation.nii.gz",
        "test_eigenvec_1_E01.1D",
        "_references.txt",
        "test_fitts.nii.gz",
        "test_beta.nii.gz",
        "test_MAD_E01.nii.gz",
    ]

    os.chdir(data_dir)
    args = [
        "-i",
        "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz",
        "-o",
        "test",
        "-m",
        "mask_tiny.nii.gz",
        "-tr",
        "2",
        "-crit",
        "mad_update",
        "--dir",
        "single_echo",
        "--debias",
        "--block",
        "-max_iter",
        "2",
        "-min_iter",
        "1",
    ]
    splora._main(args)

    files = os.listdir(os.path.join(data_dir, "single_echo"))
    for file in single_echo_files:
        assert file in files

    logfile = [i for i in files if "splora_" in i]
    assert logfile[0] in files

    shutil.rmtree(os.path.join(data_dir, "single_echo"))


def test_integration_multi_echo(skip_integration):
    if skip_integration:
        pytest.skip("Skipping five-echo integration test")

    data = [
        "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz",
        "p06.SBJ01_S09_Task11_e3.spc.det.nii.gz",
    ]
    mask = "mask_tiny.nii.gz"
    te = [29.7, 44.0]

    os.chdir(data_dir)
    splora.splora(
        data_filename=data,
        mask_filename=mask,
        tr=2,
        output_filename="test",
        out_dir="multi_echo",
        te=te,
        group=0.2,
        do_debias=False,
        block_model=False,
        lambda_crit="mad_update",
        max_iter=2,
        min_iter=1,
    )

    multi_echo_files = [
        "call.sh",
        "_references.txt",
        "test_DR2.nii.gz",
        "test_lambda.nii.gz",
        "test_eigenmap_1.nii.gz",
        "test_eigenvec_1_E01.1D",
        "test_eigenvec_1_E02.1D",
        "test_dr2HRF_E01.nii.gz",
        "test_dr2HRF_E02.nii.gz",
        "test_MAD_E01.nii.gz",
        "test_MAD_E02.nii.gz",
    ]

    files = os.listdir(os.path.join(data_dir, "multi_echo"))
    for file in multi_echo_files:
        assert file in files

    logfile = [i for i in files if "splora_" in i]
    assert logfile[0] in files

    shutil.rmtree(os.path.join(data_dir, "multi_echo"))
