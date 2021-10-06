import os
import shutil

import numpy as np
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
        "test_eigenvec_1.1D",
        "_references.txt",
        "test_fitts.nii.gz",
        "test_beta.nii.gz",
        "test_MAD.nii.gz",
    ]

    os.chdir(data_dir)
    args = [
        "-i",
        "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz",
        "-o",
        "test",
        "-m",
        "mask.nii.gz",
        "-tr",
        "2",
        "-crit",
        "mad_update",
        "--dir",
        "single_echo",
        "--debias",
        "--block",
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
        "p06.SBJ01_S09_Task11_e1.spc.det.nii.gz",
        "p06.SBJ01_S09_Task11_e2.spc.det.nii.gz",
        "p06.SBJ01_S09_Task11_e3.spc.det.nii.gz",
        "p06.SBJ01_S09_Task11_e4.spc.det.nii.gz",
    ]
    mask = "mask.nii.gz"
    te = [15.4, 29.7, 44.0, 58.3]

    os.chdir(data_dir)
    splora.splora(
        data_filename=data,
        mask_filename=mask,
        tr=2,
        output_filename="test",
        out_dir="multi_echo",
        te=te,
        group=0.2,
        do_debias=True,
        block_model=False,
        lambda_crit="mad_update",
    )

    multi_echo_files = [
        "test_fitts_E01.nii.gz",
        "test_eigenvec_1.1D",
        "test_fitts_E03.nii.gz",
        "call.sh",
        "test_eigenmap_2.nii.gz",
        "test_MAD.nii.gz",
        "test_beta.nii.gz",
        "test_lambda.nii.gz",
        "test_fitts_E02.nii.gz",
        "_references.txt",
        "test_eigenvec_3.1D",
        "test_eigenmap_3.nii.gz",
        "test_fitts_E04.nii.gz",
        "test_eigenmap_1.nii.gz",
        "test_eigenvec_2.1D",
    ]

    files = os.listdir(os.path.join(data_dir, "multi_echo"))
    for file in multi_echo_files:
        assert file in files

    logfile = [i for i in files if "splora_" in i]
    assert logfile[0] in files

    shutil.rmtree(os.path.join(data_dir, "multi_echo"))
