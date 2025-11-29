
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skipintegration", action="store_true", default=False, help="Skip integration tests."
    )


@pytest.fixture
def skip_integration(request):
    return request.config.getoption("--skipintegration")


# def fetch_file(osf_id, path, filename):
#     """
#     Fetches file located on OSF and downloads to `path`/`filename`1
#     Parameters
#     ----------
#     osf_id : str
#         Unique OSF ID for file to be downloaded. Will be inserted into relevant
#         location in URL: https://osf.io/{osf_id}/download
#     path : str
#         Path to which `filename` should be downloaded. Ideally a temporary
#         directory
#     filename : str
#         Name of file to be downloaded (does not necessarily have to match name
#         of file on OSF)
#     Returns
#     -------
#     full_path : str
#         Full path to downloaded `filename`
#     """
#     # This restores the same behavior as before.
#     # this three lines make tests dowloads work in windows
#     if os.name == "nt":
#         orig_sslsocket_init = ssl.SSLSocket.__init__
#         ssl.SSLSocket.__init__ = (
#             lambda *args, cert_reqs=ssl.CERT_NONE, **kwargs: orig_sslsocket_init(
#                 *args, cert_reqs=ssl.CERT_NONE, **kwargs
#             )
#         )
#         ssl._create_default_https_context = ssl._create_unverified_context
#     url = "https://osf.io/{}/download".format(osf_id)
#     full_path = os.path.join(path, filename)
#     if not os.path.isfile(full_path):
#         urlretrieve(url, full_path)
#     return full_path


# def download_test_data(osf_id, outpath):
#     """
#     Downloads tar.gz data stored at `osf` and unpacks into `outpath`
#     Parameters
#     ----------
#     osf : str
#         URL to OSF file that contains data to be downloaded
#     outpath : str
#         Path to directory where OSF data should be extracted
#     """
#     osf = "https://osf.io/{}/download".format(osf_id)
#     req = requests.get(osf)
#     req.raise_for_status()
#     t = tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content)))
#     os.makedirs(outpath, exist_ok=True)
#     t.extractall(outpath)


# @pytest.fixture(scope="session")
# def testpath(tmp_path_factory):
#     """ Test path that will be used to download all files """
#     return tmp_path_factory.getbasetemp()


# @pytest.fixture
# def hrf_matrix_test(testpath):
#     download_test_data("g3mfj", testpath)


# @pytest.fixture
# def voxel_visual_task(testpath):
#     download_test_data("5zsnk", testpath)
