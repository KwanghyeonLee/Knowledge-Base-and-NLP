from __future__ import annotations

import re
import urllib.request
import zipfile
from functools import partial
from pathlib import Path
from urllib.parse import unquote

import paramiko
from requests import get
from scp import SCPClient, SCPException
from tqdm.rich import tqdm

from src import utils

log = utils.get_pylogger(__name__)


class DataDownloader:
    beir_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
    gpl_url = "https://public.ukp.informatik.tu-darmstadt.de/kwang/gpl/generated-data/beir/"
    gpl_main_url = "https://public.ukp.informatik.tu-darmstadt.de/kwang/gpl/generated-data/main/"
    def __init__(
        self,
        dataset_format: str,
        data_dir: str,
        data_essentials: list[str] = None,
    ):
        if dataset_format.lower() not in ["beir", "gpl", "gpl_main"]:
            raise ValueError("`dataset_format` must be either 'beir' or 'gpl'.")
        self.data_dir = Path(data_dir)
        self.data_essentials = data_essentials

        self.dataset_url = {
            "beir": self.beir_url,
            "gpl": self.gpl_url,
            "gpl_main": self.gpl_main_url,
        }[dataset_format.lower()]
        self.pbar = None


    def __check_data_exist(self, dataset_dir: Path) -> bool:
        return dataset_dir.exists() and all(
            map(lambda f: (dataset_dir / f).exists(), self.data_essentials)
        )

    @staticmethod
    def get_data_name_list(url: str) -> list[str]:
        grep_names = re.compile(r'<a href="\S+.zip">(\S+).zip</a>')
        return grep_names.findall(str(get(url).text))

    def __check_data_name_available(self, dataset_name: str) -> bool:
        data_list = self.get_data_name_list(self.dataset_url)
        return dataset_name in data_list

    def show_progress(self, desc, block_num, block_size, total_size):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, desc=desc, unit="b", unit_scale=True)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()
            self.pbar = None

    def download(self, dataset_name) -> None:
        dataset_name = dataset_name.lower()
        dataset_dir = self.data_dir / dataset_name
        if self.__check_data_exist(dataset_dir):
            log.info("Skip Dataset Downloading ...")
            return

        if not self.__check_data_name_available(dataset_name):
            raise ValueError(f"Dataset {dataset_name} is not available.")

        log.info(f"Creating Dataset Directory: {dataset_dir.absolute()}")
        dataset_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"Start Downloading Dataset {dataset_name}")
        self.__download_and_unzip(
            url=f"{self.dataset_url}/{dataset_name}.zip", target_dir=self.data_dir
        )
        log.info(f"Dataset {dataset_name} Downloaded Successfully")

    def __download_and_unzip(self, url: str, target_dir: Path):
        if not url.lower().startswith("http"):
            raise ValueError("url must start with http")
        file_name = unquote(url.split("/")[-1])
        pure_file_name = file_name.split(".zip")[0]
        zip_save_path = target_dir / file_name
        try:
            show_prog = partial(self.show_progress, f"Download {file_name}")
            urllib.request.urlretrieve(url, zip_save_path, show_prog)  # nosec
        except Exception as e:
            raise RuntimeError(f"Download failed: {file_name}")

        try:
            with zipfile.ZipFile(zip_save_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            # delete zip
            zip_save_path.unlink()
        except Exception as e:
            raise RuntimeError(f"Unzip failed: {file_name}")


class SSHDownloader:
    """
    usage:
    >>> import SSHDownloader
    >>> ssh_manager = SSHDownloader()
    >>> ssh_manager.create_ssh_client(hostname, username, password)
    >>> ssh_manager.send_file("/path/to/local_path", "/path/to/remote_path")
    >>> ssh_manager.get_file("/path/to/remote_path", "/path/to/local_path")
    ...
    >>> ssh_manager.close_ssh_client()
    """

    def __init__(self, hostname, username, password):
        self.pbar = None

        self.ssh_client = None
        # self.create_ssh_client(hostname, username, password)

    def show_progress(self, desc, filename, size, sent):

        if self.pbar is None:
            self.pbar = tqdm(total=size, desc=desc, unit="b", unit_scale=True)
        downloaded = sent
        if downloaded < size:
            self.pbar.display(sent)
        else:
            self.pbar.close()
            self.pbar = None

    def create_ssh_client(self, hostname, username, password):
        """Create SSH client session to remote server."""
        if self.ssh_client is None:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(hostname, username=username, password=password)
        else:
            print("SSH client session exist.")

    def close_ssh_client(self):
        """Close SSH client session."""
        self.ssh_client.close()
        log.info("SSH Client Closed")

    def send_file(self, local_path, remote_path):
        """Send a single file to remote path."""
        try:
            show_prog = partial(self.show_progress, "Uploading via SSH")
            with SCPClient(self.ssh_client.get_transport(), progress=show_prog) as scp:
                scp.put(local_path, remote_path, preserve_times=True)
        except SCPException:
            raise SCPException.message

    def get_file(self, remote_path, local_path):
        """Get a single file from remote path."""
        try:
            show_prog = partial(self.show_progress, "Downloading via SSH")
            with SCPClient(self.ssh_client.get_transport(), progress=show_prog) as scp:
                scp.get(remote_path, local_path)
        except SCPException:
            raise SCPException.message


if __name__ == "__main__":
    data_downloader = DataDownloader()
