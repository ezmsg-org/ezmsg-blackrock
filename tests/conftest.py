"""Shared test fixtures for ezmsg-blackrock."""

import ctypes
import ctypes.util
import os
import platform
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve

import pytest

CERELINK_RELEASE_URL = "https://github.com/CerebusOSS/CereLink/releases/download/v9.3.0"
CACHE_DIR = Path(__file__).parent / ".test_cache"

_lock_counter = 0


def _sem_unlink(name: str) -> None:
    """Unlink a POSIX named semaphore via libc sem_unlink."""
    if platform.system() == "Windows":
        return
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        libc_name = "libc.dylib" if sys.platform == "darwin" else "libc.so.6"
    libc = ctypes.CDLL(libc_name, use_errno=True)
    libc.sem_unlink.argtypes = [ctypes.c_char_p]
    libc.sem_unlink.restype = ctypes.c_int
    sem_name = f"/{name}".encode()
    libc.sem_unlink(sem_name)  # ignore errors (ENOENT is fine)


def _nplay_asset_name() -> str | None:
    """Return the nPlayServer zip asset name for this platform, or None."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        return "nPlayServer-MacOS-arm.zip"
    elif system == "Windows":
        return "nPlayServer-Win64.zip"
    elif system == "Linux" and machine in ("x86_64", "amd64"):
        return "nPlayServer-Linux-x64.zip"
    return None


def _download(url: str, dest: Path) -> None:
    """Download url to dest, skipping if dest already exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    urlretrieve(url, dest)


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip_path into dest_dir, skipping if dest_dir already exists."""
    if dest_dir.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def _download_dataset(name: str) -> Path:
    """Download and extract a dataset zip, return the extraction directory."""
    zip_path = CACHE_DIR / f"{name}.zip"
    extract_dir = CACHE_DIR / name
    _download(f"{CERELINK_RELEASE_URL}/{name}.zip", zip_path)
    _extract_zip(zip_path, extract_dir)
    return extract_dir


@contextmanager
def run_nplayserver(binary: Path, ns6: Path):
    """Context manager: start nPlayServer, yield the process, kill on exit."""
    global _lock_counter
    _lock_counter += 1
    lock_name = f"nplay_test_{os.getpid()}_{_lock_counter}"

    proc = subprocess.Popen(
        [
            str(binary),
            "--audio",
            "none",
            "--lockfile",
            lock_name,
            "-A",
            str(ns6),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(3)

    if proc.poll() is not None:
        stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        pytest.fail(f"nPlayServer exited immediately (rc={proc.returncode})\n" f"stdout: {stdout}\nstderr: {stderr}")

    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _sem_unlink(lock_name)


# -- Data path fixtures (session-scoped) --------------------------------------


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Download and extract 4-channel test data (dnss.zip)."""
    return _download_dataset("dnss")


@pytest.fixture(scope="session")
def test_data_dir_256() -> Path:
    """Download and extract 256-channel test data (dnss256.zip)."""
    return _download_dataset("dnss256")


@pytest.fixture(scope="session")
def ccf_path(test_data_dir: Path) -> Path:
    """Path to the .ccf file in the 4-channel test data."""
    matches = list(test_data_dir.rglob("*.ccf"))
    assert matches, "No .ccf file found in test data"
    return matches[0]


@pytest.fixture(scope="session")
def ccf_256_path(test_data_dir_256: Path) -> Path:
    """Path to the .ccf file in the 256-channel test data."""
    matches = list(test_data_dir_256.rglob("*.ccf"))
    assert matches, "No .ccf file found in 256-channel test data"
    return matches[0]


@pytest.fixture(scope="session")
def cmp_path() -> Path:
    """Path to the bundled 96-channel CMP file."""
    p = Path(__file__).parent / "96ChannelDefaultMapping.cmp"
    assert p.exists(), f"CMP file not found at {p}"
    return p


@pytest.fixture(scope="session")
def ns6_path(test_data_dir: Path) -> Path:
    """Path to the .ns6 file in the 4-channel test data."""
    matches = list(test_data_dir.rglob("*.ns6"))
    assert matches, "No .ns6 file found in test data"
    return matches[0]


@pytest.fixture(scope="session")
def ns6_256_path(test_data_dir_256: Path) -> Path:
    """Path to the .ns6 file in the 256-channel test data."""
    matches = list(test_data_dir_256.rglob("*.ns6"))
    assert matches, "No .ns6 file found in 256-channel test data"
    return matches[0]


@pytest.fixture(scope="session")
def nplayserver_binary() -> Path | None:
    """Locate or download the nPlayServer binary for this platform."""
    env_path = os.environ.get("NPLAYSERVER_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        pytest.fail(f"NPLAYSERVER_PATH={env_path} does not exist")

    asset_name = _nplay_asset_name()
    if asset_name is None:
        return None

    zip_path = CACHE_DIR / asset_name
    extract_dir = CACHE_DIR / asset_name.removesuffix(".zip")
    _download(f"{CERELINK_RELEASE_URL}/{asset_name}", zip_path)
    _extract_zip(zip_path, extract_dir)

    exe_suffix = ".exe" if platform.system() == "Windows" else ""
    for candidate in extract_dir.rglob(f"nPlayServer{exe_suffix}"):
        if candidate.is_file():
            if platform.system() != "Windows":
                candidate.chmod(0o755)
            return candidate
    return None
