"""Direct dataset fetchers that bypass the slow canonical mirrors.

torchvision's default download sources for MNIST and CIFAR-10 (LeCun's site,
``ossci-datasets.s3``, and ``cs.toronto.edu``) throttle heavily -- measured at
roughly 15 kB/s from this network, which puts CIFAR-10 at several hours.  The
mirrors used here serve byte-identical content two orders of magnitude faster:

* MNIST -- Google Cloud ``cvdf-datasets`` mirror of the original IDX files.
* Fashion-MNIST -- Google Cloud ``tf-keras-datasets``, the source Keras itself
  uses.
* CIFAR-10 -- the Hugging Face ``uoft-cs/cifar10`` Parquet export.

Each fetcher parses the raw format directly and caches the decoded arrays as a
single ``.npz`` per dataset, so the parsing cost is paid once.  Integrity is
established downstream by the split digests in
:func:`shiftbench.datasets.split_digest`, which compare the actual pixel bytes
rather than a download checksum, and therefore catch a wrong-but-well-formed
mirror as well as a corrupted one.
"""

from __future__ import annotations

import gzip
import io
import os
import struct
import urllib.request
from typing import Tuple

import numpy as np

__all__ = ["fetch_mnist", "fetch_fashion_mnist", "fetch_cifar10", "fetch"]

_MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FASHION_BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
_CIFAR_BASE = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
)

_IDX_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _get(url: str, dest: str, chunk: int = 1 << 20, attempts: int = 5) -> str:
    """Download ``url`` to ``dest``, resuming and retrying on a short read.

    A plain read loop that stops at the first empty block treats a dropped
    connection as a completed download and writes a truncated file, which then
    fails much later with an opaque decompression error.  This observed
    failure mode -- a file ending at exactly one chunk boundary -- is why the
    transfer is validated against ``Content-Length`` and resumed with a Range
    request rather than restarted.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    name = os.path.basename(dest)

    for attempt in range(1, attempts + 1):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers = {"User-Agent": "shiftbench/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                # With a Range request the server reports only the remaining
                # bytes, so the expected total is what we already have plus
                # what it is about to send.
                remaining = int(r.headers.get("Content-Length") or 0)
                total = have + remaining if remaining else 0
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(tmp, mode) as fh:
                    while True:
                        block = r.read(chunk)
                        if not block:
                            break
                        fh.write(block)
                        have += len(block)
                        if total:
                            print(f"\r    {name}: {100.0 * have / total:5.1f}% "
                                  f"({have >> 20}/{total >> 20} MiB)",
                                  end="", flush=True)
            if total:
                print()
            if total and os.path.getsize(tmp) < total:
                raise IOError(
                    f"short read: {os.path.getsize(tmp)} of {total} bytes"
                )
        except Exception as exc:  # network error, short read, or timeout
            if attempt == attempts:
                raise
            print(f"\n    {name}: attempt {attempt} failed ({exc}); resuming")
            continue
        os.replace(tmp, dest)
        return dest
    raise IOError(f"could not download {url}")  # pragma: no cover


def _read_idx(path: str) -> np.ndarray:
    """Parse an IDX file (the MNIST binary format), gzip-compressed."""
    with gzip.open(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        # Low byte of the magic number is the number of dimensions.
        ndim = magic & 0xFF
        shape = struct.unpack(f">{ndim}I", fh.read(4 * ndim))
        data = np.frombuffer(fh.read(), dtype=np.uint8)
    return data.reshape(shape)


def _fetch_idx_pair(base: str, cache_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    raw = os.path.join(cache_dir, tag, "raw")
    paths = [_get(base + f, os.path.join(raw, f)) for f in _IDX_FILES]
    xtr, ytr, xte, yte = (_read_idx(p) for p in paths)
    x = np.concatenate([xtr, xte], axis=0)[..., None]  # (N, 28, 28, 1)
    y = np.concatenate([ytr, yte], axis=0)
    return x.astype(np.uint8), y.astype(np.int64)


def fetch_mnist(cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    return _fetch_idx_pair(_MNIST_BASE, cache_dir, "mnist")


def fetch_fashion_mnist(cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    return _fetch_idx_pair(_FASHION_BASE, cache_dir, "fashion_mnist")


def fetch_cifar10(cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fetch CIFAR-10 from the Hugging Face Parquet export.

    The export stores each image as PNG-encoded bytes in a struct column, so
    decoding is required.  PNG is lossless, so the decoded pixels are identical
    to the original binary batches.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    raw = os.path.join(cache_dir, "cifar10", "raw")
    files = [
        ("train-00000-of-00001.parquet", "train"),
        ("test-00000-of-00001.parquet", "test"),
    ]
    xs, ys = [], []
    for fname, _ in files:
        path = _get(_CIFAR_BASE + fname, os.path.join(raw, fname))
        table = pq.read_table(path)
        cols = table.column_names
        img_col = "img" if "img" in cols else "image"
        lbl_col = "label" if "label" in cols else "fine_label"
        images = table.column(img_col).to_pylist()
        labels = np.asarray(table.column(lbl_col).to_pylist(), dtype=np.int64)
        arr = np.empty((len(images), 32, 32, 3), dtype=np.uint8)
        for i, rec in enumerate(images):
            data = rec["bytes"] if isinstance(rec, dict) else rec
            arr[i] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        xs.append(arr)
        ys.append(labels)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


_FETCHERS = {
    "mnist": fetch_mnist,
    "fashion_mnist": fetch_fashion_mnist,
    "cifar10": fetch_cifar10,
}


def fetch(name: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for ``name``, downloading and caching as needed.

    The decoded arrays are memoised as ``<cache_dir>/<name>/<name>.npz`` so the
    PNG decoding and IDX parsing happen once rather than on every run.
    """
    if name not in _FETCHERS:
        raise ValueError(f"no fast fetcher for {name!r}")
    npz = os.path.join(cache_dir, name, f"{name}.npz")
    if os.path.exists(npz):
        with np.load(npz) as z:
            return z["x"], z["y"]
    print(f"  fetching {name} ...")
    x, y = _FETCHERS[name](cache_dir)
    os.makedirs(os.path.dirname(npz), exist_ok=True)
    np.savez_compressed(npz, x=x, y=y)
    print(f"  cached {name}: x={x.shape} y={y.shape}")
    return x, y
