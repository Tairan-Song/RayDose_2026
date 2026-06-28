"""Minimal MetaImage (.mha) IO utilities for DoseRAD dose/mask files.

This module intentionally keeps dependencies small. It supports the dataset's
LOCAL binary MHA files, including zlib/deflate-compressed payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zlib

import numpy as np


_DTYPES = {
    "MET_FLOAT": np.float32,
    "MET_UCHAR": np.uint8,
}


@dataclass
class MHAImage:
    array: np.ndarray
    meta: dict[str, str]


def _parse_header(header: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


def _dim_size(meta: dict[str, str]) -> tuple[int, int, int]:
    values = tuple(int(v) for v in meta["DimSize"].split())
    if len(values) != 3:
        raise ValueError(f"Expected 3D DimSize, got {meta['DimSize']!r}")
    return values


def _decompress_payload(payload: bytes) -> bytes:
    # Some MHA writers store raw deflate, while others include a zlib header.
    for wbits in (-zlib.MAX_WBITS, zlib.MAX_WBITS):
        try:
            return zlib.decompress(payload, wbits)
        except zlib.error:
            pass
    raise ValueError("Could not decompress MHA payload")


def read_mha(path: str | Path) -> MHAImage:
    path = Path(path)
    data = path.read_bytes()
    marker = b"ElementDataFile = LOCAL"
    marker_idx = data.find(marker)
    if marker_idx < 0:
        raise ValueError(f"Only LOCAL MHA files are supported: {path}")
    header_end = data.find(b"\n", marker_idx)
    if header_end < 0:
        raise ValueError(f"Invalid MHA header: {path}")
    header_end += 1

    header_text = data[:header_end].decode("ascii")
    meta = _parse_header(header_text)
    payload = data[header_end:]

    compressed = meta.get("CompressedData", "False").lower() == "true"
    raw = _decompress_payload(payload) if compressed else payload

    element_type = meta["ElementType"]
    if element_type not in _DTYPES:
        raise ValueError(f"Unsupported ElementType {element_type!r}")
    dtype = _DTYPES[element_type]
    dims = _dim_size(meta)

    array = np.frombuffer(raw, dtype=dtype)
    expected = dims[0] * dims[1] * dims[2]
    if array.size != expected:
        raise ValueError(f"Payload voxel count {array.size} does not match DimSize {dims}")

    # Keep MetaImage axis order as stored: x, y, z flattened.
    return MHAImage(array=array.reshape(dims, order="C"), meta=meta)


def write_mask_mha(path: str | Path, mask: np.ndarray, reference_meta: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mask = np.asarray(mask, dtype=np.uint8)
    compressed = zlib.compress(mask.ravel(order="C").tobytes(), level=6)

    header_keys = [
        "ObjectType",
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "TransformMatrix",
        "Offset",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "DimSize",
    ]
    lines = []
    for key in header_keys:
        if key in reference_meta:
            lines.append(f"{key} = {reference_meta[key]}")
    if "BinaryData" not in reference_meta:
        lines.append("BinaryData = True")
    if "BinaryDataByteOrderMSB" not in reference_meta:
        lines.append("BinaryDataByteOrderMSB = False")
    lines.extend(
        [
            "CompressedData = True",
            f"CompressedDataSize = {len(compressed)}",
            "ElementType = MET_UCHAR",
            "ElementDataFile = LOCAL",
        ]
    )

    path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + compressed)


def write_float_mha(
    path: str | Path,
    array: np.ndarray,
    reference_meta: dict[str, str],
    offset: np.ndarray | None = None,
    dim_size: tuple[int, int, int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(array, dtype=np.float32)
    compressed = zlib.compress(array.ravel(order="C").tobytes(), level=6)

    meta = dict(reference_meta)
    if offset is not None:
        meta["Offset"] = " ".join(f"{float(v):.6g}" for v in offset)
    if dim_size is not None:
        meta["DimSize"] = " ".join(str(int(v)) for v in dim_size)

    header_keys = [
        "ObjectType",
        "NDims",
        "BinaryData",
        "BinaryDataByteOrderMSB",
        "TransformMatrix",
        "Offset",
        "CenterOfRotation",
        "AnatomicalOrientation",
        "ElementSpacing",
        "DimSize",
    ]
    lines = []
    for key in header_keys:
        if key in meta:
            lines.append(f"{key} = {meta[key]}")
    if "ObjectType" not in meta:
        lines.append("ObjectType = Image")
    if "NDims" not in meta:
        lines.append("NDims = 3")
    if "BinaryData" not in meta:
        lines.append("BinaryData = True")
    if "BinaryDataByteOrderMSB" not in meta:
        lines.append("BinaryDataByteOrderMSB = False")
    lines.extend(
        [
            "CompressedData = True",
            f"CompressedDataSize = {len(compressed)}",
            "ElementType = MET_FLOAT",
            "ElementDataFile = LOCAL",
        ]
    )

    path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + compressed)
