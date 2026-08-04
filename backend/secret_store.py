from __future__ import annotations

import ctypes
import os
import uuid
from ctypes import wintypes
from pathlib import Path


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


class SecretStore:
    """Small Windows-DPAPI store. Plain provider keys never enter SQLite."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _blob(data: bytes) -> tuple[_DataBlob, ctypes.Array[ctypes.c_char]]:
        buffer = ctypes.create_string_buffer(data)
        return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte))), buffer

    @staticmethod
    def _protect(data: bytes) -> bytes:
        if os.name != "nt":
            raise RuntimeError("Persistent provider credentials require Windows DPAPI")
        source, keepalive = SecretStore._blob(data)
        output = _DataBlob()
        if not ctypes.windll.crypt32.CryptProtectData(
            ctypes.byref(source), "ResearchLab", None, None, None, 0, ctypes.byref(output)
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output.pbData)
            del keepalive

    @staticmethod
    def _unprotect(data: bytes) -> bytes:
        if os.name != "nt":
            raise RuntimeError("Persistent provider credentials require Windows DPAPI")
        source, keepalive = SecretStore._blob(data)
        output = _DataBlob()
        if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(source), None, None, None, None, 0, ctypes.byref(output)
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output.pbData)
            del keepalive

    def put(self, value: str) -> str:
        reference = str(uuid.uuid4())
        path = self.directory / f"{reference}.bin"
        path.write_bytes(self._protect(value.encode("utf-8")))
        return reference

    def get(self, reference: str) -> str:
        path = self.directory / f"{reference}.bin"
        return self._unprotect(path.read_bytes()).decode("utf-8")

    def delete(self, reference: str) -> None:
        path = self.directory / f"{reference}.bin"
        if path.exists():
            path.unlink()
