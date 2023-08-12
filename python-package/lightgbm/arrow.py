# coding: utf-8
"""Utilities for handling Arrow in LightGBM."""
from contextlib import contextmanager
from typing import Iterator, List

import pyarrow as pa
from pyarrow.cffi import ffi

__all__: List[str] = []


class _ArrowCArray:
    """Simple wrapper around the C representation of an Arrow type."""

    n_chunks: int
    chunks: ffi.CData
    schema: ffi.CData

    def __init__(self, n_chunks: int, chunks: ffi.CData, schema: ffi.CData):
        self.n_chunks = n_chunks
        self.chunks = chunks
        self.schema = schema

    @property
    def chunks_ptr(self) -> int:
        """Returns the address of the pointer to the list of chunks making up the array."""
        return int(ffi.cast("uintptr_t", ffi.addressof(self.chunks[0])))

    @property
    def schema_ptr(self) -> int:
        """Returns the address of the pointer to the schema of the array."""
        return int(ffi.cast("uintptr_t", self.schema))


@contextmanager
def _export_arrow_to_c(data: pa.Table) -> Iterator[_ArrowCArray]:
    """Export an Arrow type to its C representation."""
    # Obtain objects to export
    if isinstance(data, pa.Table):
        export_objects = data.to_batches()
    else:
        raise ValueError(f"data of type '{type(data)}' cannot be exported to Arrow")

    # Prepare export
    chunks = ffi.new(f"struct ArrowArray[{len(export_objects)}]")
    schema = ffi.new("struct ArrowSchema*")

    # Export all objects
    for i, obj in enumerate(export_objects):
        chunk_ptr = int(ffi.cast("uintptr_t", ffi.addressof(chunks[i])))
        if i == 0:
            schema_ptr = int(ffi.cast("uintptr_t", schema))
            obj._export_to_c(chunk_ptr, schema_ptr)
        else:
            obj._export_to_c(chunk_ptr)

    yield _ArrowCArray(len(chunks), chunks, schema)
