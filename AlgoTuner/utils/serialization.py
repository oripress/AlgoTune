import json
import logging
import numpy as np
import base64
import os
import copy
import traceback
import functools
import inspect
from enum import Enum
from typing import Any, Dict, List, Union, Type, Optional
from scipy import sparse
import uuid
from pathlib import Path

# Size threshold (in bytes) for storing numpy arrays inline vs. externally
NDARRAY_INLINE_THRESHOLD_BYTES = 100 * 1024  # 100 KB

# Size threshold for storing bytes objects inline vs. externally
BYTES_INLINE_THRESHOLD_BYTES = 100 * 1024  # 100 KB


def _is_large_ndarray(obj: Any) -> bool:
    """Return *True* if *obj* is a NumPy array larger than the inline threshold."""
    return isinstance(obj, np.ndarray) and obj.nbytes > NDARRAY_INLINE_THRESHOLD_BYTES


def _is_large_bytes(obj: Any) -> bool:
    """Return *True* if *obj* is a bytes object larger than the inline threshold."""
    return isinstance(obj, bytes) and len(obj) > BYTES_INLINE_THRESHOLD_BYTES


# JSON encoder
class DatasetEncoder(json.JSONEncoder):
    """Encoder that preserves rich Python / NumPy / SciPy objects."""

    # Fallbacks for non‑JSON primitives
    def default(self, obj):
        # numpy.bool_ → bool
        try:
            if isinstance(obj, np.bool_):
                return bool(obj)
        except Exception:
            pass

        # Python integers that might be too large for standard JSON number types
        if isinstance(obj, int):
            # Check if the integer is outside the 64-bit signed integer range
            # (-(2**63) to (2**63)-1)
            # orjson typically handles up to 64-bit unsigned, but being conservative
            # with signed 64-bit is safer for broader compatibility if other JSON
            # tools consume this output.
            # Python integers have arbitrary precision, so direct serialization can fail.
            if obj < -9223372036854775808 or obj > 9223372036854775807:
                return str(obj) # Convert to string if too large
            # Otherwise, let it be handled by subsequent checks or super().default
            # if it's a standard-sized int that orjson can handle as a number.

        # Bytes → base64 string
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "data_b64": base64.b64encode(obj).decode("ascii"),
            }

        # NumPy arrays
        if isinstance(obj, np.ndarray):
            if _is_large_ndarray(obj):
                # Large arrays are expected to be handled by the
                # _process_placeholders pre‑pass.  If we get here we
                # fallback to a (slow) list representation but log it.
                logging.error(
                    "DatasetEncoder encountered a large ndarray! Pre‑processing "
                    "did not externalise it as expected. Falling back to list "
                    "encoding (very slow)."
                )
                return {
                    "__type__": "ndarray",
                    "class": "numpy.ndarray",
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
            # Small array → base64 inline
            try:
                data_b64 = base64.b64encode(obj.tobytes()).decode("ascii")
                return {
                    "__type__": "ndarray_b64",
                    "class": "numpy.ndarray",
                    "data_b64": data_b64,
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
            except Exception as exc:
                logging.error(
                    "Base64 encoding failed for ndarray (%s). Falling back to list.",
                    exc,
                )
                return {
                    "__type__": "ndarray",
                    "class": "numpy.ndarray",
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }

        # Scalar NumPy numbers → Python scalars
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # Complex numbers
        if isinstance(obj, (np.complexfloating, complex)):
            return {"__type__": "complex", "real": float(obj.real), "imag": float(obj.imag)}

        # SciPy CSR sparse matrix
        if sparse.isspmatrix_csr(obj):
            return {
                "__type__": "csr_matrix",
                "class": "scipy.sparse.csr_matrix",
                "data": obj.data.tolist(),
                "indices": obj.indices.tolist(),
                "indptr": obj.indptr.tolist(),
                "shape": obj.shape,
            }

        # Custom objects with to_dict
        if hasattr(obj, "to_dict"):
            return {
                "__type__": "custom",
                "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "data": obj.to_dict(),
            }

        # Generic objects (fallback)
        if hasattr(obj, "__dict__"):
            return {
                "__type__": "object",
                "class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "data": obj.__dict__,
            }

        # Let the base class handle anything else
        return super().default(obj)

    # Tuple tagging (round‑trip safety)
    def iterencode(self, obj, _one_shot=False):
        # Apply recursive key conversion and tuple encoding before standard encoding
        processed_obj = self._preprocess_for_json(obj)
        return super().iterencode(processed_obj, _one_shot)

    def _preprocess_for_json(self, obj):
        """Recursively converts tuples and ensures dict keys are strings."""
        if isinstance(obj, tuple):
            # Encode tuple with type info
            return {"__type__": "tuple", "data": [self._preprocess_for_json(el) for el in obj]}
        if isinstance(obj, list):
            # Process list items recursively
            return [self._preprocess_for_json(el) for el in obj]
        if isinstance(obj, dict):
            # Process dictionary keys and values recursively
            new_dict = {}
            for k, v in obj.items():
                # Ensure key is string
                new_key = str(k) if not isinstance(k, str) else k
                # Recursively process value
                new_dict[new_key] = self._preprocess_for_json(v)
            return new_dict
        # Return other types unchanged for the default() method to handle
        return obj


# Helper for externalizing large arrays

def externalize_large_arrays(obj: Any, base_dir: Path, npy_subdir_name: str = "_npy_data") -> Any:
    """
    Recursively traverse obj, saving large ndarrays externally and replacing them with references.

    Args:
        obj: The object (dict, list, etc.) to process.
        base_dir: The base directory where the main JSONL file is being saved.
                  Used to create the npy subdirectory and relative paths.
        npy_subdir_name: The name of the subdirectory to store .npy files.

    Returns:
        The modified object with large arrays replaced by references.
    """
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            new_dict[k] = externalize_large_arrays(v, base_dir, npy_subdir_name)
        return new_dict
    elif isinstance(obj, list):
        # Heuristic: convert large numerical (nested) lists back to ndarray so
        # they can benefit from the same externalisation / mmap path as native
        # NumPy arrays.  This guards against tasks that mistakenly call
        # `.tolist()` on large arrays before returning the problem dict.
        try:
            # Fast check: if first element is list/int/float assume numeric list
            if obj and isinstance(obj[0], (list, int, float, np.number)):
                arr_candidate = np.asarray(obj, dtype=float)
                if arr_candidate.ndim >= 1 and arr_candidate.nbytes > NDARRAY_INLINE_THRESHOLD_BYTES:
                    # Recursively handle the ndarray path – this will externalise
                    # it to .npy and return a reference dict.
                    return externalize_large_arrays(arr_candidate, base_dir, npy_subdir_name)
        except Exception:
            # If conversion fails just fall through to the standard list handling.
            pass

        # Default behaviour: recurse into elements
        new_list = [externalize_large_arrays(item, base_dir, npy_subdir_name) for item in obj]
        return new_list
    elif isinstance(obj, tuple):
        # Handle tuples similarly to lists for externalization
        return tuple(externalize_large_arrays(item, base_dir, npy_subdir_name) for item in obj)
    elif sparse.isspmatrix(obj): # Check for any SciPy sparse matrix
        logging.debug(f"Externalizing SciPy sparse matrix of type {type(obj)}, shape {obj.shape}")
        # Determine the type of sparse matrix to reconstruct it later
        if sparse.isspmatrix_csr(obj):
            matrix_type_str = "scipy_csr_matrix_ref"
        elif sparse.isspmatrix_csc(obj):
            matrix_type_str = "scipy_csc_matrix_ref"
        elif sparse.isspmatrix_coo(obj):
            matrix_type_str = "scipy_coo_matrix_ref"
        # Add other sparse types if needed (bsr, dia, lil)
        else:
            logging.warning(f"Unsupported SciPy sparse matrix type {type(obj)} for externalization. Returning as is.")
            return obj # Or handle as an error

        # Externalize components
        ext_data = externalize_large_arrays(obj.data, base_dir, npy_subdir_name)
        ext_indices = externalize_large_arrays(obj.indices, base_dir, npy_subdir_name)
        ext_indptr = None
        if hasattr(obj, 'indptr'): # COO matrices do not have indptr
            ext_indptr = externalize_large_arrays(obj.indptr, base_dir, npy_subdir_name)

        ref_dict = {
            "__type__": matrix_type_str,
            "shape": obj.shape,
            "data": ext_data,
            "indices": ext_indices,
        }
        if ext_indptr is not None:
            ref_dict["indptr"] = ext_indptr

        # For COO, it has row and col attributes instead of indices/indptr for data construction
        if sparse.isspmatrix_coo(obj):
            # In COO, 'indices' usually refers to stacked row/col, but CSR/CSC use 'indices' for column/row indices
            # So, for COO, we store 'row' and 'col' explicitly if they are primary attributes
            # However, obj.indices for COO might not be what we want if it's not explicitly handled as row/col during creation
            # Safest to store what's needed for reconstruction.
            # scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
            # obj.row and obj.col are the attributes
            ref_dict["row"] = externalize_large_arrays(obj.row, base_dir, npy_subdir_name)
            ref_dict["col"] = externalize_large_arrays(obj.col, base_dir, npy_subdir_name)
            # We've already stored obj.data as "data"
            # Remove "indices" if it was from a generic sparse.isspmatrix path and not specific to COO needs
            if "indices" in ref_dict and matrix_type_str == "scipy_coo_matrix_ref":
                 # For COO, data is typically reconstructed with (data, (row, col))
                 # obj.indices is not a standard construction parameter for coo_matrix directly with data, row, col.
                 # Let's ensure we are not confusing. scipy.sparse.coo_matrix constructor takes (data, (row, col))
                 # The attributes are .data, .row, .col
                 # So, we remove the generic .indices which might have been picked up.
                 del ref_dict["indices"] # We will use row and col for COO

        return ref_dict
    elif _is_large_ndarray(obj):
        # Ensure the subdirectory exists
        npy_dir = base_dir / npy_subdir_name
        npy_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filename and save the array
        filename = f"{uuid.uuid4()}.npy"
        npy_file_path = npy_dir / filename
        try:
            np.save(npy_file_path, obj, allow_pickle=False) # Save without pickle for security/portability
            logging.debug(f"Externalized large array ({obj.shape}, {obj.dtype}, {obj.nbytes} bytes) to {npy_file_path}")
        except Exception as e:
            logging.error(f"Failed to save large ndarray to {npy_file_path}: {e}", exc_info=True)
            # Decide on fallback: re-raise, return original obj, or return error marker?
            # For now, let's return the original object to avoid breaking serialization completely,
            # which will likely trigger the existing fallback in DatasetEncoder.default
            return obj

        # Return the reference dictionary
        return {
            "__type__": "ndarray_ref",
            "npy_path": f"{npy_subdir_name}/{filename}" # Relative path
        }
    elif _is_large_bytes(obj):
        # Externalize large bytes objects similar to arrays
        npy_dir = base_dir / npy_subdir_name
        npy_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename and save the bytes
        filename = f"{uuid.uuid4()}.bin"
        bin_file_path = npy_dir / filename
        try:
            with open(bin_file_path, 'wb') as f:
                f.write(obj)
            logging.debug(f"Externalized large bytes ({len(obj)} bytes) to {bin_file_path}")
        except Exception as e:
            logging.error(f"Failed to save large bytes to {bin_file_path}: {e}", exc_info=True)
            return obj
        
        # Return the reference dictionary
        return {
            "__type__": "bytes_ref",
            "bin_path": f"{npy_subdir_name}/{filename}",  # Relative path
            "size": len(obj)
        }
    else:
        # Return other types unchanged
        return obj


# Import helper for custom classes

def get_class(class_path: str) -> Optional[Type]:
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except Exception as exc:
        logging.warning("Could not import class %s: %s", class_path, exc)
        return None


# Decoder (stream‑friendly, mmap for big arrays)

def dataset_decoder(dct: Any, base_dir: Optional[str] = None) -> Any:
    """Inverse of *DatasetEncoder* with lazy mmap loading for ndarray refs."""

    # Fast path – untyped structures: recurse into dicts/lists
    if not (isinstance(dct, dict) and "__type__" in dct):
        if isinstance(dct, dict):
            res = {}
            for k, v in dct.items():
                # Auto-convert numeric string keys → int keys
                try:
                    key = int(k) if k.lstrip("-").isdigit() else k
                except Exception:
                    key = k
                res[key] = dataset_decoder(v, base_dir=base_dir)
            return res
        if isinstance(dct, list):
            return [dataset_decoder(item, base_dir=base_dir) for item in dct]
        return dct

    type_name = dct["__type__"]

    # ---------------- tuples
    if type_name == "tuple":
        return tuple(dataset_decoder(el, base_dir=base_dir) for el in dct.get("data", []))

    # ---------------- bytes
    if type_name == "bytes":
        try:
            return base64.b64decode(dct["data_b64"].encode("ascii"))
        except Exception as exc:
            logging.warning("Failed to decode base64 bytes: %s", exc)
            return dct

    # ---------------- external ndarray (LAZY mmap!)
    if type_name == "ndarray_ref":
        if base_dir is None:
            logging.error("dataset_decoder: base_dir not provided; returning reference dict")
            return dct

        npy_path = os.path.join(base_dir, dct["npy_path"])
        if not os.path.exists(npy_path):
            logging.error("dataset_decoder: npy file not found at %s", npy_path)
            return dct

        try:
            # Use mmap_mode='r' to load lazily from disk
            arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)

            # OPTIONAL: transform memmap → plain ndarray view so
            # accidental JSON dumps won't choke, yet no data is copied.

            return arr
        except Exception as exc:
            logging.error("Failed to mmap-load %s: %s", npy_path, exc)
            logging.error(traceback.format_exc())
            return dct

    # ---------------- external bytes (memory-mapped file)
    if type_name == "bytes_ref":
        if base_dir is None:
            logging.error("dataset_decoder: base_dir not provided for bytes_ref; returning reference dict")
            return dct
        
        bin_path = os.path.join(base_dir, dct["bin_path"])
        if not os.path.exists(bin_path):
            logging.error("dataset_decoder: bin file not found at %s", bin_path)
            return dct
        
        try:
            # Heuristic: eagerly load "moderate"-sized blobs because certain
            # third-party libraries (e.g. *cryptography*’s AEAD ciphers) expect a
            # *bytes* instance and can raise ``MemoryError`` when handed an
            # ``mmap.mmap`` object.  The threshold can be overridden via the
            # env-var ``DATASET_EAGER_BYTES_MAX`` (bytes).

            max_eager = int(os.environ.get("DATASET_EAGER_BYTES_MAX", str(16 * 2**20)))  # 16 MB default
            file_size = os.path.getsize(bin_path)

            if file_size <= max_eager:
                with open(bin_path, "rb") as f:
                    return f.read()

            # For very large blobs fall back to zero-copy mmap to avoid huge RAM spikes.
            import mmap

            with open(bin_path, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            return mm  # mmap object (bytes-like & zero-copy)

        except Exception as exc:
            # mmap can fail on some exotic filesystems (e.g. *procfs* or
            # network mounts).  Fall back to an eager read so callers still get
            # something usable, albeit at the cost of extra RAM.
            logging.warning(
                "dataset_decoder: memory-mapping failed for %s (%s) – falling back to eager read", 
                bin_path, exc,
            )
            try:
                with open(bin_path, "rb") as f:
                    return f.read()
            except Exception as exc2:
                logging.error("dataset_decoder: fallback read failed for %s: %s", bin_path, exc2)
                logging.error(traceback.format_exc())
                return dct

    # ---------------- base64 ndarray
    if type_name == "ndarray_b64":
        try:
            raw = base64.b64decode(dct.get("data_b64", ""))
            arr = np.frombuffer(raw, dtype=np.dtype(dct.get("dtype")))
            shape = tuple(dct.get("shape", []))
            if shape:
                try:
                    return arr.reshape(shape)
                except ValueError:
                    logging.warning(f"Could not reshape buffer to {shape}, returning as 1D array.")
                    return arr # Fallback to flat array if reshape fails
            else:
                # Shape was empty '()', indicating a scalar
                if arr.size == 1:
                     logging.warning(f"Decoded ndarray_b64 resulted in a scalar. Wrapping in 1x1 array.")
                     # Ensure correct dtype is preserved when wrapping
                     return np.array([[arr.item()]], dtype=arr.dtype)
                else:
                     logging.warning(f"Decoded ndarray_b64 has empty shape but non-scalar buffer size ({arr.size}). Returning flat array.")
                     return arr # Fallback for unexpected non-scalar buffer with empty shape
        except Exception as exc:
            logging.error("Failed to decode ndarray_b64: %s", exc)
            logging.error(traceback.format_exc())
            return dct

    # ---------------- inline ndarray (list fallback)
    if type_name == "ndarray":
        return np.array(dct["data"], dtype=np.dtype(dct.get("dtype")))

    # ---------------- complex
    if type_name == "complex":
        return complex(dct["real"], dct["imag"])

    # ---------------- SciPy CSR matrix (old direct handler)
    # This will now primarily handle cases where a CSR matrix was small enough
    # not to be externalized by externalize_large_arrays, or if externalization failed
    # and the original DatasetEncoder.default path for csr_matrix was hit.
    if type_name == "csr_matrix":
        try:
            # These components are expected to be lists if coming from the old encoder path
            data = np.array(dataset_decoder(dct["data"], base_dir=base_dir))
            indices = np.array(dataset_decoder(dct["indices"], base_dir=base_dir))
            indptr = np.array(dataset_decoder(dct["indptr"], base_dir=base_dir))
            shape = tuple(dataset_decoder(dct["shape"], base_dir=base_dir))
            return sparse.csr_matrix((data, indices, indptr), shape=shape)
        except Exception as exc:
            logging.error(f"Failed to decode legacy csr_matrix: {exc}", exc_info=True)
            return dct

    # ---------------- SciPy Sparse Matrix References (New)
    if type_name == "scipy_csr_matrix_ref":
        try:
            # Components are expected to be ndarray_ref dicts or actual ndarrays (if small and inlined)
            # The recursive call to dataset_decoder will handle loading them.
            data = dataset_decoder(dct["data"], base_dir=base_dir)
            indices = dataset_decoder(dct["indices"], base_dir=base_dir)
            indptr = dataset_decoder(dct["indptr"], base_dir=base_dir)
            shape = tuple(dataset_decoder(dct["shape"], base_dir=base_dir)) # Shape is usually small list/tuple
            return sparse.csr_matrix((data, indices, indptr), shape=shape)
        except Exception as exc:
            logging.error(f"Failed to decode scipy_csr_matrix_ref: {exc}", exc_info=True)
            return dct

    if type_name == "scipy_csc_matrix_ref":
        try:
            data = dataset_decoder(dct["data"], base_dir=base_dir)
            indices = dataset_decoder(dct["indices"], base_dir=base_dir)
            indptr = dataset_decoder(dct["indptr"], base_dir=base_dir)
            shape = tuple(dataset_decoder(dct["shape"], base_dir=base_dir))
            return sparse.csc_matrix((data, indices, indptr), shape=shape)
        except Exception as exc:
            logging.error(f"Failed to decode scipy_csc_matrix_ref: {exc}", exc_info=True)
            return dct

    if type_name == "scipy_coo_matrix_ref":
        try:
            data = dataset_decoder(dct["data"], base_dir=base_dir)
            row = dataset_decoder(dct["row"], base_dir=base_dir)
            col = dataset_decoder(dct["col"], base_dir=base_dir)
            shape = tuple(dataset_decoder(dct["shape"], base_dir=base_dir))
            return sparse.coo_matrix((data, (row, col)), shape=shape)
        except Exception as exc:
            logging.error(f"Failed to decode scipy_coo_matrix_ref: {exc}", exc_info=True)
            return dct

    # ---------------- custom / generic object
    if type_name in ("custom", "object"):
        cls = get_class(dct["class"])
        data_decoded = {k: dataset_decoder(v, base_dir=base_dir) for k, v in dct.get("data", {}).items()}
        if cls and type_name == "custom" and hasattr(cls, "from_dict"):
            return cls.from_dict(data_decoded)
        if cls:
            try:
                obj = cls.__new__(cls)
                if hasattr(obj, "__dict__"):
                    obj.__dict__.update(data_decoded)
                return obj
            except Exception as exc:
                logging.warning("Failed to reconstruct %s: %s", dct["class"], exc)
        return data_decoded

    # ---------------- fallback recursive decoding
    if isinstance(dct, dict):
        return {k: dataset_decoder(v, base_dir=base_dir) for k, v in dct.items()}
    return dct