from typing import Any
import ctypes
from operator import itemgetter

try:
    import cryptography.hazmat.bindings._rust as rust
    ChaCha20Poly1305 = rust.openssl.aead.ChaCha20Poly1305
except ImportError:
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

get_args = itemgetter("key", "nonce", "plaintext", "associated_data")

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        k, n, p, a = get_args(problem)
        c = ChaCha20Poly1305(k).encrypt(n, p, a)
        tag = c[-16:]
        
        # Truncate the bytes object in-place to avoid copying the ciphertext
        # In CPython, ob_size is at offset 16 for PyVarObject (bytes)
        ctypes.c_ssize_t.from_address(id(c) + 16).value -= 16
        
        return {"ciphertext": c, "tag": tag}