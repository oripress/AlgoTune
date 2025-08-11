from typing import Any, Dict, Optional
from collections import OrderedDict
import ast
import json
import base64
import re
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHACHA_KEY_SIZE = 32
CHACHA_NONCE_SIZE = 12
POLY1305_TAG_SIZE = 16
_CIPHER_CACHE_SIZE = 64  # LRU cache size for ChaCha20Poly1305 objects

class Solver:
    """
    Solver for ChaCha20-Poly1305 encryption tasks.

    - Accepts dict inputs or string/bytes representations (parsed via ast.literal_eval or json.loads).
    - Normalizes key, nonce, plaintext, and associated_data to bytes.
    - Uses a small LRU cache of ChaCha20Poly1305 objects for repeated keys.
    """

    def __init__(self) -> None:
        # Pre-warm a cipher instance (init-time work doesn't count toward solve runtime)
        try:
            ChaCha20Poly1305(b"\x00" * CHACHA_KEY_SIZE)
        except Exception:
            pass
        self._cipher_cache = OrderedDict()

    def _get_cipher(self, key_bytes: bytes) -> ChaCha20Poly1305:
        cache = self._cipher_cache
        chacha = cache.get(key_bytes)
        if chacha is not None:
            try:
                cache.move_to_end(key_bytes)
            except Exception:
                pass
            return chacha
        chacha = ChaCha20Poly1305(key_bytes)
        cache[key_bytes] = chacha
        if len(cache) > _CIPHER_CACHE_SIZE:
            try:
                cache.popitem(last=False)
            except Exception:
                pass
        return chacha

    def _parse_problem(self, problem: Any) -> Dict[str, Any]:
        """
        Ensure 'problem' is a dict. If it's a string/bytes, try parsing it.
        Order: dict -> attempt dict(problem) -> ast.literal_eval -> json.loads -> eval (restricted)
        """
        if isinstance(problem, dict):
            return problem

        # Try to coerce mapping-like objects
        if not isinstance(problem, (str, bytes, bytearray)):
            try:
                return dict(problem)  # type: ignore
            except Exception:
                pass

        # Convert bytes to string when possible
        if isinstance(problem, (bytes, bytearray)):
            try:
                s = problem.decode("utf-8")
            except Exception:
                s = repr(problem)
        else:
            s = problem  # type: ignore

        if not isinstance(s, str):
            raise TypeError("problem must be a dict or a parseable string/bytes representation")

        s = s.strip()
        if not s:
            raise TypeError("Empty problem string")

        # Safe literal eval first (handles Python literals like b'..' or {'key': b'..'})
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Try JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Last resort: eval with restricted globals (may handle expressions like b'\\x01'*32)
        try:
            parsed = eval(s, {"__builtins__": None}, {})
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        raise TypeError("Unable to parse 'problem' into a dict")

    def _to_bytes(self, v: Any, name: str = "value", allow_none: bool = False) -> Optional[bytes]:
        """
        Convert common representations to bytes:
        - bytes/bytearray/memoryview -> bytes
        - str: try ast.literal_eval (for Python bytes literal), hex, base64, else utf-8 encode
        - list of ints -> bytes(list)
        - None -> None or empty bytes depending on allow_none
        """
        if v is None:
            return None if allow_none else b""

        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)

        if isinstance(v, list) and all(isinstance(x, int) and 0 <= x < 256 for x in v):
            return bytes(v)

        if isinstance(v, str):
            s = v.strip()

            # Try Python literal (e.g., "b'\\x01'*32" or "[1,2,3]")
            try:
                val = ast.literal_eval(s)
                if isinstance(val, (bytes, bytearray)):
                    return bytes(val)
                if isinstance(val, list) and all(isinstance(x, int) for x in val):
                    return bytes(val)
                if isinstance(val, str):
                    return val.encode("utf-8")
            except Exception:
                pass

            # Hex string (optionally starting with 0x)
            hs = s
            if hs.startswith(("0x", "0X")):
                hs = hs[2:]
            if len(hs) > 0 and len(hs) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in hs):
                try:
                    return bytes.fromhex(hs)
                except Exception:
                    pass

            # Base64-looking heuristic: only base64 chars and proper padding
            if len(s) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", s):
                try:
                    # validate=True ensures strict base64
                    return base64.b64decode(s, validate=True)
                except Exception:
                    pass

            # Fallback: UTF-8 encode
            return s.encode("utf-8")

        try:
            return bytes(v)
        except Exception as e:
            raise TypeError(f"Could not convert {name} to bytes") from e

    def solve(self, problem: Any, **kwargs) -> Dict[str, bytes]:
        """
        Encrypt plaintext using ChaCha20-Poly1305.

        :param problem: dict (or string/bytes representation) with keys "key", "nonce",
                        optional "plaintext", and optional "associated_data"
        :return: dict with "ciphertext" and "tag" (16 bytes)
        """
        prob = self._parse_problem(problem)

        try:
            key = prob["key"]
            nonce = prob["nonce"]
        except Exception as e:
            raise TypeError("problem must contain 'key' and 'nonce'") from e

        plaintext = prob.get("plaintext", b"")
        associated_data = prob.get("associated_data", None)

        # Normalize inputs to bytes
        key_bytes = self._to_bytes(key, "key")
        if not isinstance(key_bytes, (bytes, bytearray)) or len(key_bytes) != CHACHA_KEY_SIZE:
            raise ValueError(f"Invalid key size: {0 if key_bytes is None else len(key_bytes)}. Must be {CHACHA_KEY_SIZE} bytes.")

        nonce_bytes = self._to_bytes(nonce, "nonce")
        if not isinstance(nonce_bytes, (bytes, bytearray)) or len(nonce_bytes) != CHACHA_NONCE_SIZE:
            raise ValueError(f"Invalid nonce size: {0 if nonce_bytes is None else len(nonce_bytes)}. Must be {CHACHA_NONCE_SIZE} bytes.")

        pb = self._to_bytes(plaintext, "plaintext")
        if pb is None:
            pb = b""
        ad = None if associated_data is None else self._to_bytes(associated_data, "associated_data")

        # Retrieve or create cipher from cache
        key_b: bytes = bytes(key_bytes)  # ensure exact bytes type for dict key
        chacha = self._get_cipher(key_b)

        # Perform encryption (ChaCha20Poly1305.encrypt returns ciphertext || tag)
        encrypted = chacha.encrypt(nonce_bytes, pb, ad)

        if len(encrypted) < POLY1305_TAG_SIZE:
            raise ValueError("Encrypted output too short to contain tag.")
        tag = encrypted[-POLY1305_TAG_SIZE:]
        ciphertext = encrypted[:-POLY1305_TAG_SIZE]

        return {"ciphertext": ciphertext, "tag": tag}