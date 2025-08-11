from typing import Any, Dict
import ast as _ast
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305 as _ChaCha20Poly1305

# Cache ChaCha20Poly1305 instances per key to avoid re‑creating objects
_CHACHA_CACHE: dict[bytes, _ChaCha20Poly1305] = {}

CHACHA_KEY_SIZE = 32  # 256-bit key
POLY1305_TAG_SIZE = 16  # 16-byte tag

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Encrypt the plaintext using ChaCha20-Poly1305.

        Parameters
        ----------
        problem : dict or str
            Dictionary (or its string representation) with keys:
                - "key": bytes (32 bytes)
                - "nonce": bytes (12 bytes)
                - "plaintext": bytes
                - "associated_data": bytes or None (optional)

        Returns
        -------
        dict
            {"ciphertext": bytes, "tag": bytes}
        """
        # Allow string input (as used by eval_input)
        if isinstance(problem, str):
            problem = _ast.literal_eval(problem)
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem.get("associated_data") or b""
        # Cache ChaCha20Poly1305 instance per key
        chacha = _CHACHA_CACHE.get(key)
        if chacha is None:
            chacha = _ChaCha20Poly1305(key); _CHACHA_CACHE[key] = chacha
        full_ct = chacha.encrypt(nonce, plaintext, associated_data)
        return {"ciphertext": full_ct[:-16], "tag": full_ct[-16:]}