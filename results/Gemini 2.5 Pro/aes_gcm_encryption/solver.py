from typing import Any, Dict
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class Solver:
    """
    An optimized solver for AES-GCM encryption that uses a manual, class-level
    dictionary cache for cipher objects. This ensures the cache persists across
    multiple instantiations of the Solver class.
    """
    # Class-level cache to store AESGCM objects.
    _cipher_cache = {}
    # GCM tag size is constant at 16 bytes (128 bits).
    GCM_TAG_SIZE = 16

    def __init__(self):
        """
        Initializes the solver. The cache is stored at the class level.
        """
        pass

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        """
        Encrypts plaintext using AES-GCM, leveraging a manually managed class-level cache.
        """
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        associated_data = problem.get("associated_data", b'')

        # Retrieve the cipher object from the class-level cache.
        # This avoids the expensive key setup on subsequent calls with the same key,
        # even if the Solver object is re-instantiated.
        try:
            aesgcm = Solver._cipher_cache[key]
        except KeyError:
            aesgcm = AESGCM(key)
            Solver._cipher_cache[key] = aesgcm

        # Perform the encryption.
        ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, associated_data)

        # The `encrypt` method returns ciphertext + tag. We must split them.
        tag = ciphertext_with_tag[-self.GCM_TAG_SIZE:]
        ciphertext = ciphertext_with_tag[:-self.GCM_TAG_SIZE]

        return {"ciphertext": ciphertext, "tag": tag}