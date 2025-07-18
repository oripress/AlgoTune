from typing import Any, Callable
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

class Solver:
    """
    A solver for the ChaChaEncryption task, highly optimized for performance.

    This implementation is based on the key insight that the primary bottleneck
    is the repeated initialization of the cipher object for the same key. The
    optimization strategy is to cache the result of this expensive operation.

    Further analysis revealed that minimizing Python-level overhead on the hot
    path (a cache hit) is crucial. This version refines the caching by storing
    the bound `encrypt` method directly. This eliminates an attribute lookup
    (`chacha.encrypt`) on every call for a cached key, creating the most direct
    path from a key to the underlying C encryption function.

    Optimizations Summary:
    1. Bound Method Caching: The `encrypt` method is cached, not the object.
       This is theoretically the fastest approach as it avoids an extra
       attribute lookup on the hot path.
    2. Fast Cache Implementation: A simple instance dictionary with a
       `try/except KeyError` block is used, which is the fastest cache lookup
       mechanism in Python for high-hit-rate scenarios.
    3. Local Variable Binding: The cache dictionary is bound to a local
       variable to accelerate access within the `solve` method.
    """
    TAG_SIZE = 16

    def __init__(self):
        """
        Initializes the solver. The cache stores bound `encrypt` methods,
        keyed by the encryption key.
        """
        # The type hint clarifies that we are storing the callable method.
        self.encrypt_cache: dict[bytes, Callable[..., bytes]] = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        """
        Encrypts plaintext using ChaCha20-Poly1305 with aggressive optimizations.
        """
        cache = self.encrypt_cache
        key = problem["key"]

        # Directly look up the bound `encrypt` method in the cache.
        # This is the most direct path for a cache hit.
        try:
            encrypt = cache[key]
        except KeyError:
            # On a cache miss, perform the one-time setup:
            # 1. Create the cipher object.
            # 2. Get its bound `encrypt` method.
            # 3. Cache the method itself for all future calls with this key.
            chacha = ChaCha20Poly1305(key)
            encrypt = chacha.encrypt
            cache[key] = encrypt

        # Perform the encryption using the retrieved method.
        # The dictionary lookups for the arguments are unavoidable.
        ciphertext_with_tag = encrypt(
            problem["nonce"],
            problem["plaintext"],
            problem["associated_data"]
        )

        # Split the tag from the ciphertext using efficient slicing.
        tag = ciphertext_with_tag[-self.TAG_SIZE:]
        ciphertext = ciphertext_with_tag[:-self.TAG_SIZE]

        return {"ciphertext": ciphertext, "tag": tag}