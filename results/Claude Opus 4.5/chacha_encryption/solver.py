from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Pre-warm during module load
_dummy_key = b'\x00' * 32
_dummy_nonce = b'\x00' * 12
_warmup = ChaCha20Poly1305(_dummy_key)
_warmup.encrypt(_dummy_nonce, b'x', None)
del _warmup, _dummy_key, _dummy_nonce

class Solver:
    __slots__ = ('_cache_key', '_cipher')
    
    def __init__(self):
        self._cache_key = None
        self._cipher = None
    
    def solve(self, problem, **kwargs):
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem["associated_data"]
        
        # Cache cipher if same key
        if key != self._cache_key:
            self._cache_key = key
            self._cipher = ChaCha20Poly1305(key)
        
        ct = self._cipher.encrypt(nonce, plaintext, aad)
        return {"ciphertext": ct[:-16], "tag": ct[-16:]}