from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from functools import lru_cache
from typing import Any, Dict, Tuple

# Precomputed backend to avoid repeated calls
BACKEND = default_backend()

@lru_cache(maxsize=2048)
def get_aes_cipher(key_tuple: Tuple[int, ...]) -> Any:
    # Convert tuple back to bytes
    key = bytes(key_tuple)
    # Create a new AES algorithm object for each key
    algorithm = algorithms.AES(key)
    # Return a function that creates a cipher with the given nonce
    return lambda nonce: Cipher(algorithm, modes.GCM(nonce), backend=BACKEND)

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        key = problem["key"]
        nonce = problem["nonce"]
        plaintext = problem["plaintext"]
        aad = problem.get("associated_data", b'')
        
        # Convert key to immutable tuple for caching
        key_tuple = tuple(key)
        
        # Get cipher creator from cache
        cipher_creator = get_aes_cipher(key_tuple)
        
        # Create cipher with GCM mode
        cipher = cipher_creator(nonce)
        encryptor = cipher.encryptor()
        
        # Authenticate additional data if provided
        if aad:
            # Use zero-copy method for AAD
            encryptor.authenticate_additional_data(memoryview(aad))
        
        # Encrypt the plaintext in a single operation
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Optimize return by using tuple instead of dict
        return {
            "ciphertext": ciphertext,
            "tag": encryptor.tag
        }
        # Initialize with key and IV
        lib.EVP_EncryptInit_ex(ctx, ffi.NULL, ffi.NULL, key_bytes, nonce)
        
        # Process AAD if provided
        if aad:
            outlen = ffi.new("int *")
            lib.EVP_EncryptUpdate(ctx, ffi.NULL, outlen, aad, len(aad))
        
        # Encrypt plaintext
        ciphertext_buf = ffi.new("unsigned char[]", len(plaintext))
        outlen = ffi.new("int *")
        lib.EVP_EncryptUpdate(ctx, ciphertext_buf, outlen, plaintext, len(plaintext))
        
        # Finalize encryption
        final_buf = ffi.new("unsigned char[]", 16)
        final_len = ffi.new("int *")
        lib.EVP_EncryptFinal_ex(ctx, final_buf, final_len)
        
        # Get authentication tag
        tag_buf = ffi.new("unsigned char[]", GCM_TAG_SIZE)
        lib.EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, GCM_TAG_SIZE, tag_buf)
        
        # Clean up
        lib.EVP_CIPHER_CTX_free(ctx)
        
        return {
            "ciphertext": bytes(ffi.buffer(ciphertext_buf, len(plaintext))),
            "tag": bytes(ffi.buffer(tag_buf, GCM_TAG_SIZE))
        }