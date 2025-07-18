import gzip

class Solver:
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        # Use compression level 9 to match reference size constraints
        compressed_data = gzip.compress(plaintext, compresslevel=9, mtime=0)
        return {"compressed_data": compressed_data}