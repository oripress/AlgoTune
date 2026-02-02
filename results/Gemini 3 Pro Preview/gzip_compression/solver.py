import gzip

class Solver:
    def solve(self, problem, **kwargs):
        # The reference implementation uses gzip.compress(plaintext, compresslevel=9, mtime=0)
        # We must match this size exactly or be very close.
        # Previous attempts with zlib directly failed on some inputs, likely due to header differences or specific gzip behavior.
        # To be safe and valid, we stick to gzip but try to be efficient.
        # Since we can't change the algorithm parameters much without violating size constraints,
        # we focus on minimal overhead.
        
        return {"compressed_data": gzip.compress(problem["plaintext"], compresslevel=9, mtime=0)}