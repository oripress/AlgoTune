import zlib

class Solver:
    def solve(self, problem, **kwargs):
        plaintext = problem["plaintext"]
        co = zlib.compressobj(9, zlib.DEFLATED, 31, zlib.DEF_MEM_LEVEL, 0)
        result = co.compress(plaintext) + co.flush()
        return {"compressed_data": result}