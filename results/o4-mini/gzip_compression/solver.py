import zlib
import threading
import struct

# Constants for gzip
HEADER = b'\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff'
PACK = struct.Struct('<II').pack
LEVEL = 9
MASK = 0xFFFFFFFF

# zlib functions
_COMPRESS = zlib.compress
_COMPRESS_OBJ = zlib.compressobj
_CRC32 = zlib.crc32

# Threshold for overlapping CRC computation in a thread
_CRC_THREAD_THRESH = 1 << 14  # 16 KiB

class Solver:
    def solve(self, problem, **kwargs):
        data = problem["plaintext"]
        n = len(data)

        # For small data, single-threaded zlib.compress + slice header/trailer
        if n <= _CRC_THREAD_THRESH:
            full = _COMPRESS(data, LEVEL)
            mv = memoryview(full)
            body_mv = mv[2:-4]
            crc_val = _CRC32(data) & MASK
            isize = n & MASK

        else:
            # Compute CRC32 and size in parallel
            res = {}
            def _calc():
                res['crc'] = _CRC32(data) & MASK
                res['isize'] = n & MASK

            t = threading.Thread(target=_calc)
            t.daemon = True
            t.start()

            # Raw DEFLATE (no wrapper) via compressobj
            co = _COMPRESS_OBJ(LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS)
            body = co.compress(data)
            body += co.flush()

            t.join()
            body_mv = memoryview(body)
            crc_val = res['crc']
            isize = res['isize']

        # Assemble gzip member: header + deflate body + footer
        blen = len(body_mv)
        out = bytearray(10 + blen + 8)
        out[0:10] = HEADER
        out[10:10 + blen] = body_mv
        out[-8:] = PACK(crc_val, isize)

        return {"compressed_data": bytes(out)}