"""Small CBOR-safe artifact envelopes for restricted Modal Functions."""
import gzip
import zlib

MAX_INLINE_BYTES=2*1024*1024
MAX_WEIGHTS_BYTES=32*1024*1024


def pack_weights(raw):
    if type(raw) is not bytes or len(raw)>MAX_WEIGHTS_BYTES:raise ValueError('weights byte limit')
    compressed=gzip.compress(raw)
    if len(compressed)>MAX_INLINE_BYTES:raise ValueError('compressed artifact exceeds restricted inline transport')
    return compressed


def unpack_weights(compressed,max_bytes=MAX_WEIGHTS_BYTES):
    if type(compressed) is not bytes or len(compressed)>MAX_INLINE_BYTES:raise ValueError('compressed byte limit')
    decoder=zlib.decompressobj(wbits=31)
    raw=decoder.decompress(compressed,max_bytes+1)
    if len(raw)>max_bytes or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
        raise ValueError('invalid, oversized or concatenated gzip artifact')
    return raw
