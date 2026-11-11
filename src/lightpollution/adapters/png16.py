"""Lossless RGB16 PNG with explicit sRGB tagging, no early 8-bit conversion."""

import struct
import zlib

import numpy as np


def write_rgb16(path,values):
    values=np.asarray(values)
    if values.dtype!=np.uint16 or values.ndim!=3 or values.shape[-1]!=3:
        raise ValueError('RGB uint16 array required.')
    height,width=values.shape[:2]
    def chunk(kind,payload):
        return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
    rows=values.astype('>u2').reshape(height,-1)
    raw=b''.join(b'\0'+row.tobytes() for row in rows)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,16,2,0,0,0))+
        chunk(b'sRGB',b'\0')+chunk(b'IDAT',zlib.compress(raw,6))+chunk(b'IEND',b''))
