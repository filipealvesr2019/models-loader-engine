from pathlib import Path
import struct


data = Path(r'.\models\qwen2.5-coder-1.5b-instruct-q2_k.gguf').read_bytes()
ptr = 24


def read_u32():
    global ptr
    v = struct.unpack_from('<I', data, ptr)[0]
    ptr += 4
    return v


def read_u64():
    global ptr
    v = struct.unpack_from('<Q', data, ptr)[0]
    ptr += 8
    return v


def read_string():
    global ptr
    l = read_u64()
    s = data[ptr:ptr+l].decode('utf-8', 'ignore')
    ptr += l
    return s

for i in range(30):
    start = ptr
    key = read_string()
    t = read_u32()
    print(i, start, key, t)
    if t in (0, 1, 7):
        ptr += 1
    elif t in (2, 3):
        ptr += 2
    elif t in (4, 5, 6):
        ptr += 4
    elif t in (10, 11, 12):
        ptr += 8
    elif t == 8:
        read_string()
    elif t == 9:
        arr_type = read_u32()
        arr_len = read_u64()
        print(' array', arr_type, arr_len)
        for _ in range(arr_len):
            if arr_type in (0, 1, 7):
                ptr += 1
            elif arr_type in (2, 3):
                ptr += 2
            elif arr_type in (4, 5, 6):
                ptr += 4
            elif arr_type in (10, 11, 12):
                ptr += 8
            elif arr_type == 8:
                read_string()
            elif arr_type == 9:
                raise RuntimeError('nested arrays unsupported')
            else:
                raise RuntimeError(f'bad arr type {arr_type}')
    else:
        raise RuntimeError(f'unknown type {t}')
