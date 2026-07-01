import struct
from pathlib import Path

path = Path(r'.\models\qwen2.5-coder-1.5b-instruct-q2_k.gguf')
data = path.read_bytes()
ptr = 0
magic = data[ptr:ptr+4]
ptr += 4
version = struct.unpack_from('<I', data, ptr)[0]
ptr += 4
n_kv = struct.unpack_from('<Q', data, ptr)[0]
ptr += 8
n_tensors = struct.unpack_from('<Q', data, ptr)[0]
ptr += 8
print('magic', magic, 'version', version, 'n_kv', n_kv, 'n_tensors', n_tensors)


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


def skip_value(t):
    global ptr
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
        for _ in range(arr_len):
            skip_value(arr_type)
    else:
        raise RuntimeError(f'unknown type {t}')

for i in range(int(n_kv)):
    key = read_string()
    t = read_u32()
    if i < 10:
        print(i, key, t)
    skip_value(t)
    if ptr > len(data):
        break
print('end ptr', ptr)
