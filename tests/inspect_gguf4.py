from pathlib import Path
import struct

path = Path(r'.\models\qwen2.5-coder-1.5b-instruct-q2_k.gguf')
data = path.read_bytes()
for off in [5931202, 5931229, 5931230, 5931250, 5931260]:
    chunk = data[off:off+32]
    print('offset', off)
    print('hex', chunk.hex())
    print('ascii', chunk)
    print()
