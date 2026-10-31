"""Import the output of PBRT's original rgb2spec_opt, without refitting it.

Generate first, using the unchanged upstream program:
    g++ -O3 -std=c++17 -pthread scripts/pbrt_rgb2spec_opt.cpp -o /tmp/rgb2spec_opt
    /tmp/rgb2spec_opt 64 /tmp/rgbspectrum_srgb.cpp sRGB
Then run:
    python scripts/generate_rgb_spectra.py /tmp/rgbspectrum_srgb.cpp
"""

from pathlib import Path
import hashlib
import json
import math
import re
import struct
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    source = Path(sys.argv[1]).read_bytes()
    text = source.decode()
    resolution = int(re.search(r'sRGBToSpectrumTable_Res = (\d+)', text)[1])
    if resolution != 64:
        raise ValueError('PBRT sRGB reconstruction requires the original resolution of 64')
    number = r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'
    scale = re.search(r'sRGBToSpectrumTable_Scale\[64\] = \{(.*?)\};', text, re.S)[1]
    data = re.search(r'sRGBToSpectrumTable_Data[^=]*= \{(.*?)\};', text, re.S)[1]
    nodes = [float(v) for v in re.findall(number, scale)]
    coefficients = [float(v) for v in re.findall(number, data)]
    assert len(nodes) == 64 and nodes[0] == 0 and nodes[-1] == 1
    assert all(a < b for a, b in zip(nodes, nodes[1:]))
    assert len(coefficients) == 3 * 64**3 * 3
    assert all(math.isfinite(v) for v in coefficients)
    values = nodes + coefficients
    payload = struct.pack('<8sII', b'BJRGB001', 64, len(values)) + struct.pack(f'<{len(values)}f', *values)
    output = root / 'bajo/core/data/srgb_spectra.bin'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    generator = root / 'scripts/pbrt_rgb2spec_opt.cpp'
    metadata = {
        'license': 'Apache-2.0; see bajo/parser/pbrt/LICENSE-pbrt.txt',
        'generator_url': 'https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/cmd/rgb2spec_opt.cpp',
        'generator_sha256': hashlib.sha256(generator.read_bytes()).hexdigest(),
        'generator_command': 'rgb2spec_opt 64 rgbspectrum_srgb.cpp sRGB',
        'generated_source_sha256': hashlib.sha256(source).hexdigest(),
        'binary_sha256': hashlib.sha256(payload).hexdigest(),
        'resolution': 64,
        'node_count': 64,
        'coefficient_count': len(coefficients),
        'storage': 'original Float32 bits, little endian; 16-byte header',
    }
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{output}: {len(payload):,} bytes; {len(coefficients):,} original coefficients')


if __name__ == '__main__':
    main()
