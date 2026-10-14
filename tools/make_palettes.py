#!/usr/bin/env python3
"""Build data/textures/retro/palettes.dds, the colour-map LUT stack used by retro.omwfx.

Each palette is a 64x64x64 lookup cube (the same resolution as Retro Skyrim's
lut.png). The cubes are stacked along the depth axis of a single 3D texture,
64 x 64 x (64 * N), in the order of PALETTES below. Slot 1 in the shader is the
first entry here, slot 2 the second, and so on (slot 0 is plain posterize).

Cube layout: texel (x, y, z) = (red index, green index, blue index + 64 * slot).

A 3D DDS is used because OpenMW never flips 3D textures on load, so the
orientation is identical in every OpenMW version (2D PNGs are not).

Retro Skyrim's own palettes (PAL, Deluxe Paint) are not regenerated: they are
copied from the existing palettes.dds, so they must stay the first entries.

Requires numpy:  python3 tools/make_palettes.py
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "textures" / "retro" / "palettes.dds"

SIZE = 64


def hexes(*values):
    return [tuple(int(v[i : i + 2], 16) for i in (0, 2, 4)) for v in values]


def levels(r, g, b):
    """Every combination of the given per-channel levels (0-255)."""
    return [(x, y, z) for x in r for y in g for z in b]


# Order matters: it must match the uPalette choice list in data/shaders/retro.omwfx.
# ORIGINAL entries are Retro Skyrim's own LUTs, kept verbatim from the existing palettes.dds.
ORIGINAL = None
PALETTES = [
    ("PAL (Retro Skyrim default)", ORIGINAL),
    ("Deluxe Paint", ORIGINAL),
    ("Game Boy", hexes("0f380f", "306230", "8bac0f", "9bbc0f")),
    ("Virtual Boy", hexes("000000", "550000", "aa0000", "ff0000")),
    ("Monochrome (4 greys)", hexes("000000", "555555", "aaaaaa", "ffffff")),
    ("CGA (cyan / magenta)", hexes("000000", "55ffff", "ff55ff", "ffffff")),
    ("CGA / EGA (16 colours)", hexes(
        "000000", "0000aa", "00aa00", "00aaaa", "aa0000", "aa00aa", "aa5500", "aaaaaa",
        "555555", "5555ff", "55ff55", "55ffff", "ff5555", "ff55ff", "ffff55", "ffffff")),
    ("EGA / Master System (64)", levels(*[(0x00, 0x55, 0xAA, 0xFF)] * 3)),
    ("Teletext / BBC Micro (8)", levels(*[(0x00, 0xFF)] * 3)),
    ("ZX Spectrum", hexes(
        "000000", "0000d7", "d70000", "d700d7", "00d700", "00d7d7", "d7d700", "d7d7d7",
        "0000ff", "ff0000", "ff00ff", "00ff00", "00ffff", "ffff00", "ffffff")),
    ("Commodore 64", hexes(
        "000000", "ffffff", "68372b", "70a4b2", "6f3d86", "588d43", "352879", "b8c76f",
        "6f4f25", "433900", "9a6759", "444444", "6c6c6c", "9ad284", "6c5eb5", "959595")),
    ("MSX", hexes(
        "000000", "3eb849", "74d07d", "5955e0", "8076f1", "b95e51", "65dbef", "db6559",
        "ff897d", "ccc35e", "ded087", "3aa241", "b766b5", "cccccc", "ffffff")),
    ("Apple II", hexes(
        "000000", "6c2940", "403578", "d93cf0", "135740", "808080", "2697f0", "bfb4f8",
        "404b07", "d9680f", "eca8bf", "26c30f", "bfca87", "93d6bf", "ffffff")),
    ("Macintosh II", hexes(
        "ffffff", "ffff00", "ff6500", "dc0000", "ff0097", "360097", "0000ca", "0097ff",
        "00a800", "006500", "653600", "976536", "b9b9b9", "868686", "454545", "000000")),
    ("PICO-8", hexes(
        "000000", "1d2b53", "7e2553", "008751", "ab5236", "5f574f", "c2c3c7", "fff1e8",
        "ff004d", "ffa300", "ffec27", "00e436", "29adff", "83769c", "ff77a8", "ffccaa")),
    ("Amstrad CPC (27)", levels(*[(0x00, 0x80, 0xFF)] * 3)),
    ("NES", hexes(
        "7c7c7c", "0000fc", "0000bc", "4428bc", "940084", "a80020", "a81000", "881400",
        "503000", "007800", "006800", "005800", "004058", "000000",
        "bcbcbc", "0078f8", "0058f8", "6844fc", "d800cc", "e40058", "f83800", "e45c10",
        "ac7c00", "00b800", "00a800", "00a844", "008888",
        "f8f8f8", "3cbcfc", "6888fc", "9878f8", "f878f8", "f85898", "f87858", "fca044",
        "f8b800", "b8f818", "58d854", "58f898", "00e8d8", "787878",
        "fcfcfc", "a4e4fc", "b8b8f8", "d8b8f8", "f8b8f8", "f8a4c0", "f0d0b0", "fce0a8",
        "f8d878", "d8f878", "b8f8b8", "b8f8d8", "00fcfc", "f8d8f8")),
    ("8-bit RGB (3-3-2)", levels(
        [round(i * 255 / 7) for i in range(8)],
        [round(i * 255 / 7) for i in range(8)],
        [round(i * 255 / 3) for i in range(4)])),
]


def original_cubes():
    """Retro Skyrim's LUTs as [b][g][r] cubes, read from the current palettes.dds.

    They were converted from Retro Skyrim's lut.png files (4096x64, column = r + 64 * b,
    row = g, as in lutSample() in the ENB shader).
    """
    data = OUTPUT.read_bytes()
    depth = struct.unpack_from("<I", data, 24)[0]  # dwDepth
    volume = np.frombuffer(data, np.uint8, offset=128).reshape(depth, SIZE, SIZE, 3)
    count = sum(1 for _, colours in PALETTES if colours is ORIGINAL)
    return [volume[i * SIZE : (i + 1) * SIZE] for i in range(count)]


def palette_cube(colours):
    """Map every cube cell to its nearest palette colour.

    Retro Skyrim's own LUT treats the palette as linear and compares it after
    gamma-encoding (pal ^ 1/2.2) against the cell's colour. The shader brightens
    the scene with pow(0.45) before the lookup, so the LUT has to darken it back.
    This reproduces the shipped LUT except for near-ties, so new palettes keep
    the same overall brightness as the original.
    """
    pal = np.asarray(sorted(set(colours)), dtype=np.float64)
    encoded = (pal / 255.0) ** (1.0 / 2.2)
    i = (np.arange(SIZE) + 0.5) / SIZE
    b, g, r = np.meshgrid(i, i, i, indexing="ij")
    cells = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    nearest = np.empty(len(cells), dtype=np.int64)
    for start in range(0, len(cells), 1 << 15):
        chunk = cells[start : start + (1 << 15)]
        dist = ((chunk[:, None, :] - encoded[None, :, :]) ** 2).sum(-1)
        nearest[start : start + len(chunk)] = dist.argmin(1)
    return pal[nearest].astype(np.uint8).reshape(SIZE, SIZE, SIZE, 3)


def dds_volume_header(width, height, depth):
    DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PITCH = 0x1, 0x2, 0x4, 0x8
    DDSD_PIXELFORMAT, DDSD_DEPTH = 0x1000, 0x800000
    DDPF_RGB = 0x40
    DDSCAPS_COMPLEX, DDSCAPS_TEXTURE = 0x8, 0x1000
    DDSCAPS2_VOLUME = 0x200000

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PITCH | DDSD_PIXELFORMAT | DDSD_DEPTH
    # 24-bit, red in the lowest byte: bytes are stored R, G, B (OSG's "B8G8R8" -> GL_RGB).
    pixel_format = struct.pack("<8I", 32, DDPF_RGB, 0, 24, 0x0000FF, 0x00FF00, 0xFF0000, 0)
    header = struct.pack("<7I", 124, flags, height, width, width * 3, depth, 0)
    header += b"\0" * 44 + pixel_format
    header += struct.pack("<5I", DDSCAPS_COMPLEX | DDSCAPS_TEXTURE, DDSCAPS2_VOLUME, 0, 0, 0)
    assert len(header) == 124
    return b"DDS " + header


def main():
    originals = original_cubes()
    assert all(colours is ORIGINAL for _, colours in PALETTES[: len(originals)]), "originals must come first"
    cubes = []
    for slot, (name, colours) in enumerate(PALETTES, start=1):
        cube = originals[slot - 1] if colours is ORIGINAL else palette_cube(colours)
        count = len(np.unique(cube.reshape(-1, 3), axis=0))
        print(f"{slot:2d}  {name:28s} {count:3d} colours in LUT", file=sys.stderr)
        cubes.append(cube)

    volume = np.concatenate(cubes, axis=0)  # [b + 64 * slot][g][r]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(dds_volume_header(SIZE, SIZE, volume.shape[0]) + volume.tobytes())
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(PALETTES)} palettes)", file=sys.stderr)


if __name__ == "__main__":
    main()
