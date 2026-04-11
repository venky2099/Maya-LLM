"""
sign_paper.py â€” LSB Steganographic Figure Signing
Nexus Learning Labs â€” Maya Research Series (P6+ convention)

Embeds ORCID and canary string into figure PNGs via LSB steganography.
Usage: python sign_paper.py figures/
"""

import os
import sys
import struct
from PIL import Image
from maya_llm.utils.config import CANARY_STRING, ORCID_MAGIC

WATERMARK = f"{CANARY_STRING}|ORCID:0000-0002-3315-7907|MAGIC:{ORCID_MAGIC}"


def _str_to_bits(s: str) -> list:
    bits = []
    for ch in s.encode("utf-8"):
        for i in range(7, -1, -1):
            bits.append((ch >> i) & 1)
    return bits


def sign_image(path: str) -> None:
    img = Image.open(path).convert("RGB")
    pixels = list(img.getdata())
    payload = WATERMARK + "|||END|||"
    bits = _str_to_bits(payload)

    if len(bits) > len(pixels) * 3:
        print(f"[sign_paper] WARNING: {path} too small to embed full watermark.")
        return

    new_pixels = []
    bit_idx = 0
    for r, g, b in pixels:
        if bit_idx < len(bits):
            r = (r & ~1) | bits[bit_idx]; bit_idx += 1
        if bit_idx < len(bits):
            g = (g & ~1) | bits[bit_idx]; bit_idx += 1
        if bit_idx < len(bits):
            b = (b & ~1) | bits[bit_idx]; bit_idx += 1
        new_pixels.append((r, g, b))

    img.putdata(new_pixels)
    img.save(path)
    print(f"[sign_paper] Signed: {path}")


def sign_directory(directory: str) -> None:
    for fname in os.listdir(directory):
        if fname.lower().endswith(".png"):
            sign_image(os.path.join(directory, fname))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "figures"
    if os.path.isdir(target):
        sign_directory(target)
    elif os.path.isfile(target):
        sign_image(target)
    else:
        print(f"[sign_paper] Target not found: {target}")


