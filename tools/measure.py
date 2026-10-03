"""Measure between two points on a facade ortho using its sidecar.

    python3 tools/measure.py facade.json  412 1880  1038 1880
    -> 3.130 m  (10' 3-1/4")      # on a 5 mm/px ortho

Pixel positions are as most image viewers report them: x to the right, y
down, measured from the top-left corner of the image. Use this against tape
measurements (door widths, window heights) to validate a capture.
"""
import json
import math
import sys


def feet_inches(m: float) -> str:
    total_in = m / 0.0254
    ft = int(total_in // 12)
    inch = total_in - 12 * ft
    whole = int(inch)
    sixteenths = round((inch - whole) * 16)
    if sixteenths == 16:
        whole, sixteenths = whole + 1, 0
    if whole == 12:
        ft, whole = ft + 1, 0
    frac = ""
    if sixteenths:
        g = math.gcd(sixteenths, 16)
        frac = f"-{sixteenths // g}/{16 // g}"
    return f"{ft}' {whole}{frac}\""


def measure(sidecar: dict, x1, y1, x2, y2) -> float:
    gsd = sidecar["image"]["gsd_m"]
    return math.hypot((x2 - x1) * gsd, (y2 - y1) * gsd)


def main(argv):
    if len(argv) != 5:
        print(__doc__)
        return 2
    sidecar = json.loads(open(argv[0]).read())
    x1, y1, x2, y2 = (float(v) for v in argv[1:])
    d = measure(sidecar, x1, y1, x2, y2)
    print(f"{d:.3f} m  ({feet_inches(d)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
