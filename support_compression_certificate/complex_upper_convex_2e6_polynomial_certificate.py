#!/usr/bin/env python3
"""Exact nonvanishing and derivative bounds for the second radial phase."""

from fractions import Fraction as F

import complex_radial_phase_polynomial_certificate as base
from complex_upper_convex_2e6_radial_coefficients import PIM_STR, PRE_STR


base.RE = [F(x) for x in PRE_STR]
base.IM = [F(x) for x in PIM_STR]
base.ROOT_CENTERS = [
    ("30.018746546132768", "0.22837785218125736"),
    ("-9.909909936094154", "-0.65487695686102843"),
    ("17.429126057028661", "-0.79421603000494922"),
    ("14.057223751270087", "1.6020991928798534"),
    ("8.8558850817043933", "-3.9541171831703816"),
    ("6.6499363248611392", "4.1572909909173434"),
    ("4.0953759058425714", "-2.3659719313520959"),
    ("1.8304226298723982", "1.7624572418988746"),
    ("0.26887083263079758", "-0.43736425251442085"),
]


def root_disks():
    radius = F(1, 10**6)
    centers = [(F(a), F(b)) for a, b in base.ROOT_CENTERS]
    derivative_pairs = [(base.RE, base.IM)]
    for _ in range(9):
        real, imag = derivative_pairs[-1]
        derivative_pairs.append((base.derivative(real), base.derivative(imag)))

    for center in centers:
        values = [
            base.complex_evaluate(real, imag, center)
            for real, imag in derivative_pairs
        ]
        left = base.lower_norm(values[1]) * radius
        right = base.upper_norm(values[0])
        factorial = 1
        for k in range(2, 10):
            factorial *= k
            right += base.upper_norm(values[k]) * radius**k / factorial
        assert left > right
    for i, left in enumerate(centers):
        for right in centers[i + 1 :]:
            assert max(
                abs(left[0] - right[0]), abs(left[1] - right[1])
            ) > 2 * radius
    assert len(centers) == len(base.RE) - 1

    distances = [abs(center[1]) - radius for center in centers]
    assert all(distance > 0 for distance in distances)
    c1 = sum(1 / distance for distance in distances)
    c2 = sum(1 / distance**2 for distance in distances)
    assert c1 < 12
    assert c2 < 30
    return c1, c2


def main():
    assert base.sturm_nonvanishing() == 0
    c1, c2 = root_disks()
    print("second radial polynomial exact algebra: PASS")
    print("|P(s)| > 3/5 for every s >= 0")
    print("nine disjoint root disks of radius 10^-6 certified")
    print("sup |theta'| <", float(c1), "< 12")
    print("sup |theta''| <", float(c2), "< 30")


if __name__ == "__main__":
    main()
