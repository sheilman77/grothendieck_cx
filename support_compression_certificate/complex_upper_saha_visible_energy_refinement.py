#!/usr/bin/env python3
"""Certified visible-energy subtraction for the origin-layer tail.

The earlier +4.57e-6 certificate bounded each radial Sobolev energy by 13
and spent that entire amount on modes n >= 120.  This certificate proves
that more than 11.497 and 11.498 units, respectively, are already present
in the first 120 modes.  Combining these lower bounds with the sharper
total-energy bounds 12.671 and 12.674 leaves tail energies below 1.174 and
1.176.  Endpoint cancellation then gives an l1 tail below 2.615e-4.
"""

from __future__ import annotations

from fractions import Fraction as Q

import numpy as np
from flint import acb, arb, ctx, fmpq

from complex_upper_origin_layer_definitions import A, B, LEFT, RIGHT
from complex_upper_origin_layer_energy_certificate import (
    complex_polynomial,
    energy,
)


ctx.dps = 60
COUNT = 120
TABLE = "complex_upper_origin_layer_coefficients_n120_analytic.npz"

LEFT_TOTAL = Q(12671, 1000)
RIGHT_TOTAL = Q(12674, 1000)
LEFT_VISIBLE = Q(11497, 1000)
RIGHT_VISIBLE = Q(11498, 1000)
LEFT_TAIL_ENERGY = LEFT_TOTAL - LEFT_VISIBLE       # 1.174
RIGHT_TAIL_ENERGY = RIGHT_TOTAL - RIGHT_VISIBLE   # 1.176
ORIGIN_TAIL = Q(523, 2_000_000)                   # 2.615e-4


def aq(value: Q) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value: float) -> arb:
    numerator, denominator = float(value).as_integer_ratio()
    return arb(fmpq(numerator, denominator))


def interval(lower: float, upper: float) -> arb:
    return endpoint(lower).union(endpoint(upper))


def phase_at_zero(coefficients) -> acb:
    real, imag = coefficients[0]
    value = acb(aq(real), aq(imag))
    return value / abs(value)


def visible_energy(side: str, coefficients) -> arb:
    """Lower-bound the true weighted remainder energy below COUNT.

    The table encloses coefficients of the input restricted to 0 <= s <=
    25.  The omitted input has l2 norm at most exp(-25/2).  Multiplication
    by 2n+1 on the first COUNT coordinates has norm at most 2*COUNT-1, so
    this joint error is subtracted once after forming the boxed norm.
    """
    saved = np.load(TABLE)
    assert int(saved["count"]) == COUNT and int(saved["cutoff"]) == 25
    boxes = tuple(
        acb(
            interval(saved[f"{side}_real_lower"][n],
                     saved[f"{side}_real_upper"][n]),
            interval(saved[f"{side}_imag_lower"][n],
                     saved[f"{side}_imag_upper"][n]),
        )
        for n in range(COUNT)
    )

    endpoint_phase = phase_at_zero(coefficients)
    h = arb.pi() / 4
    boxed_energy = arb(0)
    for n, coefficient in enumerate(boxes):
        q = h.sqrt()
        if n & 1:
            q = -q
        remainder = coefficient - endpoint_phase * q
        lower = abs(remainder).lower()
        boxed_energy += (2 * n + 1) ** 2 * lower * lower
        h *= arb((2 * n + 1) ** 2) / (4 * (n + 1) * (n + 2))

    boxed_norm = boxed_energy.sqrt().lower()
    omitted = (-arb(25) / 2).exp().upper()
    true_norm = boxed_norm - (2 * COUNT - 1) * omitted
    assert true_norm > 0
    return true_norm * true_norm


def tail_comparison() -> None:
    """Verify the endpoint-cancelled product-tail bound exactly.

    If r_n and s_n are the two endpoint-subtracted radial sequences, then

      |k_n| <= sqrt(h_n)(|r_n|+|s_n|) + |r_n||s_n|.

    Cauchy--Schwarz, (2n+1)>=241, and
    sum_{n>=120} h_n/(2n+1)^2 <= 1/(48*119^3) give the comparison below.
    Squaring twice avoids introducing any floating-point square roots.
    """
    weighted_h = Q(1, 48 * (COUNT - 1) ** 3)
    product = LEFT_TAIL_ENERGY * RIGHT_TAIL_ENERGY
    product_term_squared = product / (2 * COUNT + 1) ** 4

    # First prove ORIGIN_TAIL - sqrt(product)/(241^2) is positive and
    # exceeds sqrt(E_L W)+sqrt(E_R W).  The auxiliary rational y is a
    # strict upper bound for sqrt(product)/(241^2).
    y = Q(2024, 10**8)
    assert y * y > product_term_squared
    room = ORIGIN_TAIL - y
    assert room > 0

    # For nonnegative a,b, (sqrt(a)+sqrt(b))^2=a+b+2sqrt(ab).
    # Bound the remaining radical with another directed rational ceiling.
    a = LEFT_TAIL_ENERGY * weighted_h
    b = RIGHT_TAIL_ENERGY * weighted_h
    cross = Q(1453, 10**11)
    assert cross * cross > a * b
    assert room * room > a + b + 2 * cross


def main() -> None:
    left_total = energy(complex_polynomial(LEFT), "left", A)
    right_total = energy(complex_polynomial(RIGHT), "right", B)
    assert left_total < aq(LEFT_TOTAL)
    assert right_total < aq(RIGHT_TOTAL)

    left_visible = visible_energy("left", LEFT)
    right_visible = visible_energy("right", RIGHT)
    assert left_visible > aq(LEFT_VISIBLE)
    assert right_visible > aq(RIGHT_VISIBLE)
    tail_comparison()

    print("origin-layer visible-energy refinement: PASS")
    print("left total / visible  <,>", left_total, left_visible)
    print("right total / visible <,>", right_total, right_visible)
    print("left/right tail energy <", float(LEFT_TAIL_ENERGY),
          float(RIGHT_TAIL_ENERGY))
    print("origin-layer l1 tail  <", float(ORIGIN_TAIL))


if __name__ == "__main__":
    main()
