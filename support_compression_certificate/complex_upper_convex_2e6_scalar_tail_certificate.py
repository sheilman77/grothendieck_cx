#!/usr/bin/env python3
"""Arb/Cauchy tail bounds for the eight nonmonomial scalar branches."""

from fractions import Fraction as Q
import math

from flint import acb, arb, ctx, fmpq, fmpq_poly

from complex_upper_convex_2e6_low_certificate import (
    EXPECTED_IDS,
    exact_arb,
    fq,
    prepared_candidate,
)


ctx.dps = 60
GRID = 4096


def polynomial_value(coefficients, z):
    z2 = z * z
    value = acb(0)
    for coefficient in reversed(coefficients):
        value = value * z2 + exact_arb(coefficient)
    return z * value


def segment_distance(v):
    x, y = v.real, v.imag
    modulus_squared = x * x + y * y
    if x < 0:
        return arb(1)
    if x < modulus_squared:
        return abs(y) / modulus_squared.sqrt()
    return ((1 - x) ** 2 + y**2).sqrt()


def root_check(coefficients, radius):
    ordinary = [fmpq(0)] * (2 * len(coefficients))
    for n, coefficient in enumerate(coefficients):
        ordinary[2 * n + 1] = fq(coefficient)
    for target in (-1, 1):
        shifted = ordinary[:]
        shifted[0] -= target
        polynomial = fmpq_poly(shifted)
        roots = polynomial.complex_roots()
        assert sum(multiplicity for _root, multiplicity in roots) == polynomial.degree()
        for root, _multiplicity in roots:
            assert abs(root).lower() > exact_arb(radius)


def boundary_check(coefficients, radius):
    weighted_norm = sum(
        abs(coefficient) * radius ** (2 * n + 1)
        for n, coefficient in enumerate(coefficients)
    )
    derivative_norm = sum(
        (2 * n + 1) * abs(coefficient) * radius ** (2 * n + 1)
        for n, coefficient in enumerate(coefficients)
    )
    assert weighted_norm < Q(5, 4)
    lipschitz = 2 * weighted_norm * derivative_norm
    angular_error = exact_arb(lipschitz) * arb.pi().upper() / GRID
    radius_arb = exact_arb(radius)
    minimum = arb(1)
    for j in range(GRID):
        theta = 2 * arb.pi() * j / GRID
        sine, cosine = theta.sin_cos()
        z = acb(radius_arb * cosine, radius_arb * sine)
        value = polynomial_value(coefficients, z) ** 2
        midpoint = acb(value.real.mid(), value.imag.mid())
        lower = (
            segment_distance(midpoint).lower()
            - abs(value - midpoint).upper() - angular_error
        )
        if lower < minimum:
            minimum = lower
    assert minimum > arb("0.01")
    return weighted_norm, minimum


def main():
    correlations, weights = prepared_candidate()
    total_tail = Q(0)
    print("second candidate scalar Cauchy checks:")
    for branch, identifier in enumerate(EXPECTED_IDS):
        coefficients = correlations[branch]
        if identifier == 88:
            # This special correlation is a*z-b*z^15.  Its nearest complex
            # singularity makes a Cauchy circle wasteful, but its absolute
            # coefficient tail is an exact binomial sum.  A monomial in
            # (a*z+b*z^15)^p has degree p+14*k.
            assert all(
                value == 0 for n, value in enumerate(coefficients)
                if n not in (0, 7)
            )
            a, b = abs(coefficients[0]), abs(coefficients[7])
            assert a + b <= 1
            finite = Q(0)
            haagerup_over_pi = Q(1, 4)
            for j in range(120):
                p = 2 * j + 1
                k0 = max(0, (241 - p + 13) // 14)
                binomial_tail = sum(
                    Q(math.comb(p, k)) * a ** (p - k) * b**k
                    for k in range(k0, p + 1)
                )
                finite += haagerup_over_pi * binomial_tail
                haagerup_over_pi *= Q(
                    (2 * j + 1) ** 2, 4 * (j + 1) * (j + 2)
                )
            # pi*A_j is h_j; Wallis gives sum_{j>=120} h_j<=1/(4*120).
            tail = Q(104348, 33215) * finite + Q(1, 4 * 120)
            contribution = abs(weights[branch]) * tail
            assert contribution < Q(11, 10**9)
            total_tail += contribution
            print(identifier, "exact binomial weighted tail<", float(contribution))
            continue

        radius = Q(109, 100) if identifier == 123 else Q(111, 100)
        root_check(coefficients, radius)
        weighted_norm, minimum = boundary_check(coefficients, radius)
        # Euler's integral gives boundary magnitude <10.
        tail = Q(10) * radius ** (-241) / (1 - radius ** (-2))
        contribution = abs(weights[branch]) * tail
        total_tail += contribution
        print(
            identifier,
            "R=", float(radius),
            "W=", float(weighted_norm),
            "delta>", float(minimum),
            "weighted tail<", float(contribution),
        )
    assert total_tail < Q(12, 10**9)
    print("scalar Cauchy-tail certificate: PASS")
    print("weighted scalar tail <", float(total_tail), "< 12e-9")


if __name__ == "__main__":
    main()
