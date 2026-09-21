#!/usr/bin/env python3
"""Directed Arb check of slope and the first 119 nonlinear coefficients.

The radial pair is defined *exactly* by

  U(z)=phase(z) u(|z|^2),  V(z)=phase(z) C conjugate(u(|z|^2)),

where u=P/|P| and C=(1-t^2+2it)/(1+t^2),
t=356080252073/10^12.  Thus only one radial coefficient table is needed:
after conjugate realification its kernel coefficient is
Re(conjugate(C) a_n^2).
"""

from __future__ import annotations

from fractions import Fraction as Q

import numpy as np
from flint import acb, arb, ctx, fmpq, fmpq_poly


ctx.dps = 60
COUNT = 120
INWARD = Q(10**12 - 1, 10**12)
CANDIDATE = "complex_upper_convex_2e6_candidate.npz"
TABLE = "complex_upper_convex_2e6_radial_coefficients.npz"
EXPECTED_IDS = (39, 86, 88, 94, 121, 123, 191, 195)


def rational(value):
    return Q(repr(float(value)))


def fq(value):
    if not isinstance(value, Q):
        value = Q(value)
    return fmpq(value.numerator, value.denominator)


def exact_arb(value):
    return arb(fq(value))


def endpoint(value):
    numerator, denominator = float(value).as_integer_ratio()
    return arb(fmpq(numerator, denominator))


def interval(lower, upper):
    return endpoint(lower).union(endpoint(upper))


def scalar_profile(correlation):
    """Exact rational coefficients of A(rho), where h=pi*A."""
    coefficients = [fmpq(0)] * (2 * COUNT)
    for n, value in enumerate(correlation):
        coefficients[2 * n + 1] = fq(value)
    rho = fmpq_poly(coefficients)
    square = (rho * rho).truncate(2 * COUNT)
    power = rho
    output = fmpq_poly()
    coefficient = Q(1, 4)
    for n in range(COUNT):
        output = (output + power * fq(coefficient)).truncate(2 * COUNT)
        power = (power * square).truncate(2 * COUNT)
        coefficient *= Q((2 * n + 1) ** 2, 4 * (n + 1) * (n + 2))
    return tuple(
        Q(int(output[2 * n + 1].p), int(output[2 * n + 1].q))
        for n in range(COUNT)
    )


def prepared_candidate():
    saved = np.load(CANDIDATE)
    assert tuple(int(x) for x in saved["source_ids"]) == EXPECTED_IDS
    raw_weights = tuple(rational(x) for x in saved["primal_weights"])
    weight_scale = INWARD / sum(map(abs, raw_weights))
    weights = tuple(weight_scale * value for value in raw_weights)
    assert sum(map(abs, weights)) == INWARD

    correlations = []
    for branch in range(8):
        values = tuple(
            Q(0) if abs(float(x)) < 1e-10 else rational(x)
            for x in saved["scalar_correlations"][branch]
        )
        scale = INWARD / max(Q(1), sum(map(abs, values)))
        values = tuple(scale * x for x in values)
        assert sum(map(abs, values)) <= INWARD
        correlations.append(values)
    # The zero-weight contact slot is index 8.  The following 120 slots
    # are exact monomial correlations, each moved strictly inward.
    for n in range(COUNT):
        values = [Q(0)] * COUNT
        values[n] = INWARD
        correlations.append(tuple(values))
    # Besides the radial slot, weights contains one zero contact slot.
    assert len(correlations) == len(weights) - 2
    return correlations, weights


def radial_boxes():
    table = np.load(TABLE)
    assert int(table["cutoff"]) == 25
    values = []
    for n in range(COUNT):
        values.append(
            acb(
                interval(table["real_lower"][n], table["real_upper"][n]),
                interval(table["imag_lower"][n], table["imag_upper"][n]),
            )
        )
    return values


def main():
    correlations, weights = prepared_candidate()
    scalar = [Q(0)] * COUNT
    # weights[8] is the unused contact slot and is exactly zero in the
    # rationalized candidate.  Correlation list branch 8 is mono_0.
    assert weights[8] == 0
    scalar_weights = weights[:8] + weights[9:-1]
    assert len(scalar_weights) == len(correlations)
    for weight, correlation in zip(scalar_weights, correlations):
        if weight == 0:
            continue
        profile = scalar_profile(correlation)
        for n in range(COUNT):
            scalar[n] += weight * profile[n]

    t = Q(356080252073, 10**12)
    c_real = (1 - t * t) / (1 + t * t)
    c_imag = 2 * t / (1 + t * t)
    conjugate_c = acb(exact_arb(c_real), -exact_arb(c_imag))
    beta = weights[-1]
    radial = radial_boxes()
    radial_kernel = [(conjugate_c * value * value).real for value in radial]
    mixture = [
        arb.pi() * exact_arb(scalar[n]) + exact_arb(beta) * radial_kernel[n]
        for n in range(COUNT)
    ]

    omitted = (-arb(25) / 2).exp().upper()
    radial0_error = 2 * abs(radial[0]).upper() * omitted + omitted**2
    pi_upper = Q(104348, 33215)
    contact_upper = Q(81255787, 10**8)
    gamma_upper = exact_arb(pi_upper * (1 + contact_upper) / 8)
    gain_lower = (
        mixture[0].lower()
        - abs(exact_arb(beta)).upper() * radial0_error.upper()
        - gamma_upper
    )
    assert gain_lower > arb("4.0476e-6")

    prefix = sum((abs(value).upper() for value in mixture[1:]), arb(0))
    radial_norm = sum(
        (abs(value).upper() ** 2 for value in radial[1:]), arb(0)
    ).sqrt()
    joint_kernel_error = 2 * radial_norm * omitted + omitted**2
    prefix += abs(exact_arb(beta)).upper() * joint_kernel_error
    assert prefix < arb("2.85e-9")

    print("second convex candidate low-mode certificate: PASS")
    print("absolute mixture mass  =", float(sum(map(abs, weights))))
    print("raw slope gain lower   =", gain_lower)
    print("nonlinear prefix upper =", prefix, "< 1e-8")
    print("radial weight          =", float(beta))


if __name__ == "__main__":
    main()
