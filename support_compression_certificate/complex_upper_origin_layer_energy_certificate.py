#!/usr/bin/env python3
"""Arb certificate for the two origin-layer radial Sobolev energies."""

from __future__ import annotations

from flint import acb, arb, ctx, fmpq

from complex_upper_origin_layer_definitions import A, B, DELTA, LEFT, RIGHT


ctx.dps = 45


def exact(value):
    return arb(fmpq(value.numerator, value.denominator))


def complex_polynomial(coefficients):
    return tuple(
        (exact(real), exact(imag)) for real, imag in coefficients
    )


def derivative(coefficients):
    return tuple(
        (arb(k) * coefficients[k][0], arb(k) * coefficients[k][1])
        for k in range(1, len(coefficients))
    )


def evaluate(coefficients, argument):
    real, imag = coefficients[-1]
    for cr, ci in reversed(coefficients[:-1]):
        real, imag = real * argument + cr, imag * argument + ci
    return real, imag


def energy(polynomial, label, layer_factor):
    first = derivative(polynomial)
    second = derivative(first)
    p0r, p0i = polynomial[0]
    norm0 = (p0r**2 + p0i**2).sqrt()
    u0r, u0i = p0r / norm0, p0i / norm0

    def integrand(s, analytic):
        real, imag = evaluate(polynomial, s)
        real1, imag1 = evaluate(first, s)
        real2, imag2 = evaluate(second, s)
        q = real**2 + imag**2
        q1 = 2 * (real * real1 + imag * imag1)
        q2 = 2 * (real1**2 + imag1**2 + real * real2 + imag * imag2)
        root = q.sqrt(analytic=analytic)
        a = q1 / (2 * q)
        ap = q2 / (2 * q) - q1**2 / (2 * q**2)
        ur, ui = real / root, imag / root
        u1r, u1i = (real1 - a * real) / root, (imag1 - a * imag) / root
        u2r = (real2 - ap * real - 2 * a * real1 + a**2 * real) / root
        u2i = (imag2 - ap * imag - 2 * a * imag1 + a**2 * imag) / root
        nr = -2 * s * u2r + (2 * s - 2) * u1r + (ur - u0r) / (2 * s)
        ni = -2 * s * u2i + (2 * s - 2) * u1i + (ui - u0i) / (2 * s)
        return (nr**2 + ni**2) * (-s).exp()

    epsilon = arb(1) / 10**7
    cutoff = arb(50)
    middle = acb(0)
    # Geometric panels around the layer, followed by ordinary panels.
    rational_edges = [
        epsilon,
        arb(1) / 10**6,
        arb(1) / 10**5,
        arb(1) / 10**4,
        arb(1) / 1000,
        arb(1) / 100,
        arb(1) / 10,
        arb(1),
    ]
    edges = rational_edges + [arb(k) for k in range(2, 51)]
    for left, right in zip(edges[:-1], edges[1:]):
        middle += acb.integral(
            integrand,
            left,
            right,
            abs_tol=arb("2e-15"),
            rel_tol=arb("2e-15"),
            eval_limit=500000,
            depth_limit=30,
            deg_limit=160,
        )
    assert middle.imag.contains(0)

    # P contributes <12 and <30 to the first two logarithmic derivatives.
    # The extra linear factor has root distance delta*|Im(layer_factor)|.
    distance = exact(DELTA * abs(layer_factor[1]))
    m1 = arb(12) + 1 / distance
    m2_phase = arb(30) + 1 / distance**2
    m2_function = m2_phase + m1**2
    near_bound = 2 * epsilon * m2_function + (2 * epsilon + arb("2.5")) * m1
    near = epsilon * near_bound**2

    slope = 2 * m2_function + 2 * m1
    intercept = 2 * m1 + 1 / cutoff
    far = (-cutoff).exp() * (
        slope**2 * (cutoff**2 + 2 * cutoff + 2)
        + 2 * slope * intercept * (cutoff + 1)
        + intercept**2
    )
    upper = middle.real.upper() + near.upper() + far.upper()
    print(label, "middle", middle)
    print(label, "near upper", near.upper())
    print(label, "far upper", far.upper())
    print(label, "E2 upper", upper, "< 13")
    assert upper < arb(13)
    return upper


def main():
    left = energy(complex_polynomial(LEFT), "left", A)
    right = energy(complex_polynomial(RIGHT), "right", B)
    assert left < arb(13) and right < arb(13)
    print("origin-layer two-energy certificate: PASS")


if __name__ == "__main__":
    main()
