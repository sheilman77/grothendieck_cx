"""Exact rational definitions for the origin-layer radial phase pair."""

from __future__ import annotations

from fractions import Fraction as Q

from complex_upper_convex_2e6_radial_coefficients import PIM_STR, PRE_STR


DELTA = Q(11, 500)
CAYLEY = Q(-1775, 10000)
C_PARAMETER = Q(356080252073, 10**12)


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


def multiply(x, y):
    return (x[0] * y[0] - x[1] * y[1],
            x[0] * y[1] + x[1] * y[0])


def conjugate(x):
    return (x[0], -x[1])


def scale(x, scalar):
    return (scalar * x[0], scalar * x[1])


def unit_cayley(parameter):
    return (
        (1 - parameter * parameter) / (1 + parameter * parameter),
        2 * parameter / (1 + parameter * parameter),
    )


P = tuple((Q(real), Q(imag)) for real, imag in zip(PRE_STR, PIM_STR))
C = unit_cayley(C_PARAMETER)
A = unit_cayley(CAYLEY)

# u(0)^2 is rational even though u(0) itself contains a square root.
p0_square = multiply(P[0], P[0])
p0_norm_square = P[0][0] ** 2 + P[0][1] ** 2
u0_square = scale(p0_square, 1 / p0_norm_square)
z0 = multiply(conjugate(C), u0_square)
D = (z0[1], -z0[0])  # -i*z0
B = multiply(D, A)


def multiply_polynomials(first, second):
    output = [(Q(0), Q(0))] * (len(first) + len(second) - 1)
    for i, x in enumerate(first):
        for j, y in enumerate(second):
            output[i + j] = add(output[i + j], multiply(x, y))
    return tuple(output)


LEFT = multiply_polynomials(P, (scale(A, DELTA), (Q(1), Q(0))))
RIGHT = multiply_polynomials(
    tuple(multiply(C, conjugate(value)) for value in P),
    (scale(B, DELTA), (Q(1), Q(0))),
)


def check_origin_identity():
    # The kernel's h_n asymptotic coefficient is the real part of the
    # product of the two endpoint phases.  Squared moduli are positive, so
    # it suffices to check that LEFT(0)*conj(RIGHT(0)) is purely imaginary.
    product = multiply(LEFT[0], conjugate(RIGHT[0]))
    assert product[0] == 0
    assert product[1] != 0
    # Also pin all factors to the rational unit circle.
    for factor in (C, A, B, D):
        assert factor[0] ** 2 + factor[1] ** 2 == 1
    # For real s >= 0, the two extra factors s+DELTA*A/B cannot vanish:
    # their imaginary parts are fixed nonzero exact rationals.
    assert A[1] != 0 and B[1] != 0 and DELTA > 0
    return product


check_origin_identity()
