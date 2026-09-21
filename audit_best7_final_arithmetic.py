#!/usr/bin/env python3
"""Exact-rational audit of the final seven-atom gain.

The directed coefficient bounds Q0_LO and E_HI are the only certified
numerical inputs.  Branch normalization, the contact-point comparison, the
gain, and the reciprocal bound are checked below with fractions.Fraction.
"""

from decimal import Decimal, getcontext
from fractions import Fraction as Q


def decimal(value: Q) -> str:
    getcontext().prec = 70
    return str(Decimal(value.numerator) / Decimal(value.denominator))


WEIGHTS = tuple(map(Q, (
    "0.983543086263686583",
    "0.0124174018143118538",
    "-0.0030311909556774751",
    "0.000593632758261406976",
    "0.0000169358532612748746",
    "0.000397752354801503847",
)))

Q0_LO = Q("0.71179640805640433452")
PREFIX_HI = Q("1.074829404498064e-6")
HIGH_TAIL_HI = Q("6.319618149454564e-8")
FIXED_TAIL_HI = Q("1.040224125770398e-7")
E_HI = Q("1.242047998569650e-6")
GAMMA_0 = Q("0.71179516600840569540")


def hypergeom_partial(x: Q, last_index: int = 63) -> Q:
    """Return sum_{n=0}^last_index A_n*x^(2n), exactly."""
    x_squared = x * x
    term = Q(1)
    total = term
    for n in range(last_index):
        term *= Q((2 * n + 1) ** 2, 4 * (n + 1) * (n + 2))
        term *= x_squared
        total += term
    return total


def atan_partial(inv_x: int, last_index: int) -> Q:
    """Alternating partial sum for atan(1/inv_x), exactly."""
    return sum((
        Q((-1) ** k, (2 * k + 1) * inv_x ** (2 * k + 1))
        for k in range(last_index + 1)
    ), Q(0))


def main() -> None:
    mass = sum(map(abs, WEIGHTS))
    assert mass == Q(
        1_250_000_000_000_000_121_997,
        1_250_000_000_000_000_000_000,
    )

    # These are the three outward-rounded component bounds printed in the
    # manuscript.  Their exact decimal sum is still strictly below E_HI.
    assembled = PREFIX_HI + HIGH_TAIL_HI + FIXED_TAIL_HI
    assert assembled < E_HI

    # qhat_0>Q0_LO and ||Ehat||_1<E_HI imply that the inverse-repair radius
    # is strictly larger than slope_lo.
    slope_lo = (Q0_LO - E_HI) / mass
    assert slope_lo > GAMMA_0

    # Let Phi(x)=2*x*2F1(1/2,1/2;2;x^2)-1-x.  Every term omitted from the
    # degree-63 partial sum is positive, so partial_phi>0 proves Phi(x_+)>0.
    # Since Phi'(x)>0 on (0,1), the contact point x_H is smaller than x_+.
    x_plus = Q(812_557_858_821_473, 10**15)
    partial_phi = 2 * x_plus * hypergeom_partial(x_plus) - 1 - x_plus
    assert partial_phi > Q("7.0246e-17")

    # Machin's identity.  The k=14 partial sum for atan(1/5) is an upper
    # bound and the k=3 partial sum for atan(1/239) is a lower bound.
    pi_upper = 16 * atan_partial(5, 14) - 4 * atan_partial(239, 3)
    gamma_h_upper = pi_upper * (1 + x_plus) / 8
    printed_gamma_h_upper = Q("0.71178980668499814812")
    assert gamma_h_upper < printed_gamma_h_upper

    claimed_gain = Q("5.3593234075e-6")
    assert GAMMA_0 - printed_gamma_h_upper - claimed_gain == Q(
        591, 12_500_000_000_000_000_000
    )

    k_claim = Q("1.404898554745441")
    assert GAMMA_0 * k_claim - 1 == Q(
        2_068_286_482_007_923_357,
        5_000_000_000_000_000_000_000_000_000_000_000,
    )

    print("exact final-arithmetic audit: PASS")
    print("M =", mass, "=", decimal(mass))
    print("assembled nonlinear upper =", decimal(assembled))
    print("E_HI - assembled =", decimal(E_HI - assembled))
    print("slope lower =", slope_lo, "=", decimal(slope_lo))
    print("slope margin over Gamma_0 =", decimal(slope_lo - GAMMA_0))
    print("degree-63 partial Phi(x_+) =", decimal(partial_phi))
    print("rational gamma_H upper =", decimal(gamma_h_upper))
    print("gain over claimed amount =", decimal(
        GAMMA_0 - printed_gamma_h_upper - claimed_gain
    ))
    print("reciprocal upper =", decimal(1 / GAMMA_0))
    print("reciprocal margin =", decimal(k_claim - 1 / GAMMA_0))


if __name__ == "__main__":
    main()
