#!/usr/bin/env python3
"""Rigorous exact-tail certificate for a +4.57e-6 complex bound.

The added scalar branch uses the exact complex correlation

    rho_*(z) = (3 z - 16 z |z|^2) / 19.

For this branch the coefficient tail is evaluated exactly through radial
mode 600, after which a certified Cauchy estimate on |z|=41/40 is used.
Every generator is charged for its omitted tail before the finite prefix is
repaired in the (5/8)-ball.  All saved floating-point values are interpreted
as exact decimal rationals and every final comparison is rigorous.
"""

from __future__ import annotations

import argparse
from fractions import Fraction as Q
import hashlib
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from flint import acb, arb, ctx, fmpq, fmpq_poly

from complex_upper_convex_2e6_low_certificate import scalar_profile
import complex_upper_convex_2e6_scalar_tail_certificate as scalar_check
from complex_upper_origin_layer_definitions import check_origin_identity


ctx.dps = 80
COUNT = 120
TERMINAL = 600
DELTA = Q(5, 8)
INVERSE_BALL = Q(1, 1) / DELTA
INWARD = Q(10**12 - 1, 10**12)
PI_UP = Q(104348, 33215)
PI_LO = Q(103993, 33102)
CONTACT_UP = Q(81255787, 10**8)
CONTACT_LO = Q(81255785, 10**8)
GAMMA_UP = PI_UP * (1 + CONTACT_UP) / 8
GAMMA_LO = PI_LO * (1 + CONTACT_LO) / 8

ROOT = Path(__file__).resolve().parent
CANDIDATE = ROOT / "complex_upper_saha_cubic_exacttail_candidate_n120.npz"
ORIGIN_CANDIDATE = ROOT / "complex_upper_origin_layer_candidate_n120.npz"
TABLE = ROOT / "complex_upper_origin_layer_coefficients_n120_analytic.npz"
EXPECTED_IDS = (39, 86, 88, 94, 121, 123, 191, 195)
EXPECTED_CANDIDATE_SHA256 = (
    "f827fcb96dce8fbc98e15a701c28fe5e405786a8914f75f7d0f664296e3ccdc2"
)
EXPECTED_ORIGIN_CANDIDATE_SHA256 = (
    "1c717ce0dd5d587b1f9b52cd1ce91f77aa4520a5211e9ed6da82abe4d661f62f"
)
EXPECTED_TABLE_SHA256 = (
    "fc29fa555fb66c5cd6a2ac53dbd2508207b6058cfe0b339163faadeb85ad54da"
)
PREFIX_RESERVE = Q(123, 10**8)      # 1.23e-6 in the kernel A-norm.
ORIGIN_TAIL = Q(103, 10**5)        # Per unit origin-layer weight.
CUBIC_TAIL = Q(58, 10**8)          # Per unit distinct-cubic weight.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rational_float(value) -> Q:
    return Q(repr(float(value)))


def aq(value: Q) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def endpoint(value) -> arb:
    numerator, denominator = float(value).as_integer_ratio()
    return arb(fmpq(numerator, denominator))


def interval(lower, upper) -> arb:
    return endpoint(lower).union(endpoint(upper))


def check_inputs():
    assert sha256(CANDIDATE) == EXPECTED_CANDIDATE_SHA256
    assert sha256(ORIGIN_CANDIDATE) == EXPECTED_ORIGIN_CANDIDATE_SHA256
    assert sha256(TABLE) == EXPECTED_TABLE_SHA256
    saved = np.load(CANDIDATE)
    origin = np.load(ORIGIN_CANDIDATE)
    assert int(saved["count"]) == COUNT
    assert tuple(map(int, saved["source_ids"])) == EXPECTED_IDS
    assert saved["primal_weights"].shape == (8 + COUNT + 2,)
    assert saved["primal_residual"].shape == (COUNT,)
    assert saved["scalar_correlations"].shape == (8, COUNT)
    assert saved["cubic_correlation"].shape == (COUNT,)
    assert np.array_equal(
        saved["scalar_correlations"], origin["scalar_correlations"]
    )
    assert saved["labels"][-2] == "origin-layer"
    assert saved["labels"][-1] == "distinct-cubic-3-minus-16-over-19"
    assert np.count_nonzero(saved["cubic_correlation"]) == 2
    assert float(saved["cubic_correlation"][0]) == 3 / 19
    assert float(saved["cubic_correlation"][1]) == -16 / 19
    assert float(saved["search_cubic_tail"]) == 5.8e-7

    table = np.load(TABLE)
    assert int(table["count"]) == COUNT and int(table["cutoff"]) == 25
    for side in ("left", "right"):
        for part in ("real", "imag"):
            lower = table[f"{side}_{part}_lower"]
            upper = table[f"{side}_{part}_upper"]
            assert lower.shape == upper.shape == (COUNT,)
            assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
            assert np.all(lower <= upper)
    return saved


def exact_correlations(saved):
    old = []
    for row in saved["scalar_correlations"]:
        values = tuple(
            Q(0) if abs(float(x)) < 1e-10 else rational_float(x) for x in row
        )
        factor = INWARD / max(Q(1), sum(map(abs, values)))
        values = tuple(factor * value for value in values)
        assert sum(map(abs, values)) <= INWARD
        old.append(values)
    cubic = (Q(3, 19), Q(-16, 19)) + (Q(0),) * (COUNT - 2)
    assert sum(map(abs, cubic)) == 1
    return tuple(old), cubic


def h_over_pi_coefficients(number=COUNT):
    output = [Q(1, 4)]
    for j in range(number - 1):
        output.append(
            output[-1]
            * Q((2 * j + 1) ** 2, 4 * (j + 1) * (j + 2))
        )
    return tuple(output)


def old_scalar_tails(correlations, check_boundary=False):
    """Exact per-unit tail majorants for the eight inherited branches."""
    output = []
    for branch, identifier in enumerate(EXPECTED_IDS):
        coefficients = correlations[branch]
        if identifier == 88:
            assert all(
                value == 0 for n, value in enumerate(coefficients)
                if n not in (0, 7)
            )
            a, b = abs(coefficients[0]), abs(coefficients[7])
            assert a + b <= 1
            finite, coefficient = Q(0), Q(1, 4)
            for j in range(COUNT):
                power = 2 * j + 1
                # Output radial index is j+7k, since ordinary degree is
                # (2j+1)+14k.  Keep exactly the terms at index >=120.
                first = max(0, (2 * COUNT + 1 - power + 13) // 14)
                finite += coefficient * sum(
                    Q(math.comb(power, k)) * a ** (power - k) * b**k
                    for k in range(first, power + 1)
                )
                coefficient *= Q(
                    (2 * j + 1) ** 2, 4 * (j + 1) * (j + 2)
                )
            tail = PI_UP * finite + Q(1, 4 * COUNT)
        else:
            radius = Q(109, 100) if identifier == 123 else Q(111, 100)
            if check_boundary:
                scalar_check.root_check(coefficients, radius)
                scalar_check.boundary_check(coefficients, radius)
            tail = Q(10) * radius ** (-(2 * COUNT + 1)) / (
                1 - radius ** (-2)
            )
        output.append(tail)
    return tuple(output)


def cubic_exact_tail(cubic):
    """Exact modes 120..600 plus a certified Cauchy remainder."""
    radius = Q(41, 40)
    scalar_check.root_check(cubic, radius)
    weighted_norm, minimum = scalar_check.boundary_check(cubic, radius)
    assert weighted_norm < Q(5, 4)
    assert minimum > arb("0.01")

    # rho is represented in ordinary odd degree.  The coefficient of
    # x^(2n+1) is the charge-one radial coefficient at index n.
    rho = fmpq_poly([
        fmpq(0),
        fmpq(cubic[0].numerator, cubic[0].denominator),
        fmpq(0),
        fmpq(cubic[1].numerator, cubic[1].denominator),
    ])
    square = (rho * rho).truncate(2 * TERMINAL + 2)
    power = rho
    profile = fmpq_poly()
    h_over_pi = Q(1, 4)
    for j in range(TERMINAL + 1):
        profile = (
            profile
            + power * fmpq(h_over_pi.numerator, h_over_pi.denominator)
        ).truncate(2 * TERMINAL + 2)
        power = (power * square).truncate(2 * TERMINAL + 2)
        h_over_pi *= Q(
            (2 * j + 1) ** 2, 4 * (j + 1) * (j + 2)
        )

    # Terms with outer index j>600 have minimal ordinary degree >1201,
    # so the following coefficients are the complete exact coefficients.
    finite = Q(0)
    for n in range(COUNT, TERMINAL + 1):
        coefficient = profile[2 * n + 1]
        finite += abs(Q(int(coefficient.p), int(coefficient.q)))

    # The root and segment-distance checks above validate the same Euler
    # integral continuation used by the inherited scalar-tail certificate,
    # and give |h(rho(z))|<10 on |z|=41/40.  Cauchy's estimate therefore
    # sums the uncomputed odd coefficients geometrically.
    remainder = Q(10) * radius ** (-(2 * TERMINAL + 3)) / (
        1 - radius ** (-2)
    )
    tail = PI_UP * finite + remainder
    assert tail < CUBIC_TAIL
    return tail, finite, remainder, weighted_norm, minimum


def tail_vector(correlations, cubic, check_boundary=False):
    tails = list(old_scalar_tails(correlations, check_boundary))
    for m in range(COUNT):
        divisor = 2 * m + 1
        first = (COUNT - m + divisor - 1) // divisor
        assert m + divisor * (first - 1) < COUNT <= m + divisor * first
        tails.append(Q(1, 4 * first))

    # Endpoint cancellation removes the slow h_n term.  The two certified
    # energies are <13, and sum_{n>=120} h_n/(2n+1)^2 is at most
    # 1/[48*119^3].
    product = check_origin_identity()
    assert product[0] == 0 and product[1] != 0
    energy = Q(13)
    weighted = Q(1, 48 * (COUNT - 1) ** 3)
    room = ORIGIN_TAIL - energy / (2 * COUNT + 1) ** 2
    assert room > 0 and room * room > 4 * energy * weighted
    tails.append(ORIGIN_TAIL)

    cubic_data = cubic_exact_tail(cubic)
    tails.append(CUBIC_TAIL)
    assert len(tails) == 8 + COUNT + 2
    return tuple(tails), cubic_data


def exact_candidate(saved, tails):
    raw = tuple(rational_float(value) for value in saved["primal_weights"])
    charged_mass = sum(
        abs(weight) * (1 + INVERSE_BALL * tail)
        for weight, tail in zip(raw, tails)
    )
    numerator = INWARD - INVERSE_BALL * PREFIX_RESERVE
    assert numerator > 0
    scale = numerator / charged_mass
    assert 0 < scale < 1
    weights = tuple(scale * value for value in raw)
    total = sum(
        abs(weight) * (1 + INVERSE_BALL * tail)
        for weight, tail in zip(weights, tails)
    ) + INVERSE_BALL * PREFIX_RESERVE
    assert total == INWARD
    return weights, scale, charged_mass


def boxes(prefix):
    saved = np.load(TABLE)
    return tuple(
        acb(
            interval(saved[f"{prefix}_real_lower"][n],
                     saved[f"{prefix}_real_upper"][n]),
            interval(saved[f"{prefix}_imag_lower"][n],
                     saved[f"{prefix}_imag_upper"][n]),
        )
        for n in range(COUNT)
    )


def low_component(correlations, cubic, weights):
    scalar = [Q(0)] * COUNT
    for weight, correlation in zip(weights[:8], correlations):
        if weight:
            profile = scalar_profile(correlation)
            for n in range(COUNT):
                scalar[n] += weight * profile[n]

    h = h_over_pi_coefficients()
    for m, weight in enumerate(weights[8:8 + COUNT]):
        if not weight:
            continue
        divisor = 2 * m + 1
        for j in range(COUNT):
            n = m + divisor * j
            if n >= COUNT:
                break
            scalar[n] += weight * h[j] * INWARD ** (2 * j + 1)

    cubic_profile = scalar_profile(cubic)
    cubic_weight = weights[-1]
    for n in range(COUNT):
        scalar[n] += cubic_weight * cubic_profile[n]

    left, right = boxes("left"), boxes("right")
    beta = weights[-2]
    radial = tuple((a * b.conjugate()).real for a, b in zip(left, right))
    mixture = tuple(
        arb.pi() * aq(scalar[n]) + aq(beta) * radial[n]
        for n in range(COUNT)
    )

    omitted = (-arb(25) / 2).exp().upper()
    slope_error = omitted * (
        abs(left[0]).upper() + abs(right[0]).upper() + omitted
    )
    slope = mixture[0].lower() - abs(aq(beta)).upper() * slope_error
    gain = slope - aq(GAMMA_UP)

    left_norm = sum((abs(x).upper() ** 2 for x in left[1:]), arb(0)).sqrt()
    right_norm = sum((abs(x).upper() ** 2 for x in right[1:]), arb(0)).sqrt()
    joint_error = omitted * (left_norm + right_norm) + omitted**2
    prefix = sum((abs(value).upper() for value in mixture[1:]), arb(0))
    prefix += abs(aq(beta)).upper() * joint_error
    assert prefix < aq(PREFIX_RESERVE)
    assert gain > arb("4.57e-6")
    return slope, gain, prefix


def final_assembly(weights, tails, slope, gain):
    branch_mass = sum(map(abs, weights))
    tail_mass = sum(abs(w) * t for w, t in zip(weights, tails))
    total_probability = branch_mass + INVERSE_BALL * (
        tail_mass + PREFIX_RESERVE
    )
    assert total_probability == INWARD < 1
    # The unused probability is filled with the zero kernel.  Tail and
    # prefix repairs have zero first coefficient, so the slope is unchanged.
    assert gain > arb("4.57e-6")
    reciprocal = Q(1) / (GAMMA_LO + Q(457, 10**8))
    assert reciprocal < Q(140490012, 10**8)
    print("exact-tail distinct-correlation +4.57e-6 certificate: PASS")
    print("exact branch mass             =", float(branch_mass))
    print("weighted far-tail reserve     =", float(tail_mass))
    print("finite-prefix reserve         =", float(PREFIX_RESERVE))
    print("total probability             =", float(total_probability), "< 1")
    print("certified slope lower         >", slope)
    print("certified gain lower          >", gain, "> 4.57e-6")
    print("reciprocal                    <", float(reciprocal), "< 1.40490012")
    print("THEOREM: K_G^C <= (gamma_H + 4.57e-6)^-1 < 1.40490012")


def run(script, *arguments):
    print(f"\n=== {script} {' '.join(arguments)} ===", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-components", action="store_true")
    parser.add_argument("--check-boundaries", action="store_true")
    args = parser.parse_args()
    saved = check_inputs()
    correlations, cubic = exact_correlations(saved)
    tails, cubic_data = tail_vector(
        correlations, cubic, check_boundary=args.check_boundaries
    )
    weights, scale, charged_mass = exact_candidate(saved, tails)
    slope, gain, prefix = low_component(correlations, cubic, weights)
    if args.run_components:
        run("complex_haagerup_strict_normal_cone_certificate.py")
        run("complex_dirichlet_inverse_l1_certificate.py")
        run("complex_upper_convex_2e6_polynomial_certificate.py")
        run("complex_upper_origin_layer_energy_certificate.py")
    actual_cubic, finite, remainder, weighted_norm, minimum = cubic_data
    print("candidate SHA256              =", sha256(CANDIDATE))
    print("exact cubic finite |tail|/pi  =", float(finite))
    print("exact cubic Cauchy remainder  =", float(remainder))
    print("exact cubic tail              =", float(actual_cubic), "<", float(CUBIC_TAIL))
    print("cubic boundary weighted norm  =", float(weighted_norm))
    print("cubic segment distance        >", minimum)
    print("raw charged mass              =", float(charged_mass))
    print("common exact scale            =", float(scale))
    print("actual finite prefix          <", prefix)
    final_assembly(weights, tails, slope, gain)


if __name__ == "__main__":
    main()
