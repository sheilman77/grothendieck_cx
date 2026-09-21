#!/usr/bin/env python3
"""Arb enclosures for the origin-layer coefficient tails on 25 <= s <= 36.

The pinned table integrates 0 <= s <= 25.  This companion table integrates
5 <= t <= 6 after the substitution s=t^2, allowing the remaining L2 tail
bound to be reduced from exp(-25/2) to exp(-36/2).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np


CERT = Path(__file__).resolve().parent / "support_compression_certificate"
sys.path.insert(0, str(CERT))
from complex_upper_origin_layer_definitions import LEFT, RIGHT  # noqa: E402


def coefficient(task):
    side, index = task
    from flint import acb, arb, ctx, fmpq

    ctx.dps = 50
    exact = LEFT if side == 0 else RIGHT
    polynomial_real = tuple(
        arb(fmpq(real.numerator, real.denominator)) for real, _imag in exact
    )
    polynomial_imag = tuple(
        arb(fmpq(imag.numerator, imag.denominator)) for _real, imag in exact
    )

    def evaluate(coefficients, argument):
        value = coefficients[-1]
        for c in reversed(coefficients[:-1]):
            value = value * argument + c
        return value

    sign = -1 if index & 1 else 1
    normalization = arb(index + 1).sqrt()

    def integrand(t, analytic):
        s = t * t
        real = evaluate(polynomial_real, s)
        imag = evaluate(polynomial_imag, s)
        phase = (real + 1j * imag) / (real * real + imag * imag).sqrt(
            analytic=analytic
        )
        return (
            2 * s * (-s).exp() * s.laguerre_l(index, 1)
            * phase * sign / normalization
        )

    value = acb(0)
    for part in range(50, 60):
        value += acb.integral(
            integrand,
            arb(part) / 10,
            arb(part + 1) / 10,
            abs_tol=arb("2e-19"),
            rel_tol=arb("2e-19"),
            eval_limit=300000,
            depth_limit=28,
            deg_limit=140,
        )

    def endpoints(ball):
        lower = math.nextafter(float(ball.lower()), -math.inf)
        upper = math.nextafter(float(ball.upper()), math.inf)
        lower_exact = arb(fmpq(*lower.as_integer_ratio()))
        upper_exact = arb(fmpq(*upper.as_integer_ratio()))
        assert lower_exact.union(upper_exact).contains(ball)
        return lower, upper

    rl, ru = endpoints(value.real)
    il, iu = endpoints(value.imag)
    return side, index, rl, ru, il, iu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", default="origin_tail_25_36.npz")
    args = parser.parse_args()
    tasks = [(side, index) for side in range(2) for index in range(args.count)]
    with mp.Pool(args.jobs) as pool:
        rows = list(pool.imap_unordered(coefficient, tasks))
    rows.sort()
    arrays = {}
    for side, name in ((0, "left"), (1, "right")):
        selected = [row for row in rows if row[0] == side]
        arrays[name + "_real_lower"] = np.asarray([row[2] for row in selected])
        arrays[name + "_real_upper"] = np.asarray([row[3] for row in selected])
        arrays[name + "_imag_lower"] = np.asarray([row[4] for row in selected])
        arrays[name + "_imag_upper"] = np.asarray([row[5] for row in selected])
    np.savez(
        args.output,
        **arrays,
        count=np.array(args.count),
        start=np.array(25),
        cutoff=np.array(36),
        joint_l2_tail_bound=np.array(math.nextafter(math.exp(-18), math.inf)),
    )
    widths = [
        np.max(arrays[key.replace("lower", "upper")] - value)
        for key, value in arrays.items() if key.endswith("lower")
    ]
    print("saved", args.output)
    print("maximum box width", max(widths))


if __name__ == "__main__":
    main()
