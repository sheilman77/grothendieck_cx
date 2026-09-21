#!/usr/bin/env python3
"""Arb enclosures for both origin-layer radial coefficient sequences."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

import numpy as np

from complex_upper_origin_layer_definitions import LEFT, RIGHT


def coefficient(task):
    side, index = task
    from flint import acb, arb, ctx, fmpq

    ctx.dps = 40
    exact = LEFT if side == 0 else RIGHT
    # Keep the real- and imaginary-coefficient polynomials separate.  On
    # the complex balls used by Arb's analytic integrator, taking
    # ``value.real`` after evaluating one complex polynomial would not be
    # the analytic continuation of its real-coefficient part.
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
        modulus_squared = real * real + imag * imag
        phase = (real + 1j * imag) / modulus_squared.sqrt(
            analytic=analytic
        )
        laguerre = s.laguerre_l(index, 1)
        return 2 * s * (-s).exp() * laguerre * phase * sign / normalization

    value = acb(0)
    for part in range(50):
        value += acb.integral(
            integrand,
            arb(part) / 10,
            arb(part + 1) / 10,
            abs_tol=arb("2e-17"),
            rel_tol=arb("2e-17"),
            eval_limit=300000,
            depth_limit=28,
            deg_limit=140,
        )

    def endpoints(ball):
        lower = math.nextafter(float(ball.lower()), -math.inf)
        upper = math.nextafter(float(ball.upper()), math.inf)
        # Audit the binary64 serialization itself.  This makes the saved
        # endpoint direction independent of assumptions about how
        # python-flint converts an Arb endpoint to float.
        lower_exact = arb(fmpq(*lower.as_integer_ratio()))
        upper_exact = arb(fmpq(*upper.as_integer_ratio()))
        assert lower_exact.union(upper_exact).contains(ball)
        return lower, upper

    rl, ru = endpoints(value.real)
    il, iu = endpoints(value.imag)
    return side, index, rl, ru, il, iu, str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output", default="complex_upper_origin_layer_coefficients.npz"
    )
    args = parser.parse_args()
    tasks = [(side, index) for side in range(2) for index in range(args.count)]
    with mp.Pool(args.jobs) as pool:
        rows = []
        for row in pool.imap_unordered(coefficient, tasks):
            rows.append(row)
            print(row[0], row[1], row[-1], flush=True)
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
        cutoff=np.array(25),
        joint_l2_tail_bound=np.array(
            math.nextafter(math.exp(-12.5), math.inf)
        ),
    )
    widths = [
        np.max(arrays[key.replace("lower", "upper")] - value)
        for key, value in arrays.items() if key.endswith("lower")
    ]
    print("saved", args.output)
    print("maximum box width", max(widths))


if __name__ == "__main__":
    main()
