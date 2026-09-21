#!/usr/bin/env python3
"""Arb coefficient enclosures for the second radial-phase candidate.

The decimal polynomial coefficients below are exact terminating rationals.
For u(s)=P(s)/|P(s)| this encloses, after s=t^2,

  a_n = int_0^25 e^-s (-1)^n sqrt(s) L_n^1(s) u(s)/sqrt(n+1) ds.

The jointly omitted input has coefficient l2 norm at most exp(-25/2).
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os

import numpy as np


PRE_STR = (
    "-0.08698387653870097",
    "-1.2330310998031548",
    "0.7731241827583651",
    "-0.22194697498795207",
    "0.02812176806963062",
    "-0.000900511816912206",
    "-0.00016426639538512312",
    "0.00001934672964561798",
    "-0.0000007806468542918022",
    "0.000000010807705028012223",
)

PIM_STR = (
    "-0.9438672688639868",
    "1.7787014426224697",
    "-1.2883230878434588",
    "0.37871541533769754",
    "-0.046163071426838245",
    "0.00035393667390378955",
    "0.0004808219961183191",
    "-0.00004805019315921306",
    "0.0000018538964354898714",
    "-0.0000000252261077693192",
)


def coefficient(index):
    # Import inside each process; FLINT state is not shared across workers.
    from flint import acb, arb, ctx

    ctx.dps = 40
    pre = tuple(arb(x) for x in PRE_STR)
    pim = tuple(arb(x) for x in PIM_STR)

    def polynomial(coefficients, argument):
        value = coefficients[-1]
        for coefficient_value in reversed(coefficients[:-1]):
            value = value * argument + coefficient_value
        return value

    sign = -1 if index & 1 else 1
    normalization = arb(index + 1).sqrt()

    def integrand(t, analytic):
        s = t * t
        real = polynomial(pre, s)
        imag = polynomial(pim, s)
        modulus_squared = real * real + imag * imag
        phase = (real + 1j * imag) / modulus_squared.sqrt(analytic=analytic)
        laguerre = s.laguerre_l(index, 1)
        return 2 * s * (-s).exp() * laguerre * phase * sign / normalization

    value = acb(0)
    for part in range(50):
        value += acb.integral(
            integrand,
            arb(part) / 10,
            arb(part + 1) / 10,
            abs_tol=arb("1e-18"),
            rel_tol=arb("1e-18"),
            eval_limit=200000,
            depth_limit=25,
            deg_limit=120,
        )

    def endpoints(ball):
        return (
            math.nextafter(float(ball.lower()), -math.inf),
            math.nextafter(float(ball.upper()), math.inf),
        )

    real_lower, real_upper = endpoints(value.real)
    imag_lower, imag_upper = endpoints(value.imag)
    return index, real_lower, real_upper, imag_lower, imag_upper, str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output", default="complex_upper_convex_2e6_radial_coefficients.npz"
    )
    args = parser.parse_args()
    with mp.Pool(args.jobs) as pool:
        rows = []
        for row in pool.imap_unordered(coefficient, range(args.count)):
            rows.append(row)
            print(row[0], row[-1], flush=True)
    rows.sort()
    real_lower = np.asarray([row[1] for row in rows])
    real_upper = np.asarray([row[2] for row in rows])
    imag_lower = np.asarray([row[3] for row in rows])
    imag_upper = np.asarray([row[4] for row in rows])
    np.savez(
        args.output,
        real_lower=real_lower,
        real_upper=real_upper,
        imag_lower=imag_lower,
        imag_upper=imag_upper,
        cutoff=np.array(25),
        l2_tail_bound=np.array(math.nextafter(math.exp(-12.5), math.inf)),
        defining_real=np.array(PRE_STR),
        defining_imag=np.array(PIM_STR),
    )
    print("saved", args.output)
    print(
        "maximum Arb box width",
        max(np.max(real_upper - real_lower), np.max(imag_upper - imag_lower)),
    )


if __name__ == "__main__":
    main()
