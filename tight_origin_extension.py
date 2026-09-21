"""Load the certified 25 <= s <= 36 correction to the pinned origin table."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from flint import acb, arb

import complex_upper_saha_cubic_exacttail_457e8_certificate as base


TABLE = Path(__file__).resolve().parent / "origin_tail_25_36.npz"
EXPECTED_SHA256 = (
    "8bcc654f4fd44881a7b487fe1336a8905aaa2d1b52deca45e622e1c397fb03aa"
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def boxes(prefix):
    assert sha256(TABLE) == EXPECTED_SHA256
    saved = np.load(TABLE)
    assert int(saved["count"]) == base.COUNT
    assert int(saved["start"]) == 25 and int(saved["cutoff"]) == 36
    correction = tuple(
        acb(
            base.interval(saved[f"{prefix}_real_lower"][n],
                          saved[f"{prefix}_real_upper"][n]),
            base.interval(saved[f"{prefix}_imag_lower"][n],
                          saved[f"{prefix}_imag_upper"][n]),
        )
        for n in range(base.COUNT)
    )
    return tuple(a + b for a, b in zip(base.boxes(prefix), correction))


def directed_origin_mixture(scalar, beta):
    """Return q0 lower and nonlinear-prefix upper through mode 119."""
    left, right = boxes("left"), boxes("right")
    radial = tuple((a * b.conjugate()).real for a, b in zip(left, right))
    mixture = tuple(
        arb.pi() * base.aq(scalar[n]) + base.aq(beta) * radial[n]
        for n in range(base.COUNT)
    )
    omitted = (-arb(36) / 2).exp().upper()
    slope_error = omitted * (
        abs(left[0]).upper() + abs(right[0]).upper() + omitted
    )
    q0 = mixture[0].lower() - abs(base.aq(beta)).upper() * slope_error
    left_norm = sum((abs(x).upper() ** 2 for x in left[1:]), arb(0)).sqrt()
    right_norm = sum((abs(x).upper() ** 2 for x in right[1:]), arb(0)).sqrt()
    joint_error = omitted * (left_norm + right_norm) + omitted**2
    prefix = sum((abs(x).upper() for x in mixture[1:]), arb(0))
    prefix += abs(base.aq(beta)).upper() * joint_error
    return q0, prefix
