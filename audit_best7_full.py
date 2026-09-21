#!/usr/bin/env python3
"""Run every non-regeneration verifier used by the seven-atom theorem."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CERT = ROOT / "support_compression_certificate"


def run(script: Path, cwd: Path) -> None:
    print(f"\n=== {script.name} ===", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=cwd, check=True)


def main() -> None:
    # Fail early with a direct dependency message.
    try:
        import flint  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "python-flint >= 0.9.0 is required for the interval verifiers"
        ) from error

    # These two scripts prove the nonvanishing and visible-energy facts
    # behind the origin-tail constant.  The assembly verifier intentionally
    # consumes that constant and does not silently rerun these prerequisites.
    run(CERT / "complex_upper_convex_2e6_polynomial_certificate.py", CERT)
    run(CERT / "complex_upper_saha_visible_energy_refinement.py", CERT)

    # This checks the scalar branches, both pinned hashes, prefix, all tails,
    # exact selector mass, directed slope, gain, and reciprocal.
    run(ROOT / "audit_sparse6_altgroups.py", ROOT)

    # Independent Fraction-only audit of the theorem's final arithmetic.
    run(ROOT / "audit_best7_final_arithmetic.py", ROOT)
    print("\ncomplete seven-atom certificate: PASS")


if __name__ == "__main__":
    main()
