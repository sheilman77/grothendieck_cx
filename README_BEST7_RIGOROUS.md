# Seven-atom rigorous certificate

This release accompanies `complex_grothendieck_upper_bound_best7.tex` and
certifies

\[
K_G^{\mathbb C}<1.404898554745441,
\qquad
\Gamma_0-\gamma_H>5.3593234075\times 10^{-6}.
\]

The candidate-search files are not proof inputs. The five scalar
preprocessors and six signed weights are exact rational constants in
`audit_sparse6_altgroups.py`. The two origin coefficient tables are pinned by
SHA-256 and store outward-rounded binary64 endpoints of Arb enclosures.

## Requirements

- Python 3.12
- NumPy 1.26 or newer
- python-flint 0.9.0 or newer

## Audit

From this directory run:

```text
python audit_best7_full.py
```

This runs, in order:

1. exact nonvanishing and root-disk checks for the origin polynomial;
2. directed total- and visible-energy checks for the origin tail;
3. the directed prefix, scalar-tail, origin-tail, normalization, contact,
   gain, and reciprocal checks;
4. a `fractions.Fraction`-only independent check of the final theorem
   arithmetic.

The expected last line is:

```text
complete seven-atom certificate: PASS
```

The table generators are included for a deeper audit. They write to the path
given with `--output`; use a temporary destination so that the pinned files are
not overwritten.
