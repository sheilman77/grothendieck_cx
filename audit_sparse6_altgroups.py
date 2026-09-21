#!/usr/bin/env python3
"""Directed audit of the explicit six-branch, seven-atom candidate.

The five scalar preprocessors are specified here directly as exact
terminating-decimal rationals.  No search row, search weight, or saved
147-column candidate is an input to this audit.  Four low-degree rows use
certified Cauchy tails, while the high-degree row is composed exactly through
mode 5000 and charged by a Wallis reserve thereafter.
"""

from fractions import Fraction as Q
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CERT = HERE / "support_compression_certificate"
sys.path.insert(0, str(CERT))

from flint import arb, arb_poly, ctx  # noqa: E402

import complex_upper_convex_2e6_scalar_tail_certificate as scalar_check  # noqa: E402
import complex_upper_saha_cubic_exacttail_457e8_certificate as base  # noqa: E402
import complex_upper_saha_visible_energy_refinement as origin_check  # noqa: E402
from complex_upper_convex_2e6_low_certificate import scalar_profile  # noqa: E402
import tight_origin_extension as tight  # noqa: E402


ctx.dps = 80
COUNT = 120
TERMINAL = 5000
ORIGIN_TAIL = Q(523, 2_000_000)


def exact_row(entries):
    """Build one 120-mode row from exact decimal strings."""
    row = [Q(0)] * COUNT
    for mode, coefficient in entries:
        assert 0 <= mode < COUNT and row[mode] == 0
        row[mode] = Q(coefficient)
    row = tuple(row)
    assert sum(map(abs, row)) <= 1
    return row


# Every displayed string below is an exact terminating decimal rational.
# The five scalar preprocessors are numbered by decreasing absolute raw
# weight.  The structurally different origin kernel is branch 6.
RHO_1 = exact_row((
    (0, "0.906231648312"),
    (1, "-0.093651693922"),
    (4, "-0.000116657762"),
))
RHO_2 = exact_row((
    (0, "0.910377315558"),
    (1, "-0.063651630829"),
    (3, "-0.025971053610"),
))
RHO_3 = exact_row((
    (0, "-0.912512978184"),
    (1, "0.035570362029"),
    (3, "0.050646289677"),
    (4, "0.001270370107"),
))
RHO_4 = exact_row((
    (0, "0.901913867354"),
    (1, "-0.037111896010"),
    (3, "-0.026391249065"),
    (5, "-0.033420629518"),
    (7, "-0.001162358049"),
))
RHO_5 = exact_row((
    (0, "0.094626045381"),
    (1, "-0.856254413568"),
    (10, "0.001274576105"),
    (13, "0.008312996734"),
    (14, "0.001885453067"),
    (15, "0.001662844635"),
    (16, "0.009295485863"),
    (17, "0.009399425108"),
    (18, "0.008540450681"),
    (19, "0.000847985940"),
    (20, "0.003771690693"),
    (21, "0.000873880329"),
    (27, "-0.000419180236"),
    (29, "-0.000311954497"),
    (30, "-0.000455650279"),
    (31, "-0.000996284704"),
    (35, "-0.000865568525"),
    (37, "0.000206113648"),
))


# Scalar branches 1--5, followed by the origin branch 6.
RHO_ROWS = (RHO_1, RHO_2, RHO_3, RHO_4, RHO_5)
WEIGHTS = tuple(map(Q, (
    "0.983543086263686583",
    "0.0124174018143118538",
    "-0.0030311909556774751",
    "0.000593632758261406976",
    "0.0000169358532612748746",
    "0.000397752354801503847",
)))
ORIGIN_INDEX = 5
ORIGIN_WEIGHT = WEIGHTS[ORIGIN_INDEX]


def add_complete(profile, weight, row):
    """Add outer blocks complete through TERMINAL and charge the rest."""
    support = tuple(mode for mode, value in enumerate(row) if value)
    minimum, maximum = support[0], support[-1]
    first = (TERMINAL - maximum) // (2 * maximum + 1) + 1
    assert (first - 1) + (2 * (first - 1) + 1) * maximum <= TERMINAL
    assert first + (2 * first + 1) * maximum > TERMINAL

    polynomial = arb_poly([base.aq(value) for value in row[:maximum + 1]])
    square = polynomial * polynomial
    power = polynomial
    h_over_pi = base.aq(Q(1, 4))
    factor = base.aq(weight)
    for outer in range(first):
        profile = (
            profile + power.left_shift(outer) * (factor * h_over_pi)
        ).truncate(TERMINAL + 1)
        power = (power * square).truncate(TERMINAL + 1)
        h_over_pi *= arb((2 * outer + 1) ** 2) / (
            4 * (outer + 1) * (outer + 2)
        )

    # Wallis: all outer blocks j >= first have total l1 norm at most
    # |weight|/(4*first), because ||row||_A <= 1.
    omitted = abs(weight) * Q(1, 4 * first)
    first_minimum = first + (2 * first + 1) * minimum
    return profile, omitted, (
        first, support, first_minimum, first_minimum < COUNT,
    )


def sparse_tail(row):
    """Complete-block plus Wallis tail for the high-degree scalar row."""
    profile, remainder, description = add_complete(
        arb_poly(), WEIGHTS[4], row
    )
    finite_over_pi = sum(
        (abs(profile[n]).upper() for n in range(COUNT, TERMINAL + 1)),
        arb(0),
    )
    finite = (arb.pi() * finite_over_pi).upper()
    return finite, remainder, finite + base.aq(remainder), description


def directed_prefix():
    """Directed q0 lower bound and nonlinear-prefix upper bound."""
    scalar = [Q(0)] * COUNT
    for weight, row in zip(WEIGHTS[:5], RHO_ROWS):
        profile = scalar_profile(row)
        for n in range(COUNT):
            scalar[n] += weight * profile[n]
    return tight.directed_origin_mixture(scalar, ORIGIN_WEIGHT)


def tight_contact_upper():
    """Enclose Haagerup's contact slope from above."""
    contact_lo = base.aq(Q(812_557_858_821_472, 10**15))
    contact_hi = base.aq(Q(812_557_858_821_473, 10**15))
    half = base.aq(Q(1, 2))

    def equation(x):
        return 2 * x * (x * x).hypgeom_2f1(half, half, arb(2)) - 1 - x

    assert equation(contact_lo) < 0
    assert equation(contact_hi) > 0
    return (arb.pi() * (1 + contact_hi) / 8).upper()


def audit():
    assert base.sha256(base.TABLE) == base.EXPECTED_TABLE_SHA256
    assert tight.sha256(tight.TABLE) == tight.EXPECTED_SHA256
    assert ORIGIN_TAIL == origin_check.ORIGIN_TAIL
    assert sum(map(abs, RHO_1)) == Q(999_999_999_996, 10**12)
    assert sum(map(abs, RHO_2)) == Q(999_999_999_997, 10**12)
    assert sum(map(abs, RHO_3)) == Q(999_999_999_997, 10**12)
    assert sum(map(abs, RHO_4)) == Q(999_999_999_996, 10**12)
    assert sum(map(abs, RHO_5)) == Q(999_999_999_993, 10**12)
    assert all(abs(WEIGHTS[r]) > abs(WEIGHTS[r + 1]) for r in range(4))

    cauchy_data, cauchy_tails = [], {}
    for name, radius, row in (
        ("rho_1", Q("1.2"), RHO_1),
        ("rho_2", Q("1.11"), RHO_2),
        ("rho_3", Q("1.11"), RHO_3),
        ("rho_4", Q("1.11"), RHO_4),
    ):
        scalar_check.root_check(row, radius)
        weighted_norm, distance = scalar_check.boundary_check(row, radius)
        tail = Q(10) * radius ** (-(2 * COUNT + 1)) / (
            1 - radius ** (-2)
        )
        cauchy_tails[name] = tail
        cauchy_data.append((
            name, radius, tuple(n for n, value in enumerate(row) if value),
            sum(map(abs, row)), weighted_norm, distance, tail,
        ))

    finite, remainder, high_tail, description = sparse_tail(RHO_5)
    low_tail = sum(
        abs(WEIGHTS[r]) * cauchy_tails[f"rho_{r + 1}"]
        for r in range(4)
    )
    fixed = low_tail + abs(ORIGIN_WEIGHT) * ORIGIN_TAIL
    q0, prefix = directed_prefix()
    nonlinear = prefix + high_tail + base.aq(fixed)
    mass = sum(map(abs, WEIGHTS))
    normalized = tuple(weight / mass for weight in WEIGHTS)
    gamma = (q0 - nonlinear) / base.aq(mass)
    gamma_h = tight_contact_upper()
    k_upper = (arb(1) / gamma).upper()

    top_level_support = len(WEIGHTS)
    flat_selector_atoms = top_level_support + 1
    assert top_level_support == 6 and all(WEIGHTS)
    assert flat_selector_atoms == 7 and ORIGIN_WEIGHT > 0
    assert sum(map(abs, normalized)) == 1
    assert nonlinear < q0
    assert mass == Q(1_250_000_000_000_000_121_997,
                     1_250_000_000_000_000_000_000)
    assert q0 > base.aq(Q("0.71179640805640433452"))
    assert prefix < base.aq(Q("0.000001074829404498064"))
    assert finite < base.aq(Q("0.000000000002699176355797438"))
    assert high_tail < base.aq(Q("0.00000006319618149454563"))
    assert low_tail < Q("0.00000000001018")
    assert fixed < Q("0.0000001040224125770398")
    assert nonlinear < base.aq(Q("0.000001242047998569650"))
    assert gamma > base.aq(Q("0.71179516600840569540"))
    assert gamma - gamma_h > base.aq(Q("0.000005359323407547292"))
    assert k_upper < base.aq(Q("1.404898554745441"))

    print("direct six-branch / seven-atom certificate: PASS")
    print("origin table SHA256 =", base.sha256(base.TABLE))
    print("origin extension SHA256 =", tight.sha256(tight.TABLE))
    print("top-level / flat selector support =",
          top_level_support, flat_selector_atoms)
    for data in cauchy_data:
        print("preprocessor / R / support / l1 / weighted norm / distance / tail =", data)
    print("branch mass =", mass)
    print("q0 >", q0)
    print("prefix <", prefix)
    print("high-degree finite / outer / total <",
          finite, float(remainder), high_tail)
    print("low-degree tail =", float(low_tail))
    print("fixed tail =", float(fixed))
    print("nonlinear <", nonlinear)
    print("Gamma >", gamma)
    print("gamma_H <", gamma_h)
    print("gain over gamma_H >", gamma - gamma_h)
    print("K <", k_upper)
    print("high-degree description =", description)


if __name__ == "__main__":
    audit()
