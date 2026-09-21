#!/usr/bin/env python3
"""Exact algebraic checks for the exploratory radial-phase polynomial.

This script certifies nonvanishing on the positive radial line and nine
disjoint root disks.  It does not certify the Gaussian integrals, so it is
only one component of the proposed radial-phase certificate.
"""

from fractions import Fraction as F


RE = [F(x) for x in [
    "0.00849990487670248", "1.39131022205312",
    "-0.927269756577368", "0.302918484044903",
    "-0.0536926682205182", "0.00571628234471558",
    "-0.000361894662237122", "0.0000127199277639608",
    "-0.000000216391353176944", "0.00000000137312571007899",
]]
IM = [F(x) for x in [
    "0.962697213735343", "-1.76225740748728",
    "1.37926666401498", "-0.482963509971506",
    "0.0915497131921878", "-0.010154221025951",
    "0.000658355104692007", "-0.0000235505223692011",
    "0.000000409947784359127", "-0.00000000267339126178034",
]]

ROOT_CENTERS = [
    ("57.63562709324694", "0.26208643769462814"),
    ("46.59132086687371", "1.6046895884009444"),
    ("16.444253309399823", "-1.8146822911508949"),
    ("13.766148782800096", "3.0474007844274187"),
    ("7.74880817172378", "-5.788823575102351"),
    ("5.769130844332092", "5.622623409022937"),
    ("4.226624684892251", "-2.654070515701855"),
    ("1.7792275558303052", "1.877646984737552"),
    ("0.26904056334862503", "-0.4310010464786984"),
]


def trim(p):
    p = p[:]
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def multiply(left, right):
    answer = [F(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            answer[i + j] += a * b
    return trim(answer)


def derivative(p):
    return [k * p[k] for k in range(1, len(p))] or [F(0)]


def divide_remainder(left, right):
    left = trim(left)
    right = trim(right)
    while len(left) >= len(right) and left != [0]:
        degree = len(left) - len(right)
        ratio = left[-1] / right[-1]
        for j, value in enumerate(right):
            left[degree + j] -= ratio * value
        left = trim(left)
    return left


def sign(value):
    return (value > 0) - (value < 0)


def variations(values):
    values = [value for value in values if value]
    return sum(a != b for a, b in zip(values, values[1:]))


def complex_add(left, right):
    return left[0] + right[0], left[1] + right[1]


def complex_multiply(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def complex_evaluate(real, imag, point):
    answer = (real[-1], imag[-1])
    for a, b in zip(real[-2::-1], imag[-2::-1]):
        answer = complex_add(complex_multiply(answer, point), (a, b))
    return answer


def upper_norm(value):
    return abs(value[0]) + abs(value[1])


def lower_norm(value):
    return max(abs(value[0]), abs(value[1]))


def sturm_nonvanishing():
    modulus_square = multiply(RE, RE)
    imaginary_square = multiply(IM, IM)
    modulus_square += [F(0)] * (len(imaginary_square) - len(modulus_square))
    for k, value in enumerate(imaginary_square):
        modulus_square[k] += value
    modulus_square[0] -= F(9, 25)

    sequence = [trim(modulus_square), trim(derivative(modulus_square))]
    while len(sequence[-1]) > 1:
        remainder = divide_remainder(sequence[-2], sequence[-1])
        sequence.append([-value for value in remainder])

    at_zero = variations([sign(p[0]) for p in sequence])
    at_infinity = variations([sign(p[-1]) for p in sequence])
    assert at_zero == at_infinity
    assert modulus_square[0] > 0 and modulus_square[-1] > 0
    return at_zero - at_infinity


def rouche_disks():
    radius = F(1, 10**6)
    centers = [(F(a), F(b)) for a, b in ROOT_CENTERS]
    derivative_pairs = [(RE, IM)]
    for _ in range(9):
        real, imag = derivative_pairs[-1]
        derivative_pairs.append((derivative(real), derivative(imag)))

    for center in centers:
        values = [complex_evaluate(real, imag, center)
                  for real, imag in derivative_pairs]
        left = lower_norm(values[1]) * radius
        right = upper_norm(values[0])
        factorial = 1
        for k in range(2, 10):
            factorial *= k
            right += upper_norm(values[k]) * radius**k / factorial
        assert left > right

    # Coordinate separation suffices to prove the closed disks disjoint.
    for i, left in enumerate(centers):
        for right in centers[i + 1:]:
            assert max(abs(left[0] - right[0]), abs(left[1] - right[1])) > 2 * radius

    # Nine disjoint disks, one zero in each, exhaust the degree-nine roots.
    assert len(centers) == len(RE) - 1

    imaginary_distances = [abs(center[1]) - radius for center in centers]
    assert all(distance > 0 for distance in imaginary_distances)
    c1 = sum(1 / distance for distance in imaginary_distances)
    c2 = sum(1 / distance**2 for distance in imaginary_distances)
    assert c1 < 9
    assert c2 < 22
    return c1, c2


def main():
    assert sturm_nonvanishing() == 0
    c1, c2 = rouche_disks()
    print("radial polynomial exact algebra: PASS")
    print("|P(s)| > 3/5 for every s >= 0")
    print("nine disjoint root disks of radius 10^-6 certified")
    print("sum inverse imaginary distances <", float(c1), "< 9")
    print("sum squared inverse distances   <", float(c2), "< 22")
    print("Gaussian integral checks are not part of this script")


if __name__ == "__main__":
    main()
