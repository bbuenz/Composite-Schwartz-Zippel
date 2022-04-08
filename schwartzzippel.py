import math
from itertools import chain

from gmpy2 import next_prime
from heapq import heappush, heappop
from math import log2, e
from scipy.stats import nbinom


def value(p, r):
    return r * log2(p)


def weight(n: int, p: int, r: int):
    if r <= 0:
        return 0
    # weightv = -log2(betainc(r, n, 1.0 / float(p)))
    # Negative Binomial survival function
    weightv = -nbinom.logsf(r - 1, n, 1 - 1 / float(p)) * log2(e)
    return weightv


def marginalweight(n, p, r):
    weightr = weight(n, p, r)
    weightrminone = weight(n, p, r - 1)
    return weightr - weightrminone


def marginaldensity(n, p, r):
    if r <= 0:
        return 2 ** 300

    numerator = log2(p)
    denomintor = marginalweight(n, p, r)
    return numerator / denomintor


def density(n, p, r):
    if r <= 0:
        return 2 ** 300
    return r * log2(p) / weight(n, p, r)


def computationalbound(n, bound, densfunc=marginaldensity, weightfunc=marginalweight):
    densityheap = []
    densityheap.append((-densfunc(n, 2, 1), 2, 1))
    currentweight = 0
    objective = 0
    nextprime = 2
    while currentweight < bound:
        (topdens, p, r) = heappop(densityheap)
        heappush(densityheap, (-densfunc(n, p, r + 1), p, r + 1))
        objective += log2(p)
        currentweight += weightfunc(n, p, r)
        if p == nextprime:
            nextprime = next_prime(nextprime)
            heappush(densityheap, (-densfunc(n, nextprime, 1), nextprime, 1))

    includedprimes = map(lambda obj: (obj[1], obj[2] - 1), densityheap)

    return objective, sorted(includedprimes)


for n in chain(range(2, 31), [50]):
    objlist = []
    for lam in [40, 100, 120, 240]:
        obj, primes = computationalbound(n, lam)
        objlist.append(obj)
    print(str(n) + " & " + ' & '.join(map(lambda obj: str(math.ceil(obj)), objlist)) + "\\\\")

for n in chain(range(2, 21), [50]):
    lam = 120
    obj, primes = computationalbound(n, lam)
    print(str(n) + " & " + "\\begin{dmath*}" + " \cdot ".join(
        map(lambda pr: str(pr[0]) if pr[1] == 1 else str(pr[0]) + "^{" + str(pr[1]) + "}",
            primes[:-1])) + "\\end{dmath*} \\\\")
