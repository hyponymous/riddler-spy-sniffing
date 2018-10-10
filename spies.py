#!/usr/bin/env python

import argparse
from babel import numbers
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=bool, default=False,
                   help='an integer for the accumulator')

args = parser.parse_args()

def make_ria(N=1024, K=17):
    """
    >>> len(make_ria())
    1024
    >>> all([sum(make_ria()) == 17 for _ in [None] * 100])
    True
    """
    z = np.array([False] * 1024)
    z[np.random.choice(N, size=[K,], replace=False)] = True
    return z

class Ria():
    """
    >>> len(Ria().spy_indices)
    17
    >>> Ria().go_on_retreat(range(1024))[0]
    True
    >>> Ria().go_on_retreat(range(0))[0]
    False
    >>> Ria().K
    17
    >>> Ria().N
    1024
    """

    def __init__(self, N=1024, K=17, agents=None, retreats=0, attendee_count=0):
        self.K = K
        self.N = N
        self._agents = agents if agents is not None else make_ria(N, K)
        self._spy_indices = frozenset(np.flatnonzero(self._agents))
        self.retreats = retreats
        self.attendee_count = attendee_count

    @property
    def spy_indices(self): return self._spy_indices
    @property
    def cost(self): return numbers.format_currency(1000 * self.attendee_count, 'USD')

    def go_on_retreat(self, indices):
        """
        return True if a spy meeting took place (that is, if all spies are in
        the indices queried)
        """
        return (len(self._spy_indices - set(indices)) == 0,
            Ria(self.N, self.K, self._agents, self.retreats + 1, self.attendee_count + len(indices)))

def naive(ria):
    known_spies = set()
    suspects = set(range(ria.N))
    exonerated = set()
    while len(known_spies) < ria.K:
        assert(len(suspects & known_spies) == 0)
        assert(len(suspects & exonerated) == 0)
        assert(len(known_spies & exonerated) == 0)

        # test one index at a time
        candidate = np.random.choice(list(suspects))

        meeting, ria = ria.go_on_retreat((suspects | known_spies) - set([candidate]))
        suspects.remove(candidate)
        if meeting:
            exonerated.add(candidate)
        else:
            known_spies.add(candidate)

    return (known_spies, ria)

def pool_size_nk(N, K, P=0.5):
    p = 1.
    for n in range(N):
        p *= (1. - K / (N - n))
        if p < P:
            break
    return n

def pool_size_weighted(N, K, P=0.5):
    if N - K <= 1: return 1
    p = 1.
    expected_values = np.ones(N - K - 1)
    for n in range(N - K - 1):
        p *= (1. - K / (N - n))
        expected_values[n] = n*p
    return np.argmax(expected_values)

def uniform(ria):
    """
    Pick off larger chunks of candidates at a time, assuming that candidates
    that haven't been eliminated have a uniform probability of being a spy,
    ignoring costs
    """
    suspects = set(range(ria.N))
    exonerated = set()
    while len(suspects) > ria.K:
        assert(len(suspects & exonerated) == 0)

        # test some number of agents, ideally to cut the hypothesis space in
        # half
        pool_size = max(1, pool_size_weighted(len(suspects), ria.K))
        candidates = set(np.random.choice(list(suspects), size=(pool_size)))

        meeting, ria = ria.go_on_retreat(suspects - candidates)
        if meeting:
            suspects -= candidates
            exonerated |= candidates

    known_spies = suspects
    return (known_spies, ria)


def main():
    ria = Ria()
    naive_deduced, naive_ria = naive(ria)
    uniform_deduced, uniform_ria = uniform(ria)

    print('naive')
    print('  retreats: {}'.format(naive_ria.retreats))
    print('  cost: {}'.format(naive_ria.cost))
    print('  spies not found: {}'.format(set(ria.spy_indices - naive_deduced)))
    print('  falsely accused: {}'.format(naive_deduced - ria.spy_indices))

    print('uniform')
    print('  retreats: {}'.format(uniform_ria.retreats))
    print('  cost: {}'.format(uniform_ria.cost))
    print('  spies not found: {}'.format(set(ria.spy_indices - uniform_deduced)))
    print('  falsely accused: {}'.format(uniform_deduced - ria.spy_indices))

if __name__ == "__main__":
    if args.test:
        import doctest
        doctest.testmod()
    else:
        main()
