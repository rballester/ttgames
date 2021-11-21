"""
Collection of utility functions for cooperative games
"""

import tntorch as tn
import torch
import numpy as np


class Game:

    def __init__(self):
        pass

    def sample(self, P, seed=0):
        if seed is None:
            seed = torch.randint(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max, [1]).item()
            print('Random seed:', seed)
        rng = torch.Generator()
        rng.manual_seed(seed)
        return torch.randint(0, 2, [P, self.N], generator=rng)

    def domain(self):
        return [torch.arange(2) for n in range(self.N)]

    def full(self):
        """
        Assembles the uncompressed value function for every possible coalition.

        :return: a 2^N TT tensor
        """

        if self.N > 25:
            raise ValueError('Will not assemble full tensor: it has more than {} elements'.format(2**25))
        idx = torch.Tensor(np.unravel_index(range(2**self.N), [2]*self.N)).t()
        return self.function(idx).reshape([2]*self.N)


class Shoes(Game):
    """
    Sometimes also known as "gloves". K players have a left shoe, K+1 have a right shoe. Obviously, shoes can only be sold by pairs.
    Source: http://nb.vse.cz/~zouharj/games/Lecture_7.pdf
    """

    def __init__(self, seed=0, N=15, **kwargs):

        def function(X):
            return torch.min(torch.sum(X[:, :self.L], dim=1), torch.sum(X[:, self.L:], dim=1))

        self.L = N//2-1  # Players with left shoes. The rest have right shoes

        self.function = function
        self.N = N
        super().__init__(**kwargs)


class WeightedMajority(Game):
    """
    Classical simple game where each party has a number of seats, and a coalition wins if and only if their seat number reach a threshold
    """

    def __init__(self, seed=0, N=10, seats=None, max_party_seats=10, threshold=None, **kwargs):

        def function(X):
            return (torch.sum(X*self.seats[None, :], dim=1) >= self.threshold).to(torch.get_default_dtype())

        if seats is None:
            assert N is not None
            rng = torch.Generator()
            rng.manual_seed(seed)
            self.seats = torch.sort(torch.randint(1, max_party_seats+1, [N], generator=rng)).values[range(N-1, -1, -1)]
        else:
            N = len(seats)
            self.seats = torch.sort(torch.Tensor(seats)).values[range(N-1, -1, -1)]
        if threshold is None:
            threshold = torch.sum(self.seats).item() // 2 + 1
        self.threshold = threshold
        assert len(self.seats) == N

        self.N = N
        self.function = function
        super().__init__(**kwargs)


class Bankruptcy(Game):
    """
    N creditors claim their debts over an estate whose value is less than the sum of all debts.
    More details: http://www.lamsade.dauphine.fr/~airiau/Teaching/CoopGames/2012/core.pdf (2.2.1)
    """

    def __init__(self, seed=0, N=15, **kwargs):

        def function(X):
            X = X.to(torch.get_default_dtype())
            return torch.max(torch.zeros(len(X)), self.E - torch.sum((1 - X) * self.c[None, :], dim=1))

        rng = torch.Generator()
        rng.manual_seed(seed)
        # c = torch.sort(torch.randint(1, 11, [N], generator=rng)).values
        self.c = torch.randint(1, 51, [N], generator=rng)
        self.E = torch.sum(self.c) // 2

        self.function = function
        self.N = N
        super().__init__(**kwargs)


class Airport(Game):
    """
    Classical airport cost game (see e.g. https://en.wikipedia.org/wiki/Airport_problem): N airlines need runways of different length. The cost of a coalition is the longest of their lengths
    """

    def __init__(self, seed=0, N=15, **kwargs):

        def function(X):
            return torch.max(X * self.ak[None, :], dim=1).values

        rng = torch.Generator()
        rng.manual_seed(seed)
        # ak = torch.sort(torch.rand(N, generator=rng)).values
        self.ak = torch.rand(N, generator=rng)

        shapley = torch.cat([torch.Tensor([0]), self.ak])
        shapley = shapley[1:] - shapley[:-1]
        shapley / torch.arange(N, 0, -1)
        self.shapley = torch.cumsum(shapley/torch.arange(N, 0, -1), dim=0)

        self.function = function
        self.N = N
        super().__init__(**kwargs)


class Maschler(Game):
    """
    Game considered in Aumann & Myerson (1988), "Endogenous Formation of Links Between Players and of Coalitions: An Application of the Shapley Value"
    """

    def __init__(self, seed=0, **kwargs):

        def function(X):
            return torch.Tensor([0, 0, 60, 72])[torch.sum(X, dim=1).long()]

        self.N = 3
        self.function = function
        super().__init__(**kwargs)


class Miners(Game):
    """
    N miners find a large supply of gold bars; each bar needs two miners to be carried. So the value of a coalition of size S is S/2 (if S is even), and (S-1)/2 (if S is odd)
    Source: http://nb.vse.cz/~zouharj/games/Lecture_7.pdf
    """

    def __init__(self, seed=0, N=11, **kwargs):

        def function(X):  # TODO tidy up
            result = []
            X = X.numpy()
            for x in X:
                S = np.sum(x)
                if np.mod(S, 2) == 0:
                    result.append(S / 2)
                else:
                    result.append((S - 1) / 2)
            return torch.Tensor(result)

        self.N = N
        self.function = function
        super().__init__(**kwargs)


class OneSeller(Game):
    """
    Example 15 from https://www.math.ucla.edu/~tom/Game_Theory/coal.pdf
    Player 0 owns an object of no intrinsic worth to himself. Buyer j values the object at aj dollars, with a1 > ... > am > 0
    """

    def __init__(self, seed=0, N=15, **kwargs):

        def function(X):
            # X = X.numpy()
            result = torch.max(X[:, 1:]*self.ak[None, :], dim=1).values
            result[X[:, 0].long() == 0] = 0
            return result

        M = N - 1
        rng = torch.Generator()
        rng.manual_seed(seed)
        # ak = torch.sort(torch.rand(M, generator=rng)).values[range(M-1, -1, -1)]
        self.ak = torch.rand(M, generator=rng)

        self.N = N
        self.function = function
        super().__init__(**kwargs)


class ManySellers(Game):
    """
    Example 13 from https://www.math.ucla.edu/~tom/Game_Theory/coal.pdf (non-balanced version)
    """

    def __init__(self, seed=0, N=None, B=7, C=7, price_cap=10, **kwargs):

        def function(X):
            return torch.min(torch.sum(X[:, :B] * self.Bk[None, :], dim=1),
                              torch.sum(X[:, B:] * self.Ck[None, :], dim=1))

        rng = torch.Generator()
        rng.manual_seed(seed)
        if N is not None:
            B = N // 2
            C = N - B
        self.Bk = torch.sort(torch.randint(1, price_cap+1, [B], generator=rng)).values
        self.Ck = torch.sort(torch.randint(1, price_cap+1, [C], generator=rng)).values

        self.N = B + C
        self.function = function
        super().__init__(**kwargs)


class MinimumSpanningTree(Game):
    """
    Taken from https://www.cs.ubc.ca/~kevinlb/teaching/cs532l%20-%202007-8/lectures/lect23.pdf
    """

    import scipy.stats
    from scipy.sparse.csgraph import minimum_spanning_tree

    def __init__(self, seed=0, N=10, cost_cap=10, **kwargs):

        def function(X):
            result = []
            for x in X:
                coal = torch.where(x)[0]
                noagreement = torch.sum(costs[coal, -1])
                coal = torch.cat([coal, torch.Tensor([N]).long()])
                Tcsr = minimum_spanning_tree(costs[coal, :][:, coal].numpy())
                result.append(noagreement - Tcsr.sum())
            return torch.Tensor(result)

        rng = torch.Generator()
        rng.manual_seed(seed)
        costs = torch.randint(1, cost_cap+1, [N + 1, N + 1], generator=rng)
        i, j = torch.meshgrid(torch.arange(N + 1), torch.arange(N + 1))
        costs[np.unravel_index(np.where((i <= j).flatten().numpy()), [N + 1, N + 1])] = 0
        costs = costs + costs.t()
        print(costs)

        self.N = N
        self.function = function
        super().__init__(**kwargs)
