"""
Algorithms for Cooperative Games
"""

import tntorch as tn
import torch
import numpy as np
import scipy as sp
import scipy.stats


def _process_ps(N, ps):
    """
    Set up vector of semivalue probabilities.

    :param N: number of players in the game
    :param ps: either 'shapley', 'banzhaf-coleman', 'binomial', a function taking values 0...N-1, or an array of N values
    :return: a vector of N values
    """

    if ps == 'shapley':
        ps = [1./(N*sp.special.binom(N-1, n)) for n in range(N)]
    elif ps == 'banzhaf-coleman':
        ps = [1./(2**(N-1))]*N
    elif ps == 'binomial':
        ps = [p**n * (1-p)**(N-n-1) for n in range(N)]
    elif hasattr(ps, '__call__'):
        ps = [ps(n) for n in range(N)]
    if np.abs(np.sum([sp.special.binom(N-1, n)*ps[n] for n in range(N)]) - 1) > 1e-10:
        raise ValueError('The `ps` must be a probability')
    if not all([ps[n] >= 0 for n in range(N)]):
        raise ValueError('The `ps` must be regular')
    return ps


def semivalue_weights(N, ps, p=0.5):
    """
    Assemble a semivalue weighting tensor

    References:

    - "Semivalues and applications" (R. Lucchetti), http://www.gametheory.polimi.it/uploads/4/1/4/6/41466579/dauphine_june15_2015.pdf
    - "A Note on Values and Multilinear Equations" (A. Roth), http://web.stanford.edu/~alroth/papers/77_NRLQ_NoteValuesMultilinearExt.pdf

    :param N: an integer
    :param ps: 'shapley', 'banzhaf-coleman', 'binomial', a function, or an array of N values: for each n, probability that a player joins any coalition of size n that does not include him/her. It must satisfy that ps[n] > 0 for all n (regularity) and \sum_{n=0}^{N-1} \binom(N-1}{n} ps[n] = 1 (it is a probability)
    :param p: binomial probability; float in (0, 1). Used when `ps` is 'binomial', ignored otherwise. Default is 0.5
    :return: a TT tensor
    """

    ps = _process_ps(N, ps)

    ws = tn.weight_one_hot(N)
    ws.cores.append(torch.eye(N+1)[:, :, None])
    ws.Us.append(None)
    ws.cores[-1][np.arange(N), np.arange(N), 0] = torch.Tensor(ps)
    ws.cores[-2] = torch.einsum('ijk,km->ijm', ws.cores[-2], torch.sum(ws.cores[-1][:, :-1, :], dim=1))  # Absorb last core
    ws.cores = ws.cores[:-1]
    ws.Us = ws.Us[:-1]
    ws.cores = [torch.cat([core, core[:, 0:1, :]], dim=1) for core in ws.cores]
    return ws


def semivalues(game, **kwargs):
    """
    Compute all N semivalues for each of N players. Each semivalue 1, ..., N has cost O(N^3 R) + O(N^2 R^2), where R is the game's rank

    References:

    - "Semivalues and applications" (R. Lucchetti), http://www.gametheory.polimi.it/uploads/4/1/4/6/41466579/dauphine_june15_2015.pdf
    - "A Note on Values and Multilinear Equations" (A. Roth), http://web.stanford.edu/~alroth/papers/77_NRLQ_NoteValuesMultilinearExt.pdf

    :param game: a 2^N TT (`tntorch.Tensor`)
    :param **kwargs: `ps` and `p` (see `semivalue_weights`)
    :return: array with N semivalues
    """

    N = game.dim()
    ws = semivalue_weights(N, **kwargs)

    game = tn.Tensor([torch.cat([core[:, 0:2, :], core[:, 1:2, :] - core[:, 0:1, :]], dim=1) for core in game.cores])

    result = []
    for n in range(N):
        idx = [slice(0, 2)]*N
        idx[n] = slice(2, 3)
        result.append(np.asscalar(tn.dot(game[idx], ws[idx])))
    return torch.Tensor(result)


def sampling_semivalues(game, M, ps, p=0.5):
    """
    Monte Carlo approach: approximate semivalues using random sampling

    [1] Bachrach, Y., Markakis, E., Resnick, E., Procaccia, A. D., Rosenschein, J. S., & Saberi, A. (2010). "Polynomial calculation of the Shapley value based on sampling"

    :param game: a clas instance from games.py
    :param m: number of samples
    :param ps: 'shapley', 'banzhaf-coleman', 'binomial', a function, or an array of N values: for each n, probability that a player joins any coalition of size n that does not include him/her. It must satisfy that ps[n] > 0 for all n (regularity) and \sum_{n=0}^{N-1} \binom(N-1}{n} ps[n] = 1 (it is a probability)
    :param p: binomial probability; float in (0, 1). Used when `ps` is 'binomial', ignored otherwise. Default is 0.5
    :return:
    """

    N = game.N
    ps = _process_ps(N, ps)

    import time
    sh = np.zeros(N)
    for m in range(M):
        perm = np.random.permutation(N)
        idx = torch.zeros(N)
        vwithout = game.function(idx[None, :])
        for n in range(N):
            idx[perm[n]] = 1
            vwith = game.function(idx[None, :])
            sh[perm[n]] += vwith-vwithout
            vwithout = vwith
    return torch.Tensor(sh / m)
