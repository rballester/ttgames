# ttgames: Tensor Approximation of Cooperative Games and Their Semivalues

**Cooperative game** studies how to split a payoff fairly between a set $N$ of players that **collaborate towards a common goal**, for example a reward after completing a task, a dividend, etc.; see e.g. the [*airport problem*](https://en.wikipedia.org/wiki/Airport_problem). A collaborative game is determined by its **utility function**, a map $f: \mathrm{powerset}(\{1, \dots, N\}) \to \mathbb{R}$ that assigns a payoff to each possible coalition of players.

We propose to handle the game's utility function as a hypercube of size $2^{|N|}$ that contains all possible payoffs, in other words, a **tensor**. Using this idea, we:

- Build a *compressed* representation of this tensor via the [**tensor train format (TT)**](https://epubs.siam.org/doi/abs/10.1137/090752286). We learn it by means of [**cross-approximation**](https://www.sciencedirect.com/science/article/pii/S0024379509003747), an adaptive sampling algorithm that produces a TT by querying only a subset of the tensor's entries.
- Give an algorithm to compute arbitrary **semivalues** (which include the [**Shapley values**](https://en.wikipedia.org/wiki/Shapley_value) and the [**Banzhaf power indices**](https://en.wikipedia.org/wiki/Banzhaf_power_index)) out of the game. We do so by computing the dot product between our learned TT tensor and a certain *weighting tensor* that we handcraft so as to encode the desired semivalues. The weighting tensor is inspired by **deterministic finite automata (DFA)**; see papers [1](https://www.sciencedirect.com/science/article/abs/pii/S0951832018303132) and [2](https://epubs.siam.org/doi/10.1137/17M1160252).

This is a Python package that implements these algorithms, as well as the utility functions for several classic cooperative games.

## Example Use

The package's most important function is `ttgames.semivalues.semivalues(...)`. See [this Jupyter notebook](https://github.com/rballester/ttgames/blob/main/examples/airport.ipynb) for a tutorial using the *airport problem*.

## Installation

You may install *ttgames* from the source as follows:

```
git clone https://github.com/rballester/ttgames.git
cd ttgames
pip install .
```

### Cite As

This package is an implementation of the paper ["Tensor Approximation of Cooperative Games and Their Semivalues"](https://www.sciencedirect.com/science/article/abs/pii/S0888613X21001912) (R. Ballester-Ripoll 2021), which you should cite as:

```
@article{Ballester-Ripoll:21,
    title = {Tensor Approximation of Cooperative Games and Their Semivalues},
    journal = {International Journal of Approximate Reasoning},
    volume = {142},
    pages = {94-108},
    year = {2022},
    author = {Rafael Ballester-Ripoll},
    keywords = {Cooperative game theory, Semivalues, Allocation rules, Computational game theory, Tensor approximation, Tensor train decomposition}
}
```

## Contact

For comments or questions, please open an [issue](https://github.com/rballester/ttgames/issues) or write to rafael.ballester@ie.edu. Pull requests are also welcome!
