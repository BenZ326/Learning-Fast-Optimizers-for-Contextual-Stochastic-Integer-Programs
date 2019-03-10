import numpy as np


# The codes are mainly based on
# https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding
"""
The arguments:
low: the lower bound of the state space;
high: the upper bound of the state space;
bins: a multi-dimensional array. Its length is how many parameters used to describe a state (e.g. the dimension of "low" or "high")
while, for each axis (apart from 0th), a dimension goes for how many blocks you want to split the continuous space;
offsets: a multi-dim array. Its length respects to the length of bins and basically it's the offset for each dimension
"""
def create_tiling_grid(low, high, bins, offsets):
    assert(len(bins) == len(low) and len(high)==len(low) and len(bins) == len(offsets))
    grid = [np.linspace(low[dim], high[dim], bins[dim]+1 )[1:-1]+ offsets[dim] for dim in range(len(bins))]
    return grid

def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    # TODO: Implement this
    return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))  # apply along each dimension


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    encoded_sample = [discretize(sample, grid) for grid in tilings]
    return np.concatenate(encoded_sample) if flatten else encoded_sample
