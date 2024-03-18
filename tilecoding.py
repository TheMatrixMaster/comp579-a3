import numpy as np
from typing import List, Tuple


class TileCoder:
    def __init__(
        self,
        tiles_per_dim: List[int],
        value_limits: List[Tuple[float, float]],
        tilings: int,
        offset=lambda n: 2 * np.arange(n) + 1,
    ):
        """Initialize the tile coder.

        Args:
            tiles_per_dim (List[int]): the number of tiles to divide each dimension into
            value_limits (List[Tuple[min:float, max:float]]): the limits of the value space
            tilings (int): the number of tilings
            offset (function): returns a list of tiling offsets along each dimension
        """
        assert len(tiles_per_dim) == len(value_limits), "Tiles and limits mismatch"
        assert all([len(l) == 2 for l in value_limits]), "Invalid value limits"
        assert all([v > 0 for v in tiles_per_dim]), "Invalid number of tiles"

        # Gets the number of tiles per dimension
        tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=int) + 1

        # Calculates the tiling offsets along each dimension
        self._offsets = (
            offset(len(tiles_per_dim))
            * np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T
            / float(tilings)
            % 1
        )

        # Calculates the tile size (num_tiles / (max - min)) along each dimension
        self._limits = np.array(value_limits)
        self._norm_dims = np.array(tiles_per_dim) / (
            self._limits[:, 1] - self._limits[:, 0]
        )

        # Calculates the base indices for each tiling which is equal to the
        # product of the number of tiles per dimension and the tiling index
        self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)

        # Calculates the hash vector for each tiling
        self._hash_vec = np.array(
            [np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))]
        )

        # Number of tiles is equal to the number of tilings times the product
        # of the number of tiles per dimension
        self._n_tiles = tilings * np.prod(tiling_dims)

    def __getitem__(self, x: np.ndarray) -> np.ndarray:
        """Returns the binary feature vector for the given state.

        If the state is out of bounds, the tile indices are clipped to the
        nearest tile.
        """
        x = np.clip(x, self._limits[:, 0], self._limits[:, 1])
        off_coords = (
            (x - self._limits[:, 0]) * self._norm_dims + self._offsets
        ).astype(int)

        return self.to_binary(self._tile_base_ind + np.dot(off_coords, self._hash_vec))

    def to_binary(self, ind: np.ndarray) -> np.ndarray:
        """Converts the tile indices to a binary feature vector with shape (n_tiles,)"""
        binary = np.zeros(self._n_tiles)
        binary[ind] = 1
        return binary

    @property
    def n_tiles(self):
        """Returns the number of tiles"""
        return self._n_tiles
