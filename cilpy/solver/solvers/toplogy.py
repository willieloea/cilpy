# cilpy/solver/solvers/topology.py

from abc import ABC, abstractmethod
from typing import List


class Topology(ABC):
    """
    An abstract base class for defining PSO neighborhood topologies.
    """

    def __init__(self, swarm_size: int, **kwargs):
        self.swarm_size = swarm_size

    @abstractmethod
    def get_neighbors(self, particle_index: int) -> List[int]:
        """
        Returns the indices of the neighbors for a given particle.

        Args:
            particle_index (int): The index of the particle.

        Returns:
            List[int]: A list of neighbor indices, which must include the
                       particle itself.
        """
        pass


class GlobalTopology(Topology):
    """
    A global best (gbest) topology.

    In this structure, every particle is a neighbor of every other particle.
    The social influence comes from the single best particle in the entire swarm.
    """

    def get_neighbors(self, particle_index: int) -> List[int]:
        """All particles are neighbors, so return all indices."""
        return list(range(self.swarm_size))


class RingTopology(Topology):
    """
    A local best (lbest) ring topology.

    Each particle is connected to `k` neighbors on either side of it in a
    circular array. The common choice is `k=1`, where a particle is only
    influenced by its immediate left and right neighbors.
    """

    def __init__(self, swarm_size: int, k: int = 1, **kwargs):
        """
        Initializes the ring topology.

        Args:
            swarm_size (int): The total number of particles in the swarm.
            k (int): The number of neighbors to connect on each side.
        """
        super().__init__(swarm_size)
        if k < 0 or k >= swarm_size // 2:
            raise ValueError("k must be a non-negative integer smaller than half the swarm size.")
        self.k = k

    def get_neighbors(self, particle_index: int) -> List[int]:
        """
        Returns neighbors in a circular fashion.
        """
        neighbors = [particle_index]
        for i in range(1, self.k + 1):
            # Left neighbor (with wrap-around)
            left_neighbor = (particle_index - i) % self.swarm_size
            neighbors.append(left_neighbor)

            # Right neighbor (with wrap-around)
            right_neighbor = (particle_index + i) % self.swarm_size
            neighbors.append(right_neighbor)

        return neighbors
