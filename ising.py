import numpy as np


def energy(x):
    """
    x: 2D numpy array of spins

    Returns the energy of the Ising model configuration.
    E(x) = -J * sum(s_i * s_j) - h * sum(s_i)
    where J is the interaction strength, h is the external magnetic field,
    and the sums are over nearest neighbors.
    Here, we assume J = 1 and h = 0 for simplicity.
    """
    J = 1  # Interaction strength
    h = 0  # External magnetic field
    n, m = x.shape
    E = 0

    # Sum over nearest neighbors
    for i in range(n):
        for j in range(m):
            E -= J * x[i, j] * (x[(i + 1) % n, j] + x[i, (j + 1) % m])
            E -= h * x[i, j]

    return E
