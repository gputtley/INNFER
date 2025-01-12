import numpy as np
from scipy.stats import wasserstein_distance


class MultidimMetrics():

  def __init__(self, network, parameters, sim_files):
    self.network = network
    self.parameters = parameters
    self.sim_files = sim_files

    self.dataset_exist = False
    self.sim_dataset = None
    self.synth_dataset = None

  def MakeDatasets(self):
    print()

'''
def sliced_wasserstein_distance(data1, data2, num_projections=100):
    """
    Calculate the Sliced Wasserstein Distance between two multidimensional datasets.

    Parameters:
    data1 (np.ndarray): First dataset, shape (n_samples1, n_features)
    data2 (np.ndarray): Second dataset, shape (n_samples2, n_features)
    num_projections (int): Number of random projections to approximate the SWD

    Returns:
    float: The estimated Sliced Wasserstein Distance.
    """
    if data1.shape[1] != data2.shape[1]:
        raise ValueError("The two datasets must have the same number of features.")

    # Number of features (dimensionality)
    n_features = data1.shape[1]

    # Generate random projection vectors (normalized)
    random_directions = np.random.normal(size=(num_projections, n_features))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)

    swd = 0.0

    for direction in random_directions:
        # Project data onto the random direction
        proj1 = np.dot(data1, direction)
        proj2 = np.dot(data2, direction)

        # Compute the Wasserstein distance in 1D
        swd += wasserstein_distance(proj1, proj2)

    # Average over the number of projections
    swd /= num_projections

    return swd
'''