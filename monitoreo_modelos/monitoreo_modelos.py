import numpy as np
from scipy.stats import wasserstein_distance

def jensen_shannon_distance(p, q):
    m = 0.5 * (p + q)
    js_distance = 0.5 * (wasserstein_distance(p, m) + wasserstein_distance(q,m))
    return js_distance


data_distribution_1 = np.array([0.2, 0.3, 0.1, 0.4])
data_distribution_2 = np.array([0.1, 0.4, 0.2, 0.3])


wasserstein_dist = wasserstein_distance(data_distribution_1,data_distribution_2)
print(f"Distancia de Wasserstein: {wasserstein_dist}")
      


jensen_shannon_dist = jensen_shannon_distance(data_distribution_1,data_distribution_2)
print(f"Distancia de Jensen-Shannon: {jensen_shannon_dist}")