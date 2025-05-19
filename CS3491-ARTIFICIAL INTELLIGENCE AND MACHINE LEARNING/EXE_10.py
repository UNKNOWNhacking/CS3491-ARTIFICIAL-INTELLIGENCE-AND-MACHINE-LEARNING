import numpy as np

# Generate synthetic data
np.random.seed(42)
p_z_true = 0.6
p_x_given_z_true = {0: 0.3, 1: 0.8}
n_samples = 1000
Z = np.random.binomial(1, p_z_true, n_samples)
X = np.array([np.random.binomial(1, p_x_given_z_true[z]) for z in Z])

# Initialize parameters
p_z_current = 0.5
p_x_z0_current = 0.5
p_x_z1_current = 0.5
max_iter = 100
tolerance = 1e-4
log_likelihoods = []

for iteration in range(max_iter):
    # E-Step: Compute posteriors P(Z=1 | X)
    e_z1 = []
    for x in X:
        if x == 1:
            prob_z1 = p_z_current * p_x_z1_current
            prob_z0 = (1 - p_z_current) * p_x_z0_current
        else:
            prob_z1 = p_z_current * (1 - p_x_z1_current)
            prob_z0 = (1 - p_z_current) * (1 - p_x_z0_current)
        total = prob_z0 + prob_z1
        e_z1.append(prob_z1 / total if total != 0 else 0)
    e_z1 = np.array(e_z1)
    
    # M-Step: Update parameters
    new_p_z = np.mean(e_z1)
    numerator_x1_z0 = np.sum((X == 1) * (1 - e_z1))
    denominator_z0 = np.sum(1 - e_z1)
    new_p_x_z0 = numerator_x1_z0 / denominator_z0 if denominator_z0 != 0 else 0
    numerator_x1_z1 = np.sum((X == 1) * e_z1)
    denominator_z1 = np.sum(e_z1)
    new_p_x_z1 = numerator_x1_z1 / denominator_z1 if denominator_z1 != 0 else 0
    
    # Check convergence
    deltas = [
        abs(new_p_z - p_z_current),
        abs(new_p_x_z0 - p_x_z0_current),
        abs(new_p_x_z1 - p_x_z1_current)
    ]
    p_z_current, p_x_z0_current, p_x_z1_current = new_p_z, new_p_x_z0, new_p_x_z1
    
    # Log likelihood
    log_likelihood = 0
    for x in X:
        if x == 1:
            term = (1 - p_z_current)*p_x_z0_current + p_z_current*p_x_z1_current
        else:
            term = (1 - p_z_current)*(1 - p_x_z0_current) + p_z_current*(1 - p_x_z1_current)
        log_likelihood += np.log(term) if term != 0 else 0
    log_likelihoods.append(log_likelihood)
    
    if all(delta < tolerance for delta in deltas):
        print(f"Converged at iteration {iteration + 1}")
        break

print(f"Estimated P(Z=1): {p_z_current:.4f} (True: 0.6)")
print(f"Estimated P(X=1|Z=0): {p_x_z0_current:.4f} (True: 0.3)")
print(f"Estimated P(X=1|Z=1): {p_x_z1_current:.4f} (True: 0.8)")