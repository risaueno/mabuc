
with pm.Model() as sleep_model:

    # Create the alpha and beta parameters
    # Assume a normal distribution
    alpha = pm.Normal('alpha', mu=0.0, tau=0.05, testval=0.0)
    beta = pm.Normal('beta', mu=0.0, tau=0.05, testval=0.0)

    # The sleep probability is modeled as a logistic function
    p = pm.Deterministic('p', 1. / (1. + tt.exp(beta * time + alpha)))

    # Create the bernoulli parameter which uses observed data to inform the algorithm
    observed = pm.Bernoulli('obs', p, observed=sleep_obs)

    # Using Metropolis Hastings Sampling
    step = pm.Metropolis()

    # Draw the specified number of samples
    sleep_trace = pm.sample(N_SAMPLES, step=step)
