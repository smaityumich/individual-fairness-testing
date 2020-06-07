import numpy as np

def get_data(minority_group_prop = 0.1, sample_size = 400, seed = 1):

    """
    generates data for simulation in 2-dimensional feature space

    parameters:
        minority_group_prop (float): proportion for minority group
        sample_size (int): sample size
        seed (int): seed for random data generation
    """

    
    mu_minority = np.array([1.5, 0])
    mu_majority = np.array([-1.5, 0])
    w_minority  = np.array([0.2, -0.01])
    w_majority  = np.array([-0.2, -0.01])
    d = 2
    sigma = 0.25
    np.random.seed(seed)
    x = np.random.normal(0, scale=sigma, size=(sample_size,d))
    z = np.random.choice([False, True], size=sample_size, p=[minority_group_prop, 1-minority_group_prop])
    x[z,:] = x[z,:] + mu_majority
    x[~z,:] = x[~z,:] + mu_minority
    y = np.zeros(sample_size)
    noise = np.random.normal(size = (sample_size,)) * 0.01
    #noise = np.zeros(sample_size)
    y[z] = np.sign(np.dot(x[z,:],w_majority) - np.dot(w_majority,mu_majority) + noise[z])
    y[~z] = np.sign(np.dot(x[~z,:],w_minority) - np.dot(w_minority,mu_minority) + noise[~z])
    np.save('data/x.npy', x)
    np.save('data/y.npy', (y+1)/2)
    np.save('data/group.npy', z)

if __name__ == "__main__":
    get_data()