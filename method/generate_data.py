from gp_model import ExactGP, MyBotorchModel
from botorch.models.gp_regression import SingleTaskGP
import torch
from multiprocessing import Pool
from botorch.optim import optimize_acqf
from botorch.acquisition.predictive_entropy_search import qPredictiveEntropySearch
from utils import find_closest_grid_point
from botorch.acquisition.utils import get_optimal_samples

# PARAMETERS
dim = 3
domain_size = 1
grid_size = 11
N = 2000 # NUMBER OF ROLLOUTS
T = 30 # TIME HORIZON
M = 500000 # MONTE CARLO SAMPLES
policy = 'thompson'

x = torch.linspace(0, domain_size, grid_size)
x = torch.cartesian_prod(*[x for _ in range(dim)])

priors = torch.empty(N, T, *[grid_size for _ in range(dim)])
posteriors = torch.empty(N, T, *[grid_size for _ in range(dim)])
new_xs = torch.empty(N, T, dim)
new_ys = torch.empty(N, T, 1)

def run_one_episode(n):
    model = ExactGP(None, None)

    priors_in = torch.empty(T, *[grid_size for _ in range(dim)])
    posteriors_in = torch.empty(T, *[grid_size for _ in range(dim)])
    new_xs_in = torch.empty(T, dim)
    new_ys_in = torch.empty(T, 1)
    regrets_in = torch.empty(T, 1)

    # HYPERPARAMETERS

    model.likelihood.noise_covar.noise = 0.0100
    model.covar_module.base_kernel.lengthscale = 0.2
    model.covar_module.outputscale = 1.0

    model.eval()
    posterior = model(x)

    samples = posterior.sample(sample_shape=torch.Size([M]))
    argmaxes = torch.argmax(samples, axis=1)
    unique, counts = torch.unique(argmaxes, sorted = True, return_counts=True)
    probs = counts / M

    prior_argmax_distribution = torch.zeros_like(x, dtype = torch.float32)
    prior_argmax_distribution[unique, torch.zeros_like(unique)] = probs
    prior_argmax_distribution = prior_argmax_distribution[:, 0].view(*[grid_size for _ in range(dim)])

    true_f = posterior.sample(sample_shape=torch.Size([1]))
    max_f = true_f.max()

    to_be_observed_points = torch.randperm(grid_size ** dim)[:T] # indices of points to be observed

    for t in range(T):
        if policy == 'random':
            new_x = to_be_observed_points[t].view(1,1) #index of new point
            new_y = true_f[0, new_x]
            new_y = torch.flatten(new_y)
            new_x = x[new_x, 0]
        elif policy == 'thompson':
            new_x = argmaxes[0].view(1,1)
            new_y = true_f[0, new_x]
            new_y = torch.flatten(new_y)
            new_x = x[new_x].squeeze(0)
        else:
            raise ValueError('Policy not recognized')
        
        if t == 0:
            train_x = new_x
            train_y = new_y
            model = ExactGP(train_x, train_y)
            model.likelihood.noise_covar.noise = 0.0100
            model.covar_module.base_kernel.lengthscale = 0.2
            model.covar_module.outputscale = 1.0
        else:
            train_x = torch.cat((train_x, new_x), axis = 0)
            train_y = torch.cat((train_y, new_y), axis = 0)
            model.set_train_data(train_x, train_y, strict=False)

        model.eval()
        posterior = model(x)

        samples = posterior.sample(sample_shape=torch.Size([M]))
        argmaxes = torch.argmax(samples, axis=1)
        unique, counts = torch.unique(argmaxes, sorted = True, return_counts=True)
        probs = counts / M

        posterior_argmax_distribution = torch.zeros_like(x, dtype = torch.float32)
        posterior_argmax_distribution[unique, torch.zeros_like(unique)] = probs
        posterior_argmax_distribution = posterior_argmax_distribution[:, 0].view(*[grid_size for _ in range(dim)])


        global_idx = n*T + t
        print(global_idx, 'is done out of', T*N)

        priors_in[t] = prior_argmax_distribution
        posteriors_in[t] = posterior_argmax_distribution
        new_xs_in[t] = new_x
        new_ys_in[t] = new_y
        regrets_in[t] = max_f - new_y

        prior_argmax_distribution = posterior_argmax_distribution
    
    return priors_in.unsqueeze(0), posteriors_in.unsqueeze(0), new_xs_in.unsqueeze(0), new_ys_in.unsqueeze(0), regrets_in.unsqueeze(0)

if __name__ == '__main__':

    pool = Pool(processes = 40)
    results = pool.map(run_one_episode, range(N))
    pool.close()
    pool.join()

    priors_list, posteriors_list, new_xs_list, new_ys_list, regrets_list = zip(*results)

    priors = torch.cat(priors_list, dim=0)
    posteriors = torch.cat(posteriors_list, dim=0)
    new_xs = torch.cat(new_xs_list, dim=0)
    new_ys = torch.cat(new_ys_list, dim=0)
    regrets = torch.cat(regrets_list, dim=0)

    torch.save(priors, "priors_d{}_g{}_N{}_T{}_M{}_p{}_dim{}_lowls.pt".format(domain_size, grid_size, N, T, M, policy, dim))
    torch.save(posteriors, "posteriors_d{}_g{}_N{}_T{}_M{}_p{}_dim{}_lowls.pt".format(domain_size, grid_size, N, T, M, policy, dim))
    torch.save(new_xs, "new_xs_d{}_g{}_N{}_T{}_M{}_p{}_dim{}_lowls.pt".format(domain_size, grid_size, N, T, M, policy, dim)) 
    torch.save(new_ys, "new_ys_d{}_g{}_N{}_T{}_M{}_p{}_dim{}_lowls.pt".format(domain_size, grid_size, N, T, M, policy, dim))
    torch.save(regrets, "regrets_d{}_g{}_N{}_T{}_M{}_p{}_dim{}_lowls.pt".format(domain_size, grid_size, N, T, M, policy, dim))