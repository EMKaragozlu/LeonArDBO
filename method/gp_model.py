import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.models.gpytorch import GPyTorchModel


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(0.0)
            )
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.num_outputs = 1

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MyBotorchModel(ExactGP, GPyTorchModel):
    _num_outputs = 1
