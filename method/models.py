import torch
import matplotlib.pyplot as plt
import torch.nn as nn

try:
    import conv4d as nn4  # optional, external; only needed for dim == 4 in priorConvEncoder
except ImportError:
    nn4 = None


class yEncoder(nn.Module):

    def __init__(self, layer_sizes=[1, 5, 20, 50], grid_size=11, residuals=[0, 3], dropout_rate=0.2):
        super().__init__()

        self.residuals = residuals
        self.grid_size = grid_size
        self.subnetworks_list = []

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        for j in range(1, len(residuals)):
            self.layers_list = []

            for i in range(residuals[j-1], residuals[j]):
                self.layers_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

                if i != residuals[j] - 1:
                    self.layers_list.append(nn.BatchNorm1d(layer_sizes[i+1]))
                    self.layers_list.append(nn.ReLU())
                    self.layers_list.append(nn.Dropout(dropout_rate))

            self.subnetworks_list.append(nn.Sequential(*self.layers_list))

        self.subnetworks = nn.ModuleList(self.subnetworks_list)

        self.residuals_list = []
        for i in range(len(residuals) - 1):
            self.residuals_list.append(nn.Linear(layer_sizes[residuals[i]], layer_sizes[residuals[i+1]]))

        self.residuals = nn.ModuleList(self.residuals_list)
        self.activations = nn.ModuleList()

        for i in range(len(residuals) - 1):
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(layer_sizes[residuals[i+1]])))

    def forward(self, x):
        for i in range(len(self.subnetworks)):
            identity = x
            x = self.subnetworks[i](x)
            x += self.residuals[i](identity)

            if i != len(self.subnetworks) - 1:
                x = self.activations[i](x)

                if self.dropout_rate > 0:
                    x = self.dropout(x)

        return x


class likelihoodDecoder(nn.Module):

    def __init__(self, layer_sizes=[50, 35, 20], grid_size=11, residuals=[0, 2], dim=3, dropout_rate=0.2):
        super().__init__()

        self.residuals = residuals
        self.grid_size = grid_size
        self.subnetworks_list = []
        self.dim = dim

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        for j in range(1, len(residuals)):
            self.layers_list = []

            for i in range(residuals[j-1], residuals[j]):
                self.layers_list.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

                if i != residuals[j] - 1:
                    self.layers_list.append(nn.BatchNorm1d(layer_sizes[i+1]))
                    self.layers_list.append(nn.ReLU())
                    self.layers_list.append(nn.Dropout(dropout_rate))

            self.subnetworks_list.append(nn.Sequential(*self.layers_list))

        self.subnetworks = nn.ModuleList(self.subnetworks_list)

        self.residuals_list = []
        for i in range(len(residuals) - 1):
            self.residuals_list.append(nn.Linear(layer_sizes[residuals[i]], layer_sizes[residuals[i+1]]))

        self.residuals = nn.ModuleList(self.residuals_list)
        self.activations = nn.ModuleList()

        for i in range(len(residuals) - 1):
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(layer_sizes[residuals[i+1]])))

    def generate_distance_tensor(self):
        size = 2 * self.grid_size - 1
        coords = torch.stack(torch.meshgrid(*[torch.arange(size) for _ in range(self.dim)]), dim=-1)
        mid = (size - 1) / 2
        dists = (coords - mid).float() ** 2
        dists = torch.sqrt(dists.sum(dim=-1))
        return dists

    def wrap(self, z, scale_factor=10):
        z = z.unsqueeze(1)
        size = (scale_factor) * (z.size(2) - 1) + 1
        m = nn.Upsample(size=size, mode='linear', align_corners=True)
        batch_size = z.size(0)
        lookup = m(z)

        if self.dim == 2:
            lookup = lookup.squeeze(1).unsqueeze(2).tile(1, 1, 2*self.grid_size - 1)
        elif self.dim == 3:
            lookup = lookup.squeeze(1).unsqueeze(2).unsqueeze(3).tile(1, 1, 2*self.grid_size - 1, 2*self.grid_size - 1)

        dists = self.generate_distance_tensor()
        dists.to(z.device)
        idxs = torch.round(dists * scale_factor).type(torch.int64)
        d = [1] * self.dim
        idxs = idxs.unsqueeze(0).tile(batch_size, *d).to(z.device)

        return torch.gather(lookup, 1, idxs)

    def wrap_crop_update(self, z, new_x, log_prior):
        # wrap
        if self.dim == 1:
            batch_size = z.size(0)
            zz = z[:, :self.grid_size]  # decoder output (e.g. 40) is wider than grid_size; use the first grid_size entries
            new_z = torch.zeros((batch_size, 2 * self.grid_size - 1), dtype=z.dtype, device=z.device)
            new_z[:, self.grid_size-1:2*self.grid_size-1] = zz
            new_z[:, :self.grid_size] = zz.flip(dims=[1])[:, :]
        else:
            new_z = self.wrap(z).to(new_x.device)

        # crop
        if self.dim == 1:
            idx = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x), 1)).to(new_x.device) - 1 - new_x.tile((1, self.grid_size)) + self.grid_size
            pl = torch.gather(new_z, 1, idx)
        elif self.dim == 2:
            new_x1 = new_x[:, 0].unsqueeze(1)
            new_x2 = new_x[:, 1].unsqueeze(1)
            idx1 = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x1), 1)).to(new_x1.device) - 1 - new_x1.tile((1, self.grid_size)) + self.grid_size
            idx2 = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x2), 1)).to(new_x2.device) - 1 - new_x2.tile((1, self.grid_size)) + self.grid_size

            new_z = new_z.contiguous()
            lin_idx = idx2.unsqueeze(1).tile(1, self.grid_size, 1) + (new_z.size(-1) * idx1).unsqueeze(2).tile(1, 1, self.grid_size)
            lin_idx = lin_idx.view(-1, self.grid_size * self.grid_size)

            new_z = new_z.view(-1, new_z.size(1) * new_z.size(2))
            pl = new_z.gather(-1, lin_idx)
            pl = pl.view(-1, self.grid_size, self.grid_size)
            pl = pl.view(-1, self.grid_size ** self.dim)

        elif self.dim == 3:
            new_x1 = new_x[:, 0].unsqueeze(1)
            new_x2 = new_x[:, 1].unsqueeze(1)
            new_x3 = new_x[:, 2].unsqueeze(1)
            idx1 = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x1), 1)).to(new_x1.device) - 1 - new_x1.tile((1, self.grid_size)) + self.grid_size
            idx2 = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x2), 1)).to(new_x2.device) - 1 - new_x2.tile((1, self.grid_size)) + self.grid_size
            idx3 = torch.arange(self.grid_size).unsqueeze(0).tile((len(new_x3), 1)).to(new_x3.device) - 1 - new_x3.tile((1, self.grid_size)) + self.grid_size

            new_z = new_z.contiguous()
            lin_idx = idx3.unsqueeze(1).unsqueeze(2).tile(1, self.grid_size, self.grid_size, 1) + \
                      (new_z.size(-1) * idx2).unsqueeze(1).unsqueeze(3).tile(1, self.grid_size, 1, self.grid_size) + \
                      (new_z.size(-1) * new_z.size(-1) * idx1).unsqueeze(2).unsqueeze(3).tile(1, 1, self.grid_size, self.grid_size)
            lin_idx = lin_idx.view(-1, self.grid_size * self.grid_size * self.grid_size)

            new_z = new_z.view(-1, new_z.size(1) * new_z.size(2) * new_z.size(3))
            pl = new_z.gather(-1, lin_idx)
            pl = pl.view(-1, self.grid_size, self.grid_size, self.grid_size)
            pl = pl.view(-1, self.grid_size ** self.dim)

        # update
        posterior = pl + log_prior
        return posterior

    def forward(self, z, new_x, log_prior):
        for i in range(len(self.subnetworks)):
            identity = z
            z = self.subnetworks[i](z)
            z += self.residuals[i](identity)

            if i != len(self.subnetworks) - 1:
                z = self.activations[i](z)

                if self.dropout_rate > 0:
                    z = self.dropout(z)

        z = self.wrap_crop_update(z, new_x, log_prior)
        return z


class priorEncoder(nn.Module):

    def __init__(self, layer_sizes=[100, 80, 50], grid_size=100, residuals=[0, 2], init_zero=True, dropout_rate=0.2):
        super().__init__()

        self.residuals = residuals
        self.grid_size = grid_size
        self.subnetworks_list = []

        self.pre_norm = nn.BatchNorm1d(layer_sizes[0])

        for j in range(1, len(residuals)):
            self.layers_list = []
            for i in range(residuals[j-1], residuals[j]):
                linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
                if init_zero:
                    torch.nn.init.zeros_(linear.weight)
                self.layers_list.append(linear)

                if i != residuals[j] - 1:
                    self.layers_list.append(nn.ReLU())
                    self.layers_list.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.subnetworks_list.append(nn.Sequential(*self.layers_list))

        self.subnetworks = nn.ModuleList(self.subnetworks_list)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.residuals_list = []
        for i in range(len(residuals) - 1):
            linear = nn.Linear(layer_sizes[residuals[i]], layer_sizes[residuals[i+1]])
            if init_zero:
                torch.nn.init.zeros_(linear.weight)
            self.residuals_list.append(linear)

        self.residuals = nn.ModuleList(self.residuals_list)
        self.activations = nn.ModuleList()

        for i in range(len(residuals) - 1):
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(layer_sizes[residuals[i+1]])))

    def forward(self, x):
        x = self.pre_norm(x)

        for i in range(len(self.subnetworks)):
            identity = x
            x = self.subnetworks[i](x)
            x += self.residuals[i](identity)

            if i != len(self.subnetworks) - 1:
                x = self.activations[i](x)

                if self.dropout_rate > 0:
                    x = self.dropout(x)

        return x


class priorConvEncoder(nn.Module):

    def __init__(self, num_layers=4, grid_size=11, residuals=[0, 2, 4], channels=2, kernel_size=2,
                 latent_dim=50, init_zero=True, dim=3, dropout_rate=0.5):
        super().__init__()

        self.residuals = residuals
        self.grid_size = grid_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.subnetworks_list = []
        self.dim = dim

        self.pre_norm = nn.BatchNorm1d(self.grid_size ** self.dim)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        for j in range(1, len(residuals)):
            self.layers_list = []

            for i in range(residuals[j-1], residuals[j]):

                if i == residuals[j-1] and j == 1:
                    if self.dim == 3:
                        linear = nn.Conv3d(1, self.channels, self.kernel_size)
                    elif self.dim == 2:
                        linear = nn.Conv2d(1, self.channels, self.kernel_size)
                    elif self.dim == 1:
                        linear = nn.Conv1d(1, self.channels, self.kernel_size)
                else:
                    if self.dim == 4:
                        linear = nn4.Conv4d(self.channels, self.channels, self.kernel_size, init_zero=init_zero)
                    elif self.dim == 3:
                        linear = nn.Conv3d(self.channels, self.channels, self.kernel_size)
                    elif self.dim == 2:
                        linear = nn.Conv2d(self.channels, self.channels, self.kernel_size)
                    elif self.dim == 1:
                        linear = nn.Conv1d(self.channels, self.channels, self.kernel_size)

                if init_zero:
                    if self.dim > 3:
                        pass
                    else:
                        torch.nn.init.zeros_(linear.weight)

                self.layers_list.append(linear)

                if i != residuals[j] - 1:
                    if self.dim == 4:
                        self.layers_list.append(nn4.BatchNorm4d(self.channels))
                    elif self.dim == 3:
                        self.layers_list.append(nn.BatchNorm3d(self.channels))
                    elif self.dim == 2:
                        self.layers_list.append(nn.BatchNorm2d(self.channels))
                    elif self.dim == 1:
                        self.layers_list.append(nn.BatchNorm1d(self.channels))

                    self.layers_list.append(nn.ReLU())

            self.subnetworks_list.append(nn.Sequential(*self.layers_list))

        self.subnetworks = nn.ModuleList(self.subnetworks_list)
        self.res_ker = 3

        self.residuals_list = []
        for i in range(len(residuals) - 1):
            if i == 0:
                if self.dim == 4:
                    linear = nn4.Conv4d(1, self.channels, self.res_ker, init_zero=init_zero)
                elif self.dim == 3:
                    linear = nn.Conv3d(1, self.channels, self.res_ker)
                elif self.dim == 2:
                    linear = nn.Conv2d(1, self.channels, self.res_ker)
                elif self.dim == 1:
                    linear = nn.Conv1d(1, self.channels, self.res_ker)
            else:
                if self.dim == 4:
                    linear = nn4.Conv4d(self.channels, self.channels, self.res_ker, init_zero=init_zero)
                elif self.dim == 3:
                    linear = nn.Conv3d(self.channels, self.channels, self.res_ker)
                elif self.dim == 2:
                    linear = nn.Conv2d(self.channels, self.channels, self.res_ker)
                elif self.dim == 1:
                    linear = nn.Conv1d(self.channels, self.channels, self.res_ker)

            if init_zero:
                if self.dim > 3:
                    pass
                else:
                    torch.nn.init.zeros_(linear.weight)

            self.residuals_list.append(linear)

        self.residuals = nn.ModuleList(self.residuals_list)
        self.activations = nn.ModuleList()

        for i in range(len(residuals) - 1):
            if self.dim == 4:
                bn = nn4.BatchNorm4d(self.channels)
            elif self.dim == 3:
                bn = nn.BatchNorm3d(self.channels)
            elif self.dim == 2:
                bn = nn.BatchNorm2d(self.channels)
            elif self.dim == 1:
                bn = nn.BatchNorm1d(self.channels)

            self.activations.append(nn.Sequential(nn.ReLU(), bn))

        self.flatten_size = self._compute_flatten_size()
        self.linear = nn.Linear(self.flatten_size, self.latent_dim)

    def _compute_flatten_size(self):
        """Compute the size of the flattened feature map before the final linear layer."""
        with torch.no_grad():
            a = [self.grid_size] * self.dim
            dummy = torch.zeros(2, *a)
            dummy = dummy.view(dummy.size(0), -1)
            dummy = self.pre_norm(dummy)
            dummy = dummy.view(dummy.size(0), *a)
            dummy = dummy.unsqueeze(1)

            for i in range(len(self.subnetworks)):
                identity = dummy
                dummy = self.subnetworks[i](dummy)
                dummy += self.residuals[i](identity)

                if i != len(self.subnetworks) - 1:
                    dummy = self.activations[i](dummy)

            dummy = dummy.view(dummy.size(0), -1)
            return dummy.size(1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.pre_norm(x)
        a = [self.grid_size] * self.dim
        x = x.view(x.size(0), *a)
        x = x.unsqueeze(1)

        for i in range(len(self.subnetworks)):
            identity = x
            x = self.subnetworks[i](x)
            x += self.residuals[i](identity)

            if i != len(self.subnetworks) - 1:
                x = self.activations[i](x)

                if self.dropout_rate > 0:
                    x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class posteriorConvDecoder(nn.Module):

    def __init__(
        self,
        num_layers=6,
        sizes=[5, 6, 7, 8, 9, 10, 11],
        grid_size=100,
        residuals=[0, 3, 6],
        channels=2,
        kernel_size=3,
        latent_dim=50,
        init_zero=True,
        dim=3,
        dropout_rate=0.5
    ):
        super().__init__()

        self.num_layers = num_layers
        self.sizes = sizes
        self.grid_size = grid_size
        self.residuals = residuals
        self.channels = channels
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.dim = dim

        self.pre_trans = nn.Linear(latent_dim, sizes[0] ** dim)

        a = [sizes[self.residuals[1]]] * dim
        b = [sizes[self.residuals[2]]] * dim

        if dim == 3:
            self.res1 = nn.Sequential(nn.Conv3d(1, channels, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(a), mode='nearest'))
            self.res2 = nn.Sequential(nn.Conv3d(channels, 1, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(b), mode='nearest'))
        elif dim == 2:
            self.res1 = nn.Sequential(nn.Conv2d(1, channels, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(a), mode='nearest'))
            self.res2 = nn.Sequential(nn.Conv2d(channels, 1, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(b), mode='nearest'))
        elif dim == 1:
            self.res1 = nn.Sequential(nn.Conv1d(1, channels, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(a), mode='nearest'))
            self.res2 = nn.Sequential(nn.Conv1d(channels, 1, kernel_size, padding=(kernel_size // 2)),
                                      nn.Upsample(size=(b), mode='nearest'))

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        in_ch = 1

        for i in range(num_layers):
            if i == 0:
                out_ch = channels
            elif i == num_layers - 1:
                out_ch = 1
            else:
                out_ch = channels

            if dim == 3:
                conv = nn.Conv3d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2))
            elif dim == 2:
                conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2))
            elif dim == 1:
                conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2))

            if init_zero:
                nn.init.zeros_(conv.weight)

            if i != 2 and i != 5:
                relu = nn.ReLU()
                if dim == 3:
                    bn = nn.BatchNorm3d(out_ch)
                elif dim == 2:
                    bn = nn.BatchNorm2d(out_ch)
                elif dim == 1:
                    bn = nn.BatchNorm1d(out_ch)

            a = [sizes[i+1]] * dim
            upsample = nn.Upsample(size=(a), mode='nearest')

            if i != 2 and i != 5:
                layer = nn.Sequential(conv, relu, bn, upsample)
            else:
                layer = nn.Sequential(conv, upsample)

            self.layers.append(layer)
            in_ch = out_ch

        if dim == 3:
            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm3d(channels))
        elif dim == 2:
            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(channels))
        elif dim == 1:
            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(channels))

    def forward(self, z):
        z = self.pre_trans(z)
        a = [self.sizes[0]] * self.dim
        x = z.view(z.size(0), *a)
        x = x.unsqueeze(1)

        skip0 = x.clone()

        for i in range(self.num_layers):
            x = self.layers[i](x)

            if i == 2:
                x += self.res1(skip0)
                x = self.activation(x)
                skip3 = x.clone()

            if i == 5:
                x += self.res2(skip3)

        x = x.squeeze(1)
        return x


class posteriorDecoder(nn.Module):

    def __init__(self, layer_sizes=[50, 80, 100], grid_size=100, residuals=[0, 2], init_zero=True):
        super().__init__()

        self.residuals = residuals
        self.grid_size = grid_size
        self.subnetworks_list = []

        for j in range(1, len(residuals)):
            self.layers_list = []
            for i in range(residuals[j-1], residuals[j]):
                linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
                if init_zero:
                    torch.nn.init.zeros_(linear.weight)
                self.layers_list.append(linear)

                if i != residuals[j] - 1:
                    self.layers_list.append(nn.ReLU())
                    self.layers_list.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.subnetworks_list.append(nn.Sequential(*self.layers_list))

        self.subnetworks = nn.ModuleList(self.subnetworks_list)

        self.residuals_list = []
        for i in range(len(residuals) - 1):
            linear = nn.Linear(layer_sizes[residuals[i]], layer_sizes[residuals[i+1]])
            if init_zero:
                torch.nn.init.zeros_(linear.weight)
            self.residuals_list.append(linear)

        self.residuals = nn.ModuleList(self.residuals_list)
        self.activations = nn.ModuleList()

        for i in range(len(residuals) - 1):
            self.activations.append(nn.Sequential(nn.ReLU(), nn.BatchNorm1d(layer_sizes[residuals[i+1]])))

    def forward(self, x):
        for i in range(len(self.subnetworks)):
            identity = x
            x = self.subnetworks[i](x)
            x += self.residuals[i](identity)

            if i != len(self.subnetworks) - 1:
                x = self.activations[i](x)

        return x


class convNetLeonArDBO(nn.Module):

    # latent size used for each input dimensionality
    LATENT_DIMS = {1: 100, 2: 200, 3: 400}

    def __init__(self, dim, grid_size=100, latent_dim=None, dropout=0.5):
        super().__init__()

        if latent_dim is None:
            latent_dim = self.LATENT_DIMS[dim]

        if dim == 2:
            self.yEncoder = yEncoder(layer_sizes=[1, 8, 12, 40, 80, 140, 200], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.ystarEncoder = yEncoder(layer_sizes=[1, 8, 12, 40, 80, 140, 200], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.priorEncoder = priorConvEncoder(channels=15, dropout_rate=dropout, latent_dim=200, init_zero=True, dim=2)

            self.likelihoodDecoder = likelihoodDecoder(layer_sizes=[200, 180, 160, 140, 120, 100, 80], residuals=[0, 3, 6], dropout_rate=0.5, dim=2)
            self.posteriorDecoder = posteriorConvDecoder(channels=15, dropout_rate=0.5, init_zero=True, dim=2, latent_dim=200)

            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(latent_dim))
        elif dim == 3:
            self.yEncoder = yEncoder(layer_sizes=[1, 16, 24, 80, 160, 280, 400], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.ystarEncoder = yEncoder(layer_sizes=[1, 16, 24, 80, 160, 280, 400], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.priorEncoder = priorConvEncoder(channels = 30, dropout_rate=dropout, latent_dim=400, init_zero=True, dim = 3)

            self.likelihoodDecoder = likelihoodDecoder(layer_sizes=[400, 360, 320, 280, 240, 200, 160], residuals=[0, 3, 6], dropout_rate=0.5, dim = 3)
            self.posteriorDecoder = posteriorConvDecoder(channels = 30, dropout_rate=0.5, init_zero=True, dim = 3, latent_dim=400)

            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(latent_dim))

        elif dim == 1:
            self.yEncoder = yEncoder(layer_sizes=[1, 4, 6, 20, 40, 70, 100], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.ystarEncoder = yEncoder(layer_sizes=[1, 4, 6, 20, 40, 70, 100], residuals=[0, 3, 6], grid_size=grid_size, dropout_rate=dropout)
            self.priorEncoder = priorConvEncoder(channels = 7, dropout_rate=dropout, latent_dim=100, init_zero=True, dim = 1)

            self.likelihoodDecoder = likelihoodDecoder(layer_sizes=[100, 90, 80, 70, 60, 50, 40], residuals=[0, 3, 6], dropout_rate=0.5, dim = 1)
            self.posteriorDecoder = posteriorConvDecoder(channels = 7, dropout_rate=0.5, init_zero=True, dim = 1, latent_dim=100)

            self.activation = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(latent_dim))

        else:
            raise ValueError(f"dim must be 1, 2 or 3, got {dim}")

    def forward(self, new_y, y_star, new_x, log_prior):
        log_prior = log_prior.view(log_prior.size(0), -1)

        z = self.yEncoder(new_y)
        z += self.ystarEncoder(y_star)
        z += self.priorEncoder(log_prior)
        z = self.activation(z)

        posterior = self.likelihoodDecoder(z, new_x, log_prior)
        posterior += self.posteriorDecoder(z).view(posterior.size(0), -1)
        posterior = posterior.view(posterior.size(0), -1)
        posterior = nn.functional.log_softmax(posterior, dim=1)
        return posterior
