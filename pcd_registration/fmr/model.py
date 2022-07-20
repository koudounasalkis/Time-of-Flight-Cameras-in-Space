import torch
import se_math.se3 as se3
import se_math.invmat as invmat


#######################################################
def flatten(x):                         # Flatten a feature
    return x.view(x.size(0), -1)


#######################################################
def symfn_max(x):                       # Calculate max pooling [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


#######################################################
def _mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


########################################################
class MLPNet(torch.nn.Module):

    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = _mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


#######################################################
class PointNet(torch.nn.Module):        # Encoder Network
    
    def __init__(self, dim_k=1024):
        super().__init__()
        scale = 1
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.sy = symfn_max

    def forward(self, points):
        x = points.transpose(1, 2)  # [B, 3, N]
        x = self.h1(x)
        x = self.h2(x)              # [B, K, N]
        x = flatten(self.sy(x))
        return x


#######################################################
class Decoder(torch.nn.Module):         # Decoder Network
    
    def __init__(self, num_points=2048, bottleneck_size=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)
        self.fc1 = torch.nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = torch.nn.Linear(self.bottleneck_size, bottleneck_size // 2)
        self.fc3 = torch.nn.Linear(bottleneck_size // 2, bottleneck_size // 4)
        self.fc4 = torch.nn.Linear(bottleneck_size // 4, self.num_points * 3)
        self.th = torch.nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        return x


#######################################################
class SolveRegistration(torch.nn.Module):   # Feature-metric registration NN
    
    def __init__(self, ptnet, decoder=None, isTest=False):
        super().__init__()
        self.encoder = ptnet
        self.decoder = decoder

        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp                  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform      # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        delta = 1.0e-2
        dt_initial = torch.autograd.Variable(torch.Tensor(
            [delta, delta, delta, delta, delta, delta]))    # 3 rotation angles and 3 translation
        self.dt = torch.nn.Parameter(dt_initial.view(1, 6), requires_grad=True)

        self.last_err = None                # Results
        self.prev_r = None
        self.g_series = None                # Debug purpose      
        self.g = None                       # Estimated transformation T
        self.isTest = isTest                # Whether it is testing

    def estimate_t(self, p0, p1, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
               
        if p0_zero_mean:            # Normalization
            p0_m = p0.mean(dim=1)   # [B, N, 3] -> [B, 3]
            a0 = a0.clone()
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0
        if p1_zero_mean:            # Normalization
            p1_m = p1.mean(dim=1)   # [B, N, 3] -> [B, 3]
            a1 = a1.clone()
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        # IC algorithm to estimate the transformation
        g0 = torch.eye(4).to(q0).view(1, 4, 4).expand(q0.size(0), 4, 4).contiguous()
        r, g, loss_ende = self.ic_algorithm(g0, q0, q1, maxiter, xtol, is_test=self.isTest)
        self.g = g

        if p0_zero_mean or p1_zero_mean:    # Re-normalization
            est_g = self.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            self.g = est_g
            est_gs = self.g_series          # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            self.g_series = est_gs

        # Return feature-metric projection error (r) and encoder-decoder loss (loss_ende)
        return r, loss_ende    

    def ic_algorithm(self, g0, p0, p1, maxiter, xtol, is_test=False):
        training = self.encoder.training
        batch_size = p0.size(0)

        self.last_err = None
        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        loss_enco_deco = 0.0
        f0 = self.encoder(p0)                       # [B, N, 3] -> [B, K]
        dt = self.dt.to(p0).expand(batch_size, 6)   # Convert to the type of p0, [B, 6]
        J = self.approx_Jac(p0, f0, dt)
        try:
            Jt = J.transpose(1, 2)                  # [B, 6, K]
            H = Jt.bmm(J)                           # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt)                        # [B, 6, K]
        except RuntimeError as err:
            self.last_err = err
            print(err)
            f1 = self.encoder(p1)                   # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, -1

        itr = 0
        r = None
        for itr in range(maxiter):
            p = self.transform(g.unsqueeze(1), p1)  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f1 = self.encoder(p)                    # [B, N, 3] -> [B, K]
            r = f1 - f0                             # [B,K]
            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)
            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0               # No update
                break
            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()
            self.prev_r = r

        # Feature-metric projection error (r), updated transformation (g), encoder-decoder loss
        self.encoder.train(training)
        return r, g, loss_enco_deco

    def approx_Jac(self, p0, f0, dt):
        batch_size = p0.size(0)
        num_points = p0.size(1)

        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :])                    # [6, 6]
            D = self.exp(-d)                            # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()       # [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1))     # x [B, 1, N, 3] -> [B, 6, N, 3]

        f0 = f0.unsqueeze(-1)                           # [B, K, 1]
        f1 = self.encoder(p.view(-1, num_points, 3))
        f = f1.view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]
        df = f0 - f                                     # [B, K, 6]
        J = df / dt.unsqueeze(1)                        # [B, K, 6]
        return J

    def update(self, g, dx):
        dg = self.exp(dx)                               # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        return dg.matmul(g)

    @staticmethod
    def rsq(r):
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction='sum')


#######################################################
class FMR:

    def __init__(self):
        self.dim_k = 1024
        self.max_iter = 10   
        self._loss_type = 1         # 0: unsupervised, 1: semi-supervised

    def create_model(self):
        ptnet = PointNet(dim_k=self.dim_k)
        decoder = Decoder()
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, p0, p1, device):
        solver.eval()
        with torch.no_grad():
            p0 = torch.tensor(p0,dtype=torch.float).to(device)  # template (1, N, 3)
            p1 = torch.tensor(p1,dtype=torch.float).to(device)  # source (1, M, 3)
            solver.estimate_t(p0, p1, self.max_iter)

            est_g = solver.g                                    # (1, 4, 4)
            g_hat = est_g.cpu().contiguous().view(4, 4)         # [1, 4, 4]
            return g_hat