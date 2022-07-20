# 3D rigid body transfomation group and corresponding Lie algebra
import torch
from torch import sin, cos      


#######################################################
def genmat():
    return mat(torch.eye(6))


#######################################################
def mat(x):                 # size: [*, 3] -> [*, 3, 3]  
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)
    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)


#######################################################
def sinc1(t):               # sinc1: t -> sin(t)/t
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1 - t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42))  # Taylor series O(t^8)
    r[c] = sin(t[c]) / t[c]
    return r


#######################################################
def sinc2(t):               # sinc2: t -> (1 - cos(t)) / (t**2)
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = 1 / 2 * (1 - t2[s] / 12 * (1 - t2[s] / 30 * (1 - t2[s] / 56)))  
    r[c] = (1 - cos(t[c])) / t2[c]
    return r


#######################################################
def sinc3(t):               # sinc3: t -> (t - sin(t)) / (t**3)
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)
    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72))) 
    r[c] = (t[c] - sin(t[c])) / (t[c] ** 3)
    return r


#######################################################
def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)
    W = mat(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)
    R = I + sinc1(t) * W + sinc2(t) * S
    V = I + sinc2(t) * W + sinc3(t) * S
    p = V.bmm(v.contiguous().view(-1, 3, 1))
    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)
    return g.view(*(x.size()[0:-1]), 4, 4)


#######################################################
def transform(g, a):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    if len(g.size()) == len(a.size()):
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b


class ExpMap(torch.autograd.Function):  # Exp: se(3) -> SE(3)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = genmat().to(x)
        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        dg = dg.to(grad_output)
        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)
        return grad_input

Exp = ExpMap.apply
