import torch
import torch.nn as nn

class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_atom, act=None, bn=False):
        super(GCNLayer, self).__init__()

        self.use_bn = bn
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(n_atom)
        self.activation = act

    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)
        return out, adj

class SkipConnection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out

class GatedSkipConnection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out

    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)

class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_atom, bn=True, sc='gsc'):
        super(GCNBlock, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        n_atom,
                                        nn.ReLU() if i!=n_layer-1 else None,
                                        bn))
        self.relu = nn.ReLU()
        if sc=='gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc=='sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."

    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out, adj = layer((x if i == 0 else out), adj)
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)
        return out, adj

class ReadOut(nn.Module):
    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        #out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out

class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        #import IPython; IPython.embed()
        out = self.linear(x)
        if self.activation != None:
            out = self.activation(out)
        return out

class GCNNet(nn.Module):
    def __init__(self, n_block, n_layer, in_dim, hidden_dim, n_atom, pred_dim1, pred_dim2,
                 pred_dim3, out_dim, bn=False, sc='gsc'):
        super(GCNNet, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(GCNBlock(n_layer,
                                        in_dim if i==0 else hidden_dim,
                                        hidden_dim,
                                        hidden_dim,
                                        n_atom,
                                        bn,
                                        sc))
        self.readout = ReadOut(hidden_dim, pred_dim1, act=nn.ReLU())
        self.pred1 = Predictor(pred_dim1, pred_dim2, act=nn.ReLU())
        self.pred2 = Predictor(pred_dim2, pred_dim3, act=nn.Tanh())
        self.pred3 = Predictor(pred_dim3, out_dim)

    def forward(self, x, adj):
        for i, block in enumerate(self.blocks):
            out, adj = block((x if i==0 else out), adj)
        #import IPython; IPython.embed()
        out = self.readout(out)

        #out = self.pred1(out)
        #out = self.pred2(out)
        out = self.pred3(out)
        return out
