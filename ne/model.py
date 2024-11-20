import torch
import torch_geometric as tg

class BFSEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFSEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.relu = torch.nn.ReLU()
            
    def forward(self, x):
        bfs_label = x.bfs_label.unsqueeze(1)
        input = torch.cat([bfs_label, x.h_bfs], dim=1)
        return self.relu(self.linear(input))


class BFSDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFSDecoder, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
            
    def forward(self, z, h):
        input = torch.cat([z, h], dim=1)
        return self.linear(input) 
    
    
class BFSTermination(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFSTermination, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        mean = torch.mean(x, dim=0)
        input = torch.cat((x, mean.unsqueeze(0)), dim=0)
        out = self.linear(input)
        return out[-1]

class BFEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFEncoder, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.relu = torch.nn.ReLU()
            
    def forward(self, x):
        bf_label = x.bf_label.unsqueeze(1)
        input = torch.cat([bf_label, x.h_bf], dim=1)
        return self.relu(self.linear(input))


class BFDecoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFDecoder, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
            
    def forward(self, z, h):
        input = torch.cat([z, h], dim=1)
        return self.linear(input) # TODO you will need to fix for predecessor

class BFPredecessorDecoder(tg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(BFPredecessorDecoder, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(2 * in_channels + edge_dim, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.edge_dim) # when having only one edge feature, shape will be (num_edges), so it has to be transformed
        input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.linear(input)
    
    def update(self, aggr_out):
        
        return aggr_out


class BFTermination(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BFTermination, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
            
    def forward(self, x):
        mean = torch.mean(x, dim=0)
        input = torch.cat((x, mean.unsqueeze(0)), dim=0)
        out = self.linear(input)
        return out[-1]


class MPNNProcessor(tg.nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr): #in_channels is dim of node features
        super(MPNNProcessor, self).__init__(aggr=aggr) 
        self.edge_dim = edge_dim
        M_in_channels = in_channels * 2 + edge_dim
        self.M = torch.nn.Linear(M_in_channels, out_channels)
        U_in_channels = in_channels + out_channels
        self.U = torch.nn.Linear(U_in_channels, out_channels)
        self.relu = torch.nn.ReLU()

    def reset_parameters(self):
        self.M.reset_parameters()
        self.U.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr): # x_i and x_j are node feature matrices of shape (num_edges, input_node_features_dim)
        edge_attr = edge_attr.view(-1, self.edge_dim) # when having only one edge feature, shape will be (num_edges), so it has to be transformed
        input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.relu(self.M(input))  # returns matrix of shape (num_edges, out_channels) which is aggregated with aggr function in aggr_out

    def update(self, aggr_out, x): # aggr_out has shape (num_nodes, out_channels)
        input = torch.cat([x, aggr_out], dim=-1)
        return self.relu(self.U(input))

class NeuralExecutor(torch.nn.Module):
    def __init__(self, Processor, Encoders, Decoders, TerminationNetworks, in_dims, out_dims, hidden_dim, pred=None, **kwargs): # TODO maybe rewrite so you can give encoders and termination and decoder as param, seconda is optional, and have param self.parallel if you have to do 2 at once and concat
        super(NeuralExecutor, self).__init__()
        if not isinstance(Encoders, list) or not isinstance(Decoders, list) or not isinstance(TerminationNetworks, list):
            raise TypeError("Arguments Encoders, Decoders and TerminationNetworks must be a list")
        if len(Encoders) != len(Decoders) or len(Decoders) != len(TerminationNetworks):
            raise AssertionError('You should have equal number of encoders, decoders and termination networks, one for each algorithm')
        if len(Encoders) > 2:
            raise NotImplementedError('Currently only implemented for one or two algorithms running simultaneously.')
        
        self.n = len(Encoders)
        self.processor = Processor(hidden_dim, hidden_dim, **kwargs)
        self.encoders = []
        self.decoders = []
        self.terminators = []
        for i in range(self.n):
            self.encoders.append(Encoders[i](in_dims[i] + hidden_dim, hidden_dim)) # input to the encoder is x(t) and h(t-1), output is z(t)
            self.decoders.append(Decoders[i](2 * hidden_dim, out_dims[i])) # input to the decoder is z(t) and h(t), output is y(t)
            self.terminators.append(TerminationNetworks[i](hidden_dim, 1)) # output dimension for termination network always 1
        if pred is not None:
            self.pred = pred
        else:
            self.pred = None

    def forward(self, data): 
        out = []
        for i in range(self.n):
            z = self.encoders[i](data)
            h = self.processor(z,data.edge_index, data.edge_attr)
            y = self.decoders[i](z, h)
            tau = self.terminators[i](h)
            out.append((y, tau, h))
        if self.pred is not None:
            out.append(self.pred(h, data.edge_index, data.edge_attr))
        return out


