import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, e_in_dim, e_out_dim):
        super(GATLayer, self).__init__()
        self.embed_node = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim + e_in_dim, 1, bias=False)
        self.to_node_fc = nn.Linear(out_dim + e_in_dim, out_dim, bias=False)
        self.edge_linear = nn.Linear(2 * out_dim + e_in_dim,
                                     e_out_dim,
                                     bias=False)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['w']],
                       dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a, negative_slope=0.1)}

    def message_func(self, edges):
        return {
            'h': edges.src['h'],
            'e': edges.data['e'],
            'w': edges.data['w']
        }

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        t = torch.cat([nodes.mailbox['h'], nodes.mailbox['w']], dim=-1)
        t = self.to_node_fc(t)
        h = torch.sum(alpha * t, dim=1)
        return {'h': h}

    def edge_calc(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['w']],
                       dim=1)
        w = self.edge_linear(z2)
        return {'w': w}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = self.embed_node(h)
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            g.apply_edges(self.edge_calc)
            # h_readout = dgl.mean_nodes(g, 'h')
            # gh = dgl.broadcast_nodes(g, h_readout)
            # return torch.cat((g.ndata['h'], gh), dim=1), g.edata['w']
            return g.ndata['h'], g.edata['w']


class MultiHeadGATLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 e_in_dim,
                 e_out_dim,
                 num_heads,
                 use_gpu=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        self.use_gpu = use_gpu
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, e_in_dim, e_out_dim))

    def forward(self, g, h, merge):
        if self.use_gpu:
            g.edata['w'] = g.edata['w'].cuda()
        outs = list(map(lambda x: x(g, h), self.heads))
        outs = list(map(list, zip(*outs)))
        head_outs = outs[0]
        edge_outs = outs[1]
        if merge == 'flatten':
            head_outs = torch.cat(head_outs, dim=1)
            edge_outs = torch.cat(edge_outs, dim=1)
        elif merge == 'mean':
            head_outs = torch.mean(torch.stack(head_outs), dim=0)
            edge_outs = torch.mean(torch.stack(edge_outs), dim=0)
        g.edata['w'] = edge_outs
        return head_outs, edge_outs


class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, use_gpu=True):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.gat = nn.ModuleList()

        self.gat.append(
            MultiHeadGATLayer(in_dim, hidden_dim, 12, 128, heads, use_gpu))
        for l in range(1, num_layers):
            self.gat.append(
                MultiHeadGATLayer(
                    hidden_dim * heads,
                    hidden_dim,
                    128 * heads,
                    128,
                    heads,
                    use_gpu,
                ))

        self.linear_e = nn.Sequential(
            nn.Linear(128 * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        self.linear_h = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 3),
        )

    def forward(self, g, h):
        for l in range(self.num_layers - 1):
            h, _ = self.gat[l](g, h, merge='flatten')
            h = F.elu(h)
        h, e = self.gat[-1](g, h, merge='mean')

        # Graph level prediction
        g.ndata['h'] = h
        h_readout = dgl.mean_nodes(g, 'h')
        h_pred = self.linear_h(h_readout)

        # Edge prediction
        eh = dgl.broadcast_edges(g, h_readout)
        e_fused = torch.cat((eh, e), dim=1)
        e_pred = self.linear_e(e_fused)

        return h_pred, e_pred
