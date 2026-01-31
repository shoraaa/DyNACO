import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn

# GNN for edge embeddings
# Single GNN layer for checkpointing
class GNNLayer(nn.Module):
    def __init__(self, units, act_fn, agg_fn):
        super().__init__()
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)
        self.e_lin0 = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, w, edge_index):
        x0 = x
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        w0 = w
        w1 = self.e_lin0(w0)
        w2 = torch.sigmoid(w0)
        x = x0 + self.act_fn(self.v_bn(x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w

class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=6, units=32, act_fn='silu', agg_fn='mean', grad_checkpointing=False): 
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.grad_checkpointing = grad_checkpointing
        
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        
        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])
        
        self._register_load_state_dict_pre_hook(self._load_compat_hook)

    def _load_compat_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Backward compatibility hook to map old state_dict keys to new GNNLayer structure.
        Old: v_lins1.0.weight -> New: layers.0.v_lin1.weight
        """
        keys_to_rewrite = []
        for key in state_dict.keys():
            if not key.startswith(prefix):
                continue
            
            # Remove prefix to handle local key parsing
            local_key = key[len(prefix):]
            
            # Mapping rules
            # v_lins1.i.weight -> layers.i.v_lin1.weight
            if local_key.startswith("v_lins1."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin1.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins2."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin2.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins3."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin3.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_lins4."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_lin4.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("e_lins0."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.e_lin0.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("v_bns."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.v_bn.{suffix}"
                keys_to_rewrite.append((key, new_key))
            elif local_key.startswith("e_bns."):
                parts = local_key.split('.')
                idx = parts[1]
                suffix = ".".join(parts[2:])
                new_key = f"{prefix}layers.{idx}.e_bn.{suffix}"
                keys_to_rewrite.append((key, new_key))
        
        for old_key, new_key in keys_to_rewrite:
            if new_key not in state_dict: # Don't overwrite if somehow both exist
                state_dict[new_key] = state_dict.pop(old_key)

    def forward(self, x, edge_index, edge_attr):
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        
        for layer in self.layers:
            if self.grad_checkpointing and self.training:
                 x, w = torch.utils.checkpoint.checkpoint(layer, x, w, edge_index, use_reentrant=False)
            else:
                 x, w = layer(x, w, edge_index)
        return w

# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device
    def __init__(self, units_list, act_fn, sigmoid_output=True):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])
        self.sigmoid_output = sigmoid_output
    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                if self.sigmoid_output:
                    x = torch.sigmoid(x) # last layer
        return x

# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu', logit_net=False):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn, sigmoid_output=not logit_net)
    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)
    
# Single GNN layer for checkpointing
class GNNLayer(nn.Module):
    def __init__(self, units, act_fn, agg_fn):
        super().__init__()
        self.act_fn = act_fn
        self.agg_fn = agg_fn
        self.v_lin1 = nn.Linear(units, units)
        self.v_lin2 = nn.Linear(units, units)
        self.v_lin3 = nn.Linear(units, units)
        self.v_lin4 = nn.Linear(units, units)
        self.v_bn = gnn.BatchNorm(units)
        self.e_lin0 = nn.Linear(units, units)
        self.e_bn = gnn.BatchNorm(units)

    def forward(self, x, w, edge_index):
        x0 = x
        x1 = self.v_lin1(x0)
        x2 = self.v_lin2(x0)
        x3 = self.v_lin3(x0)
        x4 = self.v_lin4(x0)
        w0 = w
        w1 = self.e_lin0(w0)
        w2 = torch.sigmoid(w0)
        x = x0 + self.act_fn(self.v_bn(x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
        w = w0 + self.act_fn(self.e_bn(w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w

# Checkpointed version of EmbNet
class EmbNetCheckpoint(nn.Module):
    def __init__(self, depth=12, feats=2, edge_feats=6, units=32, act_fn='silu', agg_fn='mean'): 
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.edge_feats = edge_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.e_lin0 = nn.Linear(self.edge_feats, self.units)
        
        self.layers = nn.ModuleList([
            GNNLayer(self.units, self.act_fn, self.agg_fn) for _ in range(self.depth)
        ])

    def forward(self, x, edge_index, edge_attr):
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        
        for layer in self.layers:
            if self.training:
                 x, w = torch.utils.checkpoint.checkpoint(layer, x, w, edge_index, use_reentrant=False)
            else:
                 x, w = layer(x, w, edge_index)
        return w

class Net(nn.Module):
    def __init__(self, feats=2, logit_net=False, grad_checkpointing=False):
        super().__init__()
        if grad_checkpointing:
            self.emb_net = EmbNetCheckpoint(feats=feats, edge_feats=6)
        else:
            self.emb_net = EmbNet(feats=feats, edge_feats=6)
        # self.par_net_phe = ParNet()
        self.par_net_heu = ParNet(logit_net=logit_net)
    def forward(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu
    
    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False
