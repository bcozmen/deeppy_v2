import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from ..base_model import BaseModel
from ...nn.network import Network
from ...nn.optimizer import Optimizer
from ...nn.loss import WasserSteinLoss
from ...nn.modules.pooling import MultiQueryAttentionPooling

from sklearn.metrics import precision_score, recall_score, f1_score

#Changes
#Graph embedding -> W_e + E_e
#Scatter add vectorized aggregation
#Use edge features instead of node features
#Smaller MLP networks and add dropout
class GraphEmbedding(nn.Module):
    def __init__(self, max_node_type = [15,100], node_embed_dim = 32, max_edge_type = 5, edge_embed_dim = 16):
        super(GraphEmbedding, self).__init__()
        self.node_embedding_1 = nn.Embedding(max_node_type[0], node_embed_dim // 2)
        self.node_embedding_2 = nn.Embedding(max_node_type[1], node_embed_dim // 2)
        self.edge_embedding = nn.Embedding(max_edge_type, edge_embed_dim)
        self.weight_embedding = nn.Linear(1, edge_embed_dim)
        self._init_weights()

    def forward(self, X):
        V, E = X  # V: Node types, E: Edge types
        V_emb_1 = self.node_embedding_1(V[...,0].long())
        V_emb_2 = self.node_embedding_2(V[...,1].long())
        V_emb = torch.cat((V_emb_1, V_emb_2), dim=-1)
        E_emb = self.edge_embedding(E[...,3].long())  # Assuming edge type is at index 3
        W_emb = self.weight_embedding(E[...,2:3])  # Assuming edge weight is at index 2

        return V_emb, E_emb + W_emb

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding_1.weight)
        nn.init.xavier_uniform_(self.node_embedding_2.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        nn.init.xavier_uniform_(self.weight_embedding.weight)
        if self.weight_embedding.bias is not None:
            nn.init.constant_(self.weight_embedding.bias, 0.01)

class ClassifierHeads(nn.Module):
    def __init__(self, network_params, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            network = Network(**network_params)  # your MLP
            self.heads.append(network)

    def forward(self, x):
        """
        x: (batch_size, num_heads, input_dim)
        returns: (batch_size, num_heads, num_classes)
        """
        batch_size = x.size(0)
        out = []

        # Loop over heads (still one loop over num_heads)
        for head_idx, head in enumerate(self.heads):
            out.append(head(x[:, head_idx, :]))  # (batch_size, num_classes)

        # Stack along head dimension
        return torch.stack(out, dim=1)  # (batch_size, num_heads, num_classes)


class GMN(BaseModel):
    dependencies = [Network, Optimizer]
    def __init__(self, optimizer_params,
                node_dim, max_node_type,
                edge_dim, max_edge_type,
                latent_dim = None,
                num_layers=1, activation=nn.ReLU, dropout=0.1,
                message_passing_steps=4, num_classes=13,
                n_task = 3,
                lamb = [1.0, 1.0 ,0.01]):
        super(GMN, self).__init__()
        self.optimizer_params = optimizer_params

        self.node_dim = node_dim
        self.max_node_type = max_node_type
        self.edge_dim = edge_dim
        self.max_edge_type = max_edge_type
        self.latent_dim = latent_dim
        if latent_dim is None:
            self.latent_dim = node_dim * 2
        self.activation = activation
        self.message_passing_steps = message_passing_steps
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.n_task = n_task
        self.lamb = torch.tensor(lamb).to(self.device)


    def forward(self, X):
        V_emb, E_emb, src, dst = self._get_embeddings(X)
        gates = []
        for _ in range(self.message_passing_steps):  # Number of message passing steps
            V_emb, E_emb, gate = self._pass_message((V_emb, E_emb, src, dst, ))
            gates.append(gate)
        latent_embedding, entropy = self.pooling(V_emb)
        return latent_embedding, entropy, torch.stack(gates).mean()  # Pooling over edge embeddings
    


    def get_loss(self, X):
        #  Augment1       Augment2
        # (V1,E1, V2,E2, ... , labels)
        B = X[0].size(0)
        labels = X[-1] # Shape (B, num_views, n_task, num_bins)
        num_bins = labels.size(-1)
        outs = []
        for b in zip(X[:-1:2], X[1:-1:2]):  # Exclude labels from edge pairing
            outs.append(self.forward(b))
        features, entropies, gates = list(zip(*outs))

        latent_embedding = torch.stack(features, dim=1)  # Shape (B, num_views, n_task, latent_dim)

        latent_embedding = latent_embedding.reshape(-1, self.n_task, self.node_dim)  # Shape (B * num_views, n_task, latent_dim)
        histograms = self.classifier_heads(latent_embedding)  # Shape (B * num_views, n_task, num_bins)
        histograms = histograms.reshape(B, -1, self.n_task, num_bins) # Reshape back to (B, num_views, n_task, latent_dim)
        loss = self.wassersteinLoss(histograms, labels)

        percentiles = [25, 50, 75, 90, 99]
        loss = loss.sum(dim=-1)  # Sum over bins -> (B, num_views, n_task)
        loss_flat = loss.view(-1).abs()
        
        calculated_percentiles = []
        for p in percentiles:
            calculated_percentiles.append(torch.quantile(loss_flat, p/100))
        self.logger.add("WS distance Percentiles", torch.tensor([calculated_percentiles]).T)

        
        loss_by_nerf = loss.mean(dim = (0,2)) # Mean over batch and tasks
        self.logger.add("WS By Nerf Type", loss_by_nerf.unsqueeze(0).T)

        loss_by_task = loss.mean(dim = (0,1)) # Mean over batch and nerf types
        self.logger.add("WS By Task", loss_by_task.unsqueeze(0).T)
        total_loss = loss.mean()
        self.logger.add("Loss", torch.tensor([[total_loss.item()]]).T)
        self.logger.add("Pooling Entropy", entropies)
        self.logger.add("Gates Entropy", gates)
        #self.logger.add("Pooling Entropy", torch.tensor([[e.item() for e in entropies]]).T)
        #self.logger.add("Gates Entropy", torch.tensor([[g.item() for g in gates]]).T)

        loss = (self.lamb[0]*total_loss) \
            +  (self.lamb[1]/len(entropies)) * sum(entropies) \
            +  (self.lamb[2]/len(gates)) * sum(gates)
        return loss

    def _get_embeddings(self, X):
        V, E = X
        V_emb, E_emb = self.embedding(X)
        src = E[:, :, 0].long().unsqueeze(-1).expand(-1, -1, V_emb.size(2))
        dst = E[:, :, 1].long().unsqueeze(-1).expand(-1, -1, V_emb.size(2))
        return V_emb, E_emb, src, dst

    def _v1(self, V_emb_in, V_emb_out, E_emb, dst):
        v_1_input = torch.cat((V_emb_in, V_emb_out, E_emb), dim=-1)
        v_1_output_all = self.net_v1(v_1_input)
        v_1_output, gates = v_1_output_all[..., :-1], v_1_output_all[..., -1]  # Remove the extra dimension used for counting
        
        gates, entropy = self._transform_gates(gates, dst=dst)  # Normalize gates over incoming edges
        v_1_output = v_1_output * gates.unsqueeze(-1)  # Zero
        return v_1_output, entropy

    def _transform_gates(self, gates, dst):
        # Softmax over incoming edges - promote to FP32 for numerical stability with AMP
        gates_fp32 = gates.float()
        exp_gates = torch.exp(gates_fp32 - torch.max(gates_fp32, dim=-1, keepdim=True)[0])
        dst_idx = dst[..., 0].long()  # shape (B, E)
        B, E = gates.shape
        nodes = dst_idx.max().item() + 1

        # Compute the normalization factor for each node
        norm_factor = exp_gates.new_zeros((B, nodes))
        norm_factor.scatter_add_(1, dst_idx, exp_gates)
        norm_factor = norm_factor.gather(1, dst_idx)  # shape (B, E)

        normalized_gates = exp_gates / (norm_factor + 1e-6)

        entropy = self._calculate_normalized_entropy(normalized_gates, dst)

        return normalized_gates.to(gates.dtype), entropy

    def _v2(self, V_emb, v_1_agg):
        v2_input = torch.concat((V_emb, v_1_agg), dim=-1)
        v2_output = self.net_v2(v2_input)
        return v2_output

    def _e(self, V_emb_in, V_emb_out, E_emb):
        edge_input = torch.cat((V_emb_in, V_emb_out, E_emb), dim=-1)
        edge_output = self.net_e(edge_input)
        return edge_output

    def _aggregate(self, v_1_output, dst, nodes):
        dst_idx = dst[..., 0].long()  # shape (B, E)
        B, E, F = v_1_output.shape
        agg = torch.zeros(B, nodes, F, device=v_1_output.device, dtype=v_1_output.dtype)
        index = dst_idx.unsqueeze(-1).expand(-1, -1, F)  # shape (B, E, F)
        agg = agg.scatter_add_(1, index, v_1_output)
        return agg

    def _pass_message(self, X, skip = False):
        #(Batch, Nodes, Node_dim), (Batch, Edges, Edge_dim), (Batch, Nodes) , (Batch, Nodes)
        V_emb, E_emb, src, dst = X
        nodes = V_emb.size(1)
        
        V_emb_in =  torch.gather(V_emb, dim=1, index=src)
        V_emb_out = torch.gather(V_emb, dim=1, index=dst)
        v2_output = V_emb
        gates = 0.0
        if not skip:            
            v_1_output, gates = self._v1(V_emb_in, V_emb_out, E_emb, dst)
            v_1_agg = self._aggregate(v_1_output, dst, nodes) # (B, N, F)
            v2_output = self._v2(V_emb, v_1_agg)
            gates = gates.mean()
        edge_output = self._e(V_emb_in, V_emb_out, E_emb)

        return v2_output, edge_output, gates
    def _normalize_gates(self, gates, dst):
        #Turn gates into a probability distribution over incoming edges for each node
        dst_idx = dst[..., 0].long()  # shape (B, E)
        B, E = gates.shape
        nodes = dst_idx.max().item() + 1

        # Compute the normalization factor for each node - use FP32 for stability
        gates_fp32 = gates.float()
        norm_factor = gates_fp32.new_zeros((B, nodes))
        norm_factor.scatter_add_(1, dst_idx, gates_fp32)
        norm_factor = norm_factor.gather(1, dst_idx)  # shape (B, E)

        # Normalize the gates with safe epsilon
        gates_normalized = gates_fp32 / (norm_factor + 1e-6)

        return gates_normalized.to(gates.dtype)

    def _calculate_normalized_entropy(self, normalized_gates, dst):
        # Calculate entropy of the normalized gates - keep in FP32 for stability
        B, E = normalized_gates.shape
        nodes = dst[..., 0].long().max().item() + 1
        dst = dst[..., 0].long() # shape (B, E, 1)

        # Clamp to avoid log(0) and ensure numerical stability in FP16
        log_gates = torch.log(torch.clamp(normalized_gates, min=1e-6))
        entropy_values = -normalized_gates * log_gates
        entropy = entropy_values.new_zeros((B, nodes))
        entropy.scatter_add_(1, dst, entropy_values)

        #normalize over log of incoming edges
        counts = dst.new_zeros((B, nodes))
        ones = torch.ones_like(normalized_gates).to(counts.dtype)
        counts.scatter_add_(1, dst, ones)

        #entropy = entropy / (torch.log(counts + 1e-6) + 1e-6)
        entropy = entropy[counts > 0] / torch.clamp(torch.log(counts[counts > 0] + 1.0), min=1e-6)
        avg_entropy = entropy.mean()
        return avg_entropy
    def back_propagate(self, loss):
        return self.optimizer.step(loss)



    # =====================================================================
	#INITIALIZATION FUNCTIONS

    def _init_networks(self):
        self.embedding, self.embedding_params = self.build_embedding()
        self.net_v1, self.net_v1_params = self.build_net_v1()
        self.net_v2, self.net_v2_params = self.build_net_v2()
        self.net_e, self.net_e_params = self.build_net_e()
        self.pooling , self.pooling_params = self.build_pooling()

        self.classifier_heads, self.classifier_heads_params = self.build_histogram_classifier()
        
    

        self.nets = [self.net_v1, self.net_v2, self.net_e, self.embedding, self.classifier_heads, self.pooling]
        self.params = [self.net_v1_params, self.net_v2_params, self.net_e_params, self.embedding_params, self.classifier_heads_params, self.pooling_params]
        #self._init_parameters()

    def _init_optimizers(self):
        self.optimizer = self._configure_optimizer()
        self.optimizers = [self.optimizer]

    def _init_loss_functions(self):
        self.wassersteinLoss = WasserSteinLoss()
        self.loss_functions = [self.wassersteinLoss]

    def _load_loss_functions(self):
        self.wassersteinLoss = self.loss_functions[0]

    def _init_logger(self):
        tags = ['Loss',  "Gates Entropy", "Pooling Entropy", "WS distance Percentiles", "WS By Nerf Type", "WS By Task" ]
        keys = [['Loss'],
                ["MLP", "Hash", "Triplane"],
                ["MLP", "Hash", "Triplane"],
                ['%25', '%50', '%75', '%90', '%99'],
                ["MLP", "Hash", "Triplane"],
                ["Red", "Green", "Blue"]]
        self.logger = self.create_logger(tags, keys)
    
    # =====================================================================


    def _configure_optimizer(self):
        decay_params = []
        nodecay_params = []

        for net in self.nets:
            for name, param in net.model.named_parameters():
                if not param.requires_grad:
                    continue

                if param.dim() >= 2 and "embedding" not in name.lower():
                    decay_params.append(param)
                else:
                    nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.optimizer_params["optimizer_args"]["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        del self.optimizer_params["optimizer_args"]["weight_decay"]
        
        return Optimizer(optim_groups, **self.optimizer_params)

    def _init_parameters(self):
        for net in self.nets:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        #init with a small bias
                        nn.init.constant_(module.bias, 0.01)

    def build_net_v1(self):
        inp_dim = self.node_dim * 2 + self.edge_dim
        arch_params = {
            "layers" : [inp_dim] + [self.latent_dim for _ in range(self.num_layers)] ,
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],
            "out_act" : self.activation,
            "residual": False
        }
        arch_params_2 = {
            "blocks" : [nn.Linear, nn.Dropout],
            "block_args" : [ 
                {
                    "in_features": self.latent_dim,
                    "out_features": self.latent_dim + 1,
                },
                { "p" : self.dropout }
            ],
        }
        network_params = {
            "arch_params": [arch_params, arch_params_2],
            "weight_init": "normal",
        }
        return Network(**network_params).to(self.device), network_params

    def build_net_v2(self):
        arch_params = {
            "layers" : [self.latent_dim + self.node_dim] + [self.latent_dim for _ in range(self.num_layers)],
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],  
            "out_act" : self.activation,
        }
        arch_params_2 = {
            "blocks" : [nn.Linear, nn.Dropout],
            "block_args" : [ 
                {
                    "in_features": self.latent_dim,
                    "out_features": self.node_dim,
                },
                { "p" : self.dropout }
            ],
        }
        network_params = {
            "arch_params": [arch_params, arch_params_2],
            "weight_init": "normal",
        }
        return Network(**network_params).to(self.device), network_params

    def build_net_e(self):
        arch_params = {
            "layers" : [self.node_dim * 2 + self.edge_dim] + [self.edge_dim * 2 for _ in range(self.num_layers)],
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],  
            "out_act" : self.activation,
        }
        arch_params_2 = {
            "blocks" : [nn.Linear, nn.Dropout],
            "block_args" : [ 
                {
                    "in_features": self.edge_dim *2,
                    "out_features": self.edge_dim,
                },
                { "p" : self.dropout }
            ],
        }
        network_params = {
            "arch_params": [arch_params, arch_params_2],
        }
        return Network(**network_params).to(self.device), network_params

    def build_embedding(self):
        arch_params = {
            "blocks" : [GraphEmbedding],
            "block_args" : [ {
                "max_node_type": self.max_node_type,
                "node_embed_dim": self.node_dim,
                "max_edge_type": self.max_edge_type,
                "edge_embed_dim": self.edge_dim,
            }],
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params

    def build_histogram_classifier(self):
        arc_params = {
            "layers" : [self.node_dim, self.node_dim*2, 10],
            "blocks" : [nn.Linear, self.activation],  
            "out_act" : nn.Softmax,
            "out_params" : {"dim": -1},
        }
        network_params = {
            "arch_params": arc_params,
            "weight_init": "normal",
        }

        arc_params_heads = {
            "blocks" : [ClassifierHeads],
            "block_args" : [ {
                "network_params": network_params,
                "num_heads": self.n_task,
            }],
        }
        network_params_heads = {
            "arch_params": arc_params_heads,
        }
        return Network(**network_params_heads).to(self.device), network_params_heads
    def build_pooling(self):
        arch_params = {
            "blocks" : [MultiQueryAttentionPooling],
            "block_args" : [ {
                "latent_dim": self.node_dim,
                "num_queries": self.n_task,
            }],
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params


def get_min_graph(V,E,attn,gate):
    attn = attn.to(E.device)
    thres = find_threshold(attn).to(E.device)

    attented_edges = E[attn > thres].to(E.device)  # (E_attended, 4)
    attented_attn = attn[attn > thres].to(E.device)
    node_attention = get_start_attention(V, attented_edges, attented_attn)

    attended_nodes = torch.cat((attended_edges[:,0], attended_edges[:,1])).long().unique().to(E.device)
    subgraph_edges = get_subgraph_edges(E, attended_nodes)
    agg = propagate_backwards(node_attention, subgraph_edges)
    
    return agg

def get_initial_state_(initial_state, E, attn):
    src_idx = E[:, 0].long().to(E.device)
    dst_idx = E[:, 1].long().to(E.device)
    
    initial_state.scatter_add_(0, src_idx, attn)
    initial_state.scatter_add_(0, dst_idx, attn)
    initial_state.div_(2.0)

def get_subgraph_edges(E, attended_nodes, gate):
    subgraph_edges_mask = torch.isin(E[:,1].long(), attended_nodes)
    subgraph_edges = E[subgraph_edges_mask]

    subgraph_gates = gate[subgraph_edges_mask]
    subgraph_edges = torch.cat((subgraph_edges, subgraph_gates.unsqueeze(-1)), dim=-1)  # (E_sub, 5)
    return subgraph_edges

def attention_flow(V, E, attn, gate):
    # V (N, F), E (E, 4), attn (E,), gate (T,E) or (E,)
    thres = find_threshold(attn).to(E.device)
    attn = attn.to(E.device)

    attented_edges = E[attn >= thres]  # (E_attended, 4)
    attn = attn[attn >= thres]

    T = gate.size(0) + 1 if gate.dim() == 2 else 2
    states = torch.zeros(T, V.size(0), device=V.device, dtype=V.dtype)
    get_initial_state_(states[0], attented_edges, attn)  # (N,)

    attended_nodes = torch.cat((attented_edges[:,0], attented_edges[:,1])).long().unique().to(E.device)
    subgraph_edges_mask = torch.isin(E[:,1].long(), attended_nodes).to(E.device)
    subgraph_edges = E[subgraph_edges_mask]

    for t in range(1, T):
        if gate.dim() == 2:
            subgraph_gates = gate[t-1][subgraph_edges_mask].to(E.device)
        else:
            subgraph_gates = gate[subgraph_edges_mask].to(E.device)
        subgraph_edges_t = torch.cat((subgraph_edges, subgraph_gates.unsqueeze(-1)), dim=-1)  # (E_sub, 5)

        states[t] = propagate_backwards(states[t-1], subgraph_edges_t)
    return states

