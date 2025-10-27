import torch
import torch.nn as nn
import torch.optim as optim

import itertools

from ..base_model import BaseModel
from ...nn.network import Network
from ...nn.optimizer import Optimizer
from ...nn.loss import HMLC

class GraphEmbedding(nn.Module):
    def __init__(self, max_node_type = 10, node_embed_dim = 32, max_edge_type = 5, edge_embed_dim = 16):
        super(GraphEmbedding, self).__init__()
        self.node_embedding = nn.Embedding(max_node_type, node_embed_dim)
        
        self.edge_embedding = nn.Embedding(max_edge_type, edge_embed_dim)
        self._init_weights()

    def forward(self, X):
        V, E = X  # V: Node types, E: Edge types
        V_emb = self.node_embedding(V.long())
        E_emb = self.edge_embedding(E.long())  # Assuming edge type is at index 3
        return V_emb, E_emb

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embedding.weight)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

class GMN(BaseModel):
    dependencies = [Network, Optimizer]
    def __init__(self, optimizer_params,
                node_dim, max_node_type,
                edge_dim, max_edge_type,
                hidden_dim, latent_dim,
                num_layers=2, activation=nn.ReLU, message_passing_steps=4):
        super(GMN, self).__init__()
        self.optimizer_params = optimizer_params

        self.node_dim = node_dim
        self.max_node_type = max_node_type
        self.edge_dim = edge_dim
        self.max_edge_type = max_edge_type
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.message_passing_steps = message_passing_steps
        self.num_layers = num_layers

    def forward(self, X):
        V_emb, E_emb, src, dst = self._get_embeddings(X)

        for _ in range(self.message_passing_steps):  # Number of message passing steps
            V_emb, E_emb = self._pass_message((V_emb, E_emb, src, dst))
        latent_embedding = V_emb[:,-1 -4 -64 -64:].mean(dim=1)
        return latent_embedding


    def get_loss(self, X):
        #  Augment1       Augment2
        # (V1,E1, V2,E2, ... , labels)
        labels = X[-1]
        features = []
        # Correctly pair node and edge tensors: (V1,E1), (V2,E2), etc.
        for b in zip(X[::2], X[1::2]):  # Exclude labels from edge pairing
            V = self.forward(b)
            features.append(V)
        # classify expects raw logits; CrossEntropyLoss applies softmax internally.
        V_emb = torch.stack(features, dim=1)  # Shape (B, num_views, n_classes)
        #print(V_emb.shape, labels.shape)
        loss = self.criterion(V_emb, labels)

        self.logger.add("Loss", torch.ones((1,1)) * loss.item())
        #print(loss)
        return loss

    def _get_embeddings(self, X):
        V, E = X
        V_emb, E_emb = self.embedding((V[:,:,0], E[:, :, 3]))
        # Note: edge_dim should already account for the +1 from concatenating edge weight
        E_emb = torch.cat((E_emb, E[:, :, 2:3]), dim=-1)  # Append edge weight to edge embedding
        
        src = E[:, :, 0].long().unsqueeze(-1).expand(-1, -1, V_emb.size(2))
        dst = E[:, :, 1].long().unsqueeze(-1).expand(-1, -1, V_emb.size(2))

        return V_emb, E_emb, src, dst

    def _pass_message(self, X):
        #(Batch, Nodes, Node_dim), (Batch, Edges, Edge_dim), (Batch, Nodes) , (Batch, Nodes)
        V_emb, E_emb, src, dst = X
        nodes = V_emb.size(1)
        
        V_emb_in =  torch.gather(V_emb, dim=1, index=src)
        V_emb_out = torch.gather(V_emb, dim=1, index=dst)

        v_1_input = torch.cat((V_emb_in, V_emb_out, E_emb), dim=-1)
        v_1_output = self.net_v1(v_1_input)

        # Aggregate messages for each destination node by MEAN (not sum)
        v_1_agg = []
        for d, v in zip(dst, v_1_output):
            # Sum messages per destination node
            agg = torch.zeros(nodes, v.size(-1), device=v.device, dtype=v.dtype)
            agg = agg.index_add(0, d[:, 0], v)

            # Count number of incoming messages per node and avoid div-by-zero
            counts = torch.bincount(d[:, 0], minlength=nodes).to(dtype=agg.dtype, device=agg.device).unsqueeze(-1)
            counts = counts.clamp(min=1)

            # Compute mean by dividing summed messages by counts
            agg = agg / counts

            v_1_agg.append(agg)
        v_1_agg = torch.stack(v_1_agg)  # Shape (B, N, latent_dim)

        v2_input = torch.concat((V_emb, v_1_agg), dim=-1)
        v2_output = self.net_v2(v2_input)

        edge_input = torch.cat((V_emb_in, V_emb_out, E_emb), dim=-1)
        edge_output = self.net_e(edge_input)

        return v2_output, edge_output
    def back_propagate(self, loss):
        return self.optimizer.step(loss)



    # =====================================================================
	#INITIALIZATION FUNCTIONS

    def _init_networks(self):
        self.net_v1, self.net_v1_params = self.build_net_v1()
        self.net_v2, self.net_v2_params = self.build_net_v2()
        self.net_e, self.net_e_params = self.build_net_e()
        #self.classify, self.classify_params = self.build_classifier()
        self.embedding, self.embedding_params = self.build_embedding()

        self.nets = [self.net_v1, self.net_v2, self.net_e, self.embedding]
        self.params = [self.net_v1_params, self.net_v2_params, self.net_e_params, self.embedding_params]
        #self._init_parameters()

    def _init_optimizers(self):
        self.optimizer = self._configure_optimizer()
        self.optimizers = [self.optimizer]

    def _init_loss_functions(self):
        self.criterion = HMLC()
        self.loss_functions = [self.criterion]

    def _load_loss_functions(self):
        self.criterion = self.loss_functions[0]

    def _init_logger(self):
        tags = ['Loss']
        keys = [['loss']]
        self.logger = self.create_logger(tags, keys)
    
    # =====================================================================


    def _configure_optimizer(self):
        params = itertools.chain(*[net.parameters() for net in self.nets])
        return Optimizer(params, **self.optimizer_params)

    def _init_parameters(self):
        for net in self.nets:
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        #init with a small bias
                        nn.init.constant_(module.bias, 0.01)
    def build_classifier(self):
        arch_params = {
            "layers" : [self.node_dim*5, self.hidden_dim, 13],
            "blocks" : [nn.Linear, self.activation],
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params
    def build_net_v1(self):
        arch_params = {
            "layers" : [self.node_dim * 2 + self.edge_dim] + [self.hidden_dim for _ in range(self.num_layers)] + [self.latent_dim],
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],  
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params
    def build_net_v2(self):
        arch_params = {
            "layers" : [self.latent_dim + self.node_dim] + [self.hidden_dim for _ in range(self.num_layers)] + [self.node_dim],
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],  
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params

    def build_net_e(self):
        arch_params = {
            "layers" : [self.node_dim * 2 + self.edge_dim] + [self.hidden_dim for _ in range(self.num_layers)] + [self.edge_dim],
            "blocks" : [nn.Linear, nn.LayerNorm, self.activation],  
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params

    def build_embedding(self):
        arch_params = {
            "blocks" : [GraphEmbedding],
            "block_args" : [ {
                "max_node_type": self.max_node_type,
                "node_embed_dim": self.node_dim,
                "max_edge_type": self.max_edge_type,
                "edge_embed_dim": self.edge_dim-1,
            }],
        }
        network_params = {
            "arch_params": arch_params,
        }
        return Network(**network_params).to(self.device), network_params



