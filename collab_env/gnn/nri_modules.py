"""Neural Relational Inference (NRI) model implementation for modern PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


def encode_onehot(labels):
    """Convert labels to one-hot encoding."""
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """Gumbel-softmax sampling from PyTorch >= 1.2."""
    if hasattr(F, 'gumbel_softmax'):
        return F.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
    else:
        # Fallback implementation
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / tau
        y_soft = gumbels.softmax(dim)
        
        if hard:
            # Straight through gradient estimation
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret


class MLP(nn.Module):
    """Basic MLP without assertions."""
    
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.dropout_prob = do_prob
        
        # Standard initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.fc2(x)
        return x


class CNNEncoder(nn.Module):
    """CNN encoder for NRI."""
    
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNNEncoder, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # Reshape for conv1d: [num_sims * num_atoms, num_dims, num_timesteps]
        x = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(3), inputs.size(2))
        
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        
        pred = self.conv_predict(x)
        pred = pred.view(inputs.size(0), inputs.size(1), -1)
        return pred.mean(dim=2)


class MLPEncoder(nn.Module):
    """Basic MLP encoder following original NRI paper."""
    
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPEncoder, self).__init__()
        
        # Simple 2-layer MLP like original NRI
        self.fc1 = None  # Will be created dynamically based on input size
        self.fc2 = nn.Linear(256, 256)  # Fixed size for basic implementation
        self.fc_out = nn.Linear(2 * 256, n_out)  # 2x because we concatenate sender+receiver
        self.dropout_prob = do_prob
        
        # Standard initialization
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [batch, num_atoms, num_timesteps, num_dims] 
        # Flatten time and features
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        
        # Create first layer dynamically
        if self.fc1 is None:
            input_dim = x.size(-1)
            print(f"Creating basic encoder: {input_dim} -> 256 -> 256 -> 2")
            self.fc1 = nn.Linear(input_dim, 256).to(x.device)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
        
        # Process each node 
        batch_size, num_nodes = x.size(0), x.size(1)
        x = x.view(-1, x.size(-1))  # [batch*nodes, features]
        
        # Two-layer MLP for nodes
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.relu(self.fc2(x))
        x = x.view(batch_size, num_nodes, -1)  # [batch, nodes, hidden]
        
        # Create edge features by concatenating sender and receiver
        receivers = torch.matmul(rel_rec, x)    # [batch, edges, hidden]
        senders = torch.matmul(rel_send, x)     # [batch, edges, hidden]  
        edges = torch.cat([receivers, senders], dim=-1)  # [batch, edges, 2*hidden]
        
        # Final layer to edge types
        edges = edges.view(-1, edges.size(-1))  # [batch*edges, 2*hidden]
        edges = F.dropout(edges, self.dropout_prob, training=self.training)
        logits = self.fc_out(edges)  # [batch*edges, n_out]
        logits = logits.view(batch_size, -1, logits.size(-1))  # [batch, edges, n_out]
        
        return logits


class MLPDecoder(nn.Module):
    """Basic MLP decoder following original NRI paper."""
    
    def __init__(self, n_in_node, n_in_edge, n_hid, do_prob=0.):
        super(MLPDecoder, self).__init__()
        
        self.dropout_prob = do_prob
        
        # Will be created dynamically - use fixed 256 for basic implementation
        self.msg_fc1 = None
        self.msg_fc2 = nn.Linear(256, 256)
        self.out_fc1 = None  
        self.out_fc2 = nn.Linear(256, 256)
        self.out_fc3 = nn.Linear(256, 4)  # Output full state (px, py, vx, vy)
        
        # Standard initialization
        nn.init.xavier_uniform_(self.msg_fc2.weight)
        nn.init.xavier_uniform_(self.out_fc2.weight) 
        nn.init.xavier_uniform_(self.out_fc3.weight)

    def forward(self, inputs, edges, rel_rec, rel_send):
        # inputs: [batch, num_atoms, num_timesteps, num_dims]  
        # edges: [batch, num_edges, num_edge_types]
        
        # Flatten time and features
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        
        # Create layers dynamically  
        if self.msg_fc1 is None:
            msg_input_dim = 2 * x.size(-1)  # sender + receiver features
            self.msg_fc1 = nn.Linear(msg_input_dim, 256).to(x.device)
            nn.init.xavier_uniform_(self.msg_fc1.weight)
            
        if self.out_fc1 is None:
            out_input_dim = x.size(-1) + 256  # node features + messages
            self.out_fc1 = nn.Linear(out_input_dim, 256).to(x.device) 
            nn.init.xavier_uniform_(self.out_fc1.weight)
        
        # Get edge features from graph connectivity
        receivers = torch.matmul(rel_rec, x)  # [batch, num_edges, features]
        senders = torch.matmul(rel_send, x)   # [batch, num_edges, features]
        
        # Create messages - basic version, use all edge types equally  
        pre_msg = torch.cat([receivers, senders], dim=-1)  # [batch, edges, 2*features]
        
        # Simple message passing - process all edges the same way
        batch_size, num_edges = pre_msg.size(0), pre_msg.size(1)
        pre_msg_flat = pre_msg.view(-1, pre_msg.size(-1))
        
        msg = F.relu(self.msg_fc1(pre_msg_flat))
        msg = F.dropout(msg, self.dropout_prob, training=self.training)
        msg = F.relu(self.msg_fc2(msg))  # [batch*edges, n_hid]
        msg = msg.view(batch_size, num_edges, -1)  # [batch, edges, n_hid]
        
        # Aggregate messages to nodes
        agg_msg = torch.matmul(rel_rec.t(), msg)  # [batch, num_nodes, n_hid]
        
        # Combine with node features and predict  
        out = torch.cat([x, agg_msg], dim=-1)  # [batch, nodes, features+n_hid]
        out = out.view(-1, out.size(-1))
        
        out = F.relu(self.out_fc1(out))
        out = F.dropout(out, self.dropout_prob, training=self.training) 
        out = F.relu(self.out_fc2(out))
        out = self.out_fc3(out)  # [batch*nodes, 4] - predict full state
        
        out = out.view(batch_size, inputs.size(1), -1)  # [batch, nodes, 4]
        return out


class NRIEncoder(nn.Module):
    """Full NRI encoder that combines node and edge encoding."""
    
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(NRIEncoder, self).__init__()
        # Calculate flattened input dimension: n_in features * timesteps
        self.encoder = MLPEncoder(n_in, n_hid, n_out, do_prob)
        
    def forward(self, inputs, rel_rec, rel_send):
        return self.encoder(inputs, rel_rec, rel_send)


class NRIDecoder(nn.Module):
    """Full NRI decoder for dynamics prediction."""
    
    def __init__(self, n_in_node, n_in_edge, n_hid, do_prob=0.):
        super(NRIDecoder, self).__init__()
        self.decoder = MLPDecoder(n_in_node, n_in_edge, n_hid, do_prob)
        
    def forward(self, inputs, edges, rel_rec, rel_send):
        return self.decoder(inputs, edges, rel_rec, rel_send)


def create_relational_matrices(num_atoms):
    """Create matrices that specify which atoms send/receive messages."""
    num_edges = num_atoms * (num_atoms - 1)
    
    rel_rec = torch.zeros(num_edges, num_atoms)
    rel_send = torch.zeros(num_edges, num_atoms)
    
    edge_idx = 0
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                rel_rec[edge_idx, j] = 1
                rel_send[edge_idx, i] = 1
                edge_idx += 1
                
    return rel_rec, rel_send