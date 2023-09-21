import copy
import math
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .que_base_model import QueBaseModel,QueEmb
from ..utils.utils import debug_print
torch.set_printoptions(precision=4, sci_mode=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype('bool') 
    return torch.from_numpy(future_mask)

def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings
    
    scores = torch.matmul(query, key.transpose(-2, -1))     # BS, head, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    
    idxs = torch.arange(scores.size(-1)).to(device)
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key.transpose(-2, -1))
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    pad_zero = torch.zeros(bs, head, 1, 1, seqlen).to(device)
    prob_attn = torch.cat([pad_zero, prob_attn[:, :, 1:, :, :]], dim=2)
    # print(prob_attn[2])
    # sys.exit()
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    
    return output, prob_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        out, self.prob_attn = relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        return out, self.prob_attn


class RAKT(nn.Module):
    def __init__(self, num_c, num_q, embed_size, num_attn_layers, num_heads, batch_size, q_matrix,
                  max_pos, grad_clip, beta=0.5, drop_prob=0.1, emb_type="qid", emb_path=""):
        """
        Arguments:
            num_q (int): number of questions
            num_c (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(RAKT, self).__init__()
        self.model_name = "rakt"
        self.emb_type = emb_type
        self.num_c = num_c
        self.num_q = num_q
        self.embed_size = embed_size
        self.drop_prob = drop_prob
        self.grad_clip = grad_clip
        self.beta = beta
        self.q_matrix = q_matrix
        
        
        if num_q <= 0:
            self.que_embeds = nn.Embedding(num_c + 1, embed_size , padding_idx=0)
        else:
            self.que_embeds = nn.Embedding(num_q + 1, embed_size , padding_idx=0)
        self.concept_embeds = nn.Parameter(torch.randn(num_c, embed_size).to(device), requires_grad=True)   #concept embeding

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.qc_linear = nn.Linear(2*embed_size, embed_size)
        self.lin_in = nn.Linear(2*embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        #self.lin_out = nn.Linear(embed_size, 1)
        self.activate = nn.Sigmoid()

        self.out = nn.Sequential(
            nn.Linear(embed_size + embed_size, embed_size), 
            nn.ReLU(), 
            nn.Dropout(self.drop_prob),
            nn.Linear(embed_size, embed_size), 
            nn.ReLU(), 
            nn.Dropout(self.drop_prob),
            nn.Linear(embed_size, 1)
        )

    def get_inter_emb(self, que_inputs, label_inputs):
        
        label_inputs = label_inputs.unsqueeze(-1).float()

        inter_emb = torch.cat([que_inputs, que_inputs], dim=-1)
        inter_emb[..., :self.embed_size] *= label_inputs  
        inter_emb[..., self.embed_size:] *= 1 - label_inputs  
        return inter_emb
    
    def get_avg_skill_emb(self,c):
        # add zero for padding
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.embed_size).to(device), self.concept_embeds], dim=0)
        # shift c
        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(axis=-2)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg
    
    def qc_relation(self, qemb, all_cemb):
        sim = torch.matmul(qemb, all_cemb.t())
        return sim

    def forward(self, dcur, rel_dict, train=True):
        q, c, r, t = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["tseqs"]  
        qshft, cshft, rshft, tshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_tseqs"]
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        c_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)
        timestamp = torch.cat((t[:,0:1], tshft), dim=1)
        #print(f"times:{t}, \n timestamp:{timestamp} \n {t.shape}")
        
        # filter the dataset only with question, no concept
        if self.num_q <= 0:
            inputs = c_data
        else:
            inputs = pid_data

        qemb = self.que_embeds(inputs)
        cemb = self.get_avg_skill_emb(c_data)
        emb_qc = torch.cat([qemb, cemb], dim=-1)
        query = self.qc_linear(emb_qc)

        inter_emb = self.get_inter_emb(query, target)
        inter_emb = F.relu(self.lin_in(inter_emb))

        # calculate qc similarity
        sim = self.qc_relation(qemb, self.concept_embeds)
        sim = self.activate(sim)
        #batch_size, seq_len = inputs.shape[0], inputs.shape[1]
        #time = computeTime(timestamp, self.time_span, batch_size, seq_len) 
        
        mask = future_mask(inter_emb.size(-2)).to(device)
        outputs, attn  = self.attn_layers[0](query, query, inter_emb, self.pos_key_embeds, self.pos_value_embeds, mask)
        outputs = self.dropout(outputs)
        
        for l in self.attn_layers[1:]:
            residual, attn = l(outputs, outputs, inter_emb, self.pos_key_embeds,self.pos_value_embeds, mask)
            outputs = self.dropout(outputs + F.relu(residual))
        
        concat_q = torch.cat([outputs, query], dim=-1)
        out = self.out(concat_q).squeeze(-1)
        pred = self.activate(out)
        
        return pred, sim
