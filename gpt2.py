import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def forward(self, x):
        # x : batch_size, seq_len, emb_dim
        size_out = x.size()[:-1] + (self.nf,)
        # b * self.bias + a *(torch.matmul(x, weight)) where b and a are hyperparametrs(default = 1)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

    
class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim = config.emb_dim
        hidden_size = config.mlp_hidden_size
        self.c_fc = Conv1D(emb_dim, hidden_size)
        self.c_proj = Conv1D(hidden_size, emb_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=config.mlp_dropout_p)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        
        return x

    
# TODO продумать случай когда может быть другая последовательность query
class GPT2Attention_classic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.emb_dim
        self.num_heads = config.attn_num_heads
        
        assert (self.embed_size % self.num_heads) == 0, 'embeddings size must be devisible by num_heads'
        #input num_samples, seq_len, embd_size
        self.head_size = self.embed_size // self.num_heads
        # num_samples
        self.Wq = nn.Linear(self.head_size, self.head_size)
        self.Wk = nn.Linear(self.head_size, self.head_size)
        self.Wv = nn.Linear(self.head_size, self.head_size)
        self.с_proj = nn.Linear(self.embed_size, self.embed_size)
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)
        self.resid_dropout = nn.Dropout(p=config.attn_resid_dropout)

    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        query = self.Wq(x).transpose(1, 2)
        key = self.Wk(x).transpose(1, 2)
        value = self.Wv(x).transpose(1, 2)
        
        product = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        
        mask= torch.tril(torch.ones(1,1, seq_len, seq_len)).to(x.device)
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))
        
        output = F.softmax(product, dim=-1)
        output = self.attn_dropout(output)
        
        output = torch.matmul(output, value) 
        output = output.transpose(1,2).contiguous().view(
            batch_size,
            seq_len,
            self.head_size * self.num_heads
        )
        output = self.out(output) 
        output = self.resid_dropout(output)
        return output


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.emb_dim
        self.num_heads = config.attn_num_heads
        
        assert (self.embed_size % self.num_heads) == 0, 'embeddings size must be devisible by num_heads'
        
        self.head_size = self.embed_size // self.num_heads
        self.c_attn = Conv1D(config.emb_dim, config.emb_dim * 3)
        self.dropout = nn.Dropout(config.attn_dropout)
        self.c_proj = Conv1D(config.emb_dim, config.emb_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def split_heads(self, x):
        # reshape to batch_size, num_heads, seq_len, emb_dim
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)
    
    def _attn(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(v.size(-1))
        
        if attn_mask is not None:
            # TODO is it correct to sum it?
            scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs
    
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)
    
    def forward(self, x):
        x = self.c_attn(x)
        q, k, v = x.split(self.embed_size, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out = self._attn(q, k, v)
        out = self.merge_heads(out)
        out = self.c_proj(out)
        return out


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_dim, eps=config.ln_eps, elementwise_affine=True)
        self.ln_2 = nn.LayerNorm(config.emb_dim, eps=config.ln_eps, elementwise_affine=True)
        self.attn = GPT2Attention(config)
        self.mlp = GPT2MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x
    
    
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.emb_dim)
        self.wpe = nn.Embedding(config.context_window, config.emb_dim)
        self.dropout = nn.Dropout(p=config.gpt2_dropout)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_blocks)])
        self.ln_f = nn.LayerNorm(config.emb_dim, eps=config.ln_eps, elementwise_affine=True)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
                 
    def forward(self, input_ids):
        embeddings = self.wte(input_ids) 
        pos_embeddings = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.wpe(pos_embeddings)
        input_ids = self.dropout(embeddings + positional_embeddings)
        
        for i in range(self.config.num_blocks):
            input_ids = self.h[i](input_ids)
        
        input_ids = self.ln_f(input_ids)
        logits = self.lm_head(input_ids)
        
        return logits