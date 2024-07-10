import torch.nn as nn

class Self_Attention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        
    def forward(self, x):
        query = self.W_query(x)
        key = self.W_key(x)
        value = self.W_value(x)
        
        attention_score = query @ key.T
        attention_weight = torch.softmax(attention_score / k.shape[-1] ** 2, dim=-1)
        context_vec = attention_weight @ value
        return context_vec
    
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, no_tokens, d_in = x.shape
        keys = self.W_query(x)
        queries = self.W_key(x)
        values = self.value(x)
        
        attention_score = queries @ keys.transpose(1,2)
        attetion_score.masked_fill_(self.mask.bool()[:no_tokens, :no_tokens], -torch.inf)
        attention_weight = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)
        
        attention_weight = self.dropout(attention_weight)
        context_vec = attention_weight @ values
        return context_vec
    
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.5, qkv_bias=False)
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.head_dim = d_out / num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)