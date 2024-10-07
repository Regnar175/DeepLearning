import math
import inspect
import warnings
import torch
from torch import nn, optim
import torch.nn.functional as F
from colorama import Fore

# Suppress the specific FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


class LayerNorm(nn.Module):
    """Custom LayerNorm with an optional bias. PyTorch doesn't support bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class BARTEmbedding(nn.Module):
    """Embedding class combines token and learned positional embeddings into one output.
    It will offset positional embeddings by accounting for the shifted right decoder sequence."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.ntoken, config.dmodel, padding_idx=0)
        self.pos_embed = nn.Embedding(config.seqlen + 2, config.dmodel) # Offset positions by 2
        self.norm = LayerNorm(config.dmodel, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        # Token Embeddings
        embed = self.token_embed(input_ids)
        # Positional Embeddings
        pos = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        positions = self.pos_embed(pos + 2) # Offset positions by 2
        # Combine token and positional embeddings
        return self.dropout(self.norm(embed + positions))

    
class SelfAttention(nn.Module):
    """Modular multi-head attention class for bi-drectional or cross-attention encoder."""
    def __init__(self, config):
        super().__init__()
        # Model's configuration parameters
        self.config = config 
        self.h_size = config.dmodel // config.nhead
        self.n_head = config.nhead
        self.d_model = config.dmodel
        # Query, key, value and output projection layers
        self.q_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Query projection
        self.k_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Key projection
        self.v_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Value projection
        self.out_proj = nn.Linear(config.shead * config.nhead, config.dmodel, bias=config.bias) # Output projection
        # Dropout for attention weights and output projection
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)

    def split_heads(self, embed, cross=None):
        B, T, _ = embed.size() # [batch_size, seq_len, d_model]
        # Linear projections for query, key, and value
        query  = self.q_proj(embed)
        if cross is not None:
            key = self.k_proj(cross)
            value = self.v_proj(cross)
        else:
            key = self.k_proj(embed)
            value = self.v_proj(embed)
        # Check the total size matches before reshaping
        assert query.size(-1) == self.d_model, "Query dimension mismatch."
        assert key.size(-1) == self.d_model, "Key dimension mismatch."
        assert value.size(-1) == self.d_model, "Value dimension mismatch."
        # Reshape linear projections
        query = query.view(B, T, self.n_head, self.h_size).transpose(1, 2) 
        key = key.view(B, -1, self.n_head, self.h_size).transpose(1, 2) 
        value = value.view(B, -1, self.n_head, self.h_size).transpose(1, 2) 
        # Reshaped attention heads [batch_size, num_heads, seq_len, head_size]
        return query, key, value

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # Q [batch_size, num_heads, seq_len, head_size] @ K [batch_size, seq_len, num_heads, head_size] / sqrt(head_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.h_size))
        # attn_scores [batch_size, num_heads, seq_len, seq_len]
        # Apply mask if provided (to bring attention to non-padded tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        # Softmax is applied to obtain attention probabilities
        weights = torch.softmax(attn_scores, dim=-1)
        weights = self.attn_drop(weights)
        # Multiply weights by values to obtain the final output
        # Weights [batch_size, num_heads, seq_len, seq_len] @ V [batch_size, seq_len, num_heads, head_size]
        context = torch.matmul(weights, V)
        return context
        
    def merge_heads(self, context):
        B, _, T, _ = context.size() # [batch_size, num_heads, seq_len, head_size]
        # Reshaped to -> [batch_size, seq_len, d_model]
        return context.transpose(1, 2).contiguous().view(B, T, self.d_model)
    
    def forward(self, embed, cross=None, mask=None):
        # Perform linear projections and split heads
        query, key, value = self.split_heads(embed, cross)
        # Perform scaled dot-product attention
        context = self.scaled_dot_product_attention(query, key, value, mask)
        # Output projection: merge heads -> linear transformation -> dropout
        output = self.proj_drop(self.out_proj(self.merge_heads(context)))
        return output
    

class CausalAttention(nn.Module):
    """Multi-head causal self-attention class for an auto-regressive decoder."""
    def __init__(self, config):
        super().__init__()
        # Model's configuration parameters
        self.config = config 
        self.h_size = config.dmodel // config.nhead
        self.n_head = config.nhead
        self.d_model = config.dmodel
        # Query, key, value and output projection layers
        self.c_proj = nn.Linear(config.dmodel, 3 * config.dmodel, bias=config.bias) # Q, K, V linear projection
        self.out_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Output linear projection
        # Dropout for attention weights and output projection
        self.attn_drop = nn.Dropout(config.dropout)
        self.proj_drop = nn.Dropout(config.dropout)
        # Causal mask to ensure attention is only applied to the left in the input sequence
        self.register_buffer("causal", torch.tril(torch.ones(config.seqlen, config.seqlen))
                                .view(1, 1, config.seqlen, config.seqlen))
        
    def split_heads(self, embed):
        B, T, C = embed.size() # [batch_size, seq_len, d_model]
        # Calculate query, key, values for all heads in batch
        query, key, value  = self.c_proj(embed).split(self.d_model, dim=2)
        # Transpose seq_len and num_heads dim
        query = query.view(B, T, self.n_head, self.h_size).transpose(1, 2) 
        key = key.view(B, T, self.n_head, self.h_size).transpose(1, 2) 
        value = value.view(B, T, self.n_head, self.h_size).transpose(1, 2) 
        # Reshapped attn heads [batch_size, num_heads, seq_len, head_size]
        return query, key, value
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # Q [batch_size, num_heads, seq_len, head_size] @ K [batch_size, seq_len, num_heads, head_size] / sqrt(head_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.h_size))
        # Apply causal attention mask (square subsequent mask or 'no peak' mask)
        attn_scores = attn_scores.masked_fill(self.causal[:,:,:Q.size(2),:K.size(2)] == 0, float('-inf'))
        # Apply attention mask if provided (to bring attention to non-padded tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        # Softmax is applied to obtain attention probabilities
        weights = F.softmax(attn_scores, dim=-1)
        weights = self.attn_drop(weights)
        # Multiply weights by values to obtain the final attention output
        # Weights [batch_size, num_heads, seq_len, seq_len] @ V [batch_size, seq_len, num_heads, head_size]
        context = torch.matmul(weights, V)
        return context
    
    def merge_heads(self, context):
        B, _, T, _ = context.size() # [batch_size, num_heads, seq_len, head_size]
        # Reshaped to -> [batch_size, seq_len, d_model]
        return context.transpose(1, 2).contiguous().view(B, T, self.d_model)
    
    def forward(self, embed, mask=None): 
        # Perform linear projection in a batch and split heads
        query, key, value = self.split_heads(embed)
        # Perform scaled dot-product attention
        context = self.scaled_dot_product_attention(query, key, value, mask)
        # Output projection: merge heads -> linear transformation -> dropout
        output = self.proj_drop(self.out_proj(self.merge_heads(context)))
        return output 
    

class EncoderBlock(nn.Module):
    """Bi-drectional auto-encoder attention layer including 
    the MLP (Feed Forward Network) projection layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttention(config)
        self.ffwd_net = nn.Sequential(
            nn.Linear(config.dmodel, 4 * config.dmodel, bias=config.bias),
            nn.GELU(), 
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.dmodel, config.dmodel, bias=config.bias),
        )
        self.norm1 = LayerNorm(config.dmodel, config.bias)
        self.norm2 = LayerNorm(config.dmodel, config.bias) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embed, cross=None, mask=None):
        attn_out = self.self_attn(embed, cross, mask)
        # Residual connection
        attn_out = self.norm1(embed + attn_out)
        # Fully connected (MLP)
        ffwd_out = self.dropout(self.ffwd_net(attn_out))
        return self.norm2(attn_out + ffwd_out)


class DecoderBlock(nn.Module):
    """Decoder transformer block with auto-regressive
    causal self-attention and bi-directional cross attention layers.
    Includes MLP (Feed Forward Network) projection layer."""
    def __init__(self, config):
        super().__init__()
        self.causal_attn = CausalAttention(config)
        self.cross_attn = SelfAttention(config)
        self.ffwd_net = nn.Sequential(
            nn.Linear(config.dmodel, 4 * config.dmodel, bias=config.bias),
            nn.GELU(), 
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.dmodel, config.dmodel, bias=config.bias),
        )
        self.norm1 = LayerNorm(config.dmodel, config.bias)
        self.norm2 = LayerNorm(config.dmodel, config.bias)
        self.norm3 = LayerNorm(config.dmodel, config.bias) 
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embed, cross=None, mask=None):  
        attn_out = self.causal_attn(embed, mask)
        # Causal attention residual connection
        attn_out = self.norm1(embed + attn_out)
        # Apply cross attention between causal output and encoder output
        cross_out = self.cross_attn(embed, cross, mask=mask) # Test with or without mask
        # Cross attention residual connection
        attn_out = self.norm2(attn_out + cross_out)
        # Multi-layer perceptron (MLP) projection
        ffwd_out = self.dropout(self.ffwd_net(attn_out))
        # Fully connected
        output = self.norm3(attn_out + ffwd_out)
        return output


class BARTModel(nn.Module):
    """
    BART Model: Bidirectional Auto Regressive Transformers.\n
    Params (Config):\n 
        device: 'cuda' if torch.cuda.is_available() else 'cpu'
        ntoken: Size of tokenizer vocabulary
        seqlen: Length of token sequence per mini-batch
        dmodel: Model dimension size (embedding dimension)
        nhead: Number of attention heads (parallel)
        nlayer: Number of transformer blocks (sequential layers)
        shead: dmodel // nhead - Individual head size
        dropout: dropout rate during training
        bias: Bias True/False in linear/normalization layers
    """
    def __init__(self, config, init_weights=True):
        super().__init__()
        self.config = config
        self.embedding = BARTEmbedding(config) # Token and positional embeddings
        self.encoder = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.nlayer)]
        )
        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.nlayer)]
        )
        self.lm_head = nn.Linear(config.dmodel, config.ntoken, bias=False)

        if init_weights:
            self.apply(self.init_weights)
            # Apply special scaled init to the attention context projection output
            for pn, p in self.named_parameters():
                if pn.endswith('out_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.nlayer))

        # Report number of parameters
        print(f"{Fore.MAGENTA}BART Model Parameters:{Fore.RESET} %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_pos_embed=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_pos_embed:
            n_params -= self.embedding.pos_embed.weight.numel()
        return n_params

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, config, init_weights=False):
        model = cls(config, init_weights)
        model_dict = torch.load(config.load_path)
        model.load_state_dict(model_dict['model_state_dict'])
        return model.to(config.device)
    
    def config_optimizer(self, lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
        """Configure AdamW optimizer with separate weight decay and non-decay groups."""
        # Organize parameters into groups with or without weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            # Skip parameters that do not require gradients
            if param.requires_grad:
                # Exclude params less than 2D from weight decay
                # e.g. biases and LayerNorm
                if param.dim() < 2:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        # Prepare parameter groups for optimizer with specific settings for each group
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': lr},
        ]
        # Check if fused version of AdamW is available
        fused_available = 'fused' in inspect.signature(optim.AdamW).parameters
        use_fused = fused_available and self.config.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        return optim.AdamW(optim_groups, betas=betas, eps=eps, **extra_args)

    def shift_tokens_right(self, input_ids, start_token_id=1):
        """Shift input ids one token to the right to add the [BOS] token."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id

        return shifted_input_ids

    def forward(self, input_ids, target_ids=None, mask=None): 
        # Adds an additional [BOS] token (1) to the first position
        # Example: [1, 2345, 4567, ... 9876, 289, 0] -> [1, 1, 2345, ... 5432, 9876, 289]
        if target_ids is None:
            shifted_ids = self.shift_tokens_right(input_ids) 
        else:
            shifted_ids = self.shift_tokens_right(target_ids) 

        # Encoder/decoder Embeddings
        enc_embed = self.embedding(input_ids)
        dec_embed = self.embedding(shifted_ids)

        # Mask [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if mask is None:
            enc_mask = None
            dec_mask = None
        else:
            enc_mask = mask[0].unsqueeze(1).unsqueeze(2)
            dec_mask = mask[1].unsqueeze(1).unsqueeze(2)

        # Encoder forward pass
        encoded = enc_embed  
        for layer in self.encoder:
            encoded = layer(encoded, cross=None, mask=enc_mask)
            
        # Decoder forward pass
        decoded = dec_embed
        for layer in self.decoder:
            decoded = layer(decoded, cross=encoded, mask=dec_mask)

        # Language modeling head output
        return self.lm_head(decoded)
    
    def generate(self, input_ids, max_length=1024, temp=1.0, top_k=None):
        """Generate function to produce an auto-regressive output from the model."""
        # Convert input ids to tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        # Initialize the generated_ids with the [BOS] token
        generated_ids = torch.tensor([[self.config.bos_id]], dtype=torch.long).to(self.config.device) 

        self.eval()
        with torch.inference_mode():
            # Convert input_ids to embeddings for encoder
            enc_embed = self.embedding(input_ids)
            # Encoder forward pass to get encoder hidden states
            encoded = enc_embed  
            for layer in self.encoder:
                encoded = layer(encoded, cross=None, mask=None)
            # Generation loop
            for _ in range(max_length):
                # Convert current generated_ids to embeddings for decoder
                dec_embed = self.embedding(generated_ids)

                # Decoder forward pass with the encoder's output
                decoded = dec_embed
                for layer in self.decoder:
                    decoded = layer(decoded, cross=encoded, mask=None) 

                # Apply language modeling head
                logits = self.lm_head(decoded)[:, -1, :] / temp
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float('-inf')
                
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=-1)
                # Sample the next token
                next_token = torch.multinomial(probs, 1)
                # Append the sampled token to the generated_ids
                generated_ids = torch.cat((generated_ids, next_token), dim=1)
                
                # Stop generation if [EOS] token is encountered
                if next_token.item() == self.config.eos_id:
                    break

            return generated_ids.squeeze(0).tolist()

    def inference(self, input_ids):
        """Simple inference function to obtain greedy decoded output."""
        self.eval()
        # Prepare input tensor
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        with torch.inference_mode():
            logits = self.forward(input_tensor)
            predictions = F.softmax(logits, dim=-1).argmax(dim=-1)

        # Convert the tensor back to a list of token IDs
        return predictions.squeeze(0).tolist()