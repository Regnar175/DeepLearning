import math
import inspect
import warnings
import torch
from torch import nn, optim
import torch.nn.functional as F
from colorama import Fore

# Suppress the specific FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

# This version of the GPT model implements past key/value state caching.
# However, generating with cached states produces poorer results in comparison to recalculating
# the full context window on each pass. Plus, the time difference (using CPU only for inference) is 
# actually slower using cached key/value states versues computing with the full generated ids.
# The code is provided for reference, but the trainer class will use the standard GPT model without cached key/value states.


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch doesn't support bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class GPTEmbedding(nn.Module):
    """Embedding class combines token and learned positional embeddings into one output"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.ntoken, config.dmodel, padding_idx=0)  # token embedding
        self.position = nn.Embedding(config.seqlen, config.dmodel)  # positional embedding
        self.norm = LayerNorm(config.dmodel, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inpt):
        pos = torch.arange(inpt.size(-1), dtype=torch.long, device=inpt.device)
        tokens = self.token(inpt)
        positions = self.position(pos)
        return self.dropout(self.norm(tokens + positions))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention class for an auto-regressive decoder."""
    def __init__(self, config):
        super().__init__()
        # Model's configuration parameters
        self.config = config 
        self.h_size = config.dmodel // config.nhead
        self.n_head = config.nhead
        self.d_model = config.dmodel
        self.seq_len = config.seqlen
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
        # Reshapped heads [batch_size, num_heads, seq_len, head_size]
        return query, key, value
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # Q [batch_size, num_heads, seq_len, head_size] @ K [batch_size, seq_len, num_heads, head_size] / sqrt(head_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.h_size))
        # attn_scores [batch_size, num_heads, seq_len, seq_len]
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
    
    def forward(self, embed, mask=None, past_states=None):
        # Perform linear projection in a batch and split heads
        query, key, value = self.split_heads(embed)
        # Allow for past key/value states to be reused
        if past_states is not None:
            past_key, past_value = past_states
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
            # Ensure the past states do not grow beyond the max sequence length
            if key.size(2) > self.seq_len:
                key = key[:, :, -self.seq_len:]  
                value = value[:, :, -self.seq_len:] 
        # Optimized attention using Flash Attention CUDA kernels
        if self.config.flash: # Only supported in linux, plus need to install triton package and compile the model
            context = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, 
                    dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        else:  # Manual implementation of attention
            context = self.scaled_dot_product_attention(query, key, value, mask)
        # Output projection: merge heads -> linear transformation -> dropout
        output = self.proj_drop(self.out_proj(self.merge_heads(context)))
        return output, (key, value) # Updated past key/value states


class DecoderBlock(nn.Module):
    """Auto-regressive decoder transformer block.
    Includes MLP (Feed Forward Network) projection layer."""
    def __init__(self, config):
        super().__init__()
        self.causal_attn = CausalSelfAttention(config)
        self.ffwd_net = nn.Sequential(
            # MLP dense layer
            nn.Linear(config.dmodel, 4 * config.dmodel, bias=config.bias),
            nn.GELU(), 
            nn.Dropout(config.dropout),
            # MLP projection layer
            nn.Linear(4 * config.dmodel, config.dmodel, bias=config.bias),
        )
        self.norm1 = LayerNorm(config.dmodel, config.bias)
        self.norm2 = LayerNorm(config.dmodel, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embed, mask=None, past_states=None):
        # Use past key/value states for attention if provided
        attn_out, new_past_states = self.causal_attn(embed, mask, past_states)
        # Residual connection 
        attn_out = self.norm1(embed + attn_out)
        # Fully connected (MLP)
        ffwd_out = self.dropout(self.ffwd_net(attn_out))
        ffwd_out = self.norm2(attn_out + ffwd_out)
        # Return the output and new key/value states
        return ffwd_out, new_past_states


class GPTModel(nn.Module):
    """
    GPT Model: Generative Pre-trained Transformer.\n
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
        flash: enable Flash Attention
    init_weights: bool = True | initialize model weights or bypass if loading weights
    """
    def __init__(self, config, init_weights=True):
        super().__init__()
        self.config = config
        self.embedding = GPTEmbedding(config)  # token and positional embedding
        self.decoder = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.nlayer)]
        )
        self.lm_head = nn.Linear(config.dmodel, config.ntoken, bias=False)
        # Set flag to use past key/value states during inference
        self.use_cache = False

        # Initialize weights and apply special scaled init to the attn out projection layer
        if init_weights:
            self.apply(self.init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('out_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.nlayer))

        # Tie token embedding weights and lm_head linear layer weights
        self.lm_head.weight = self.embedding.token.weight

        # Report number of parameters
        print(f"{Fore.MAGENTA}GPT Model Parameters:{Fore.RESET} %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_pos_embed=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_pos_embed:
            n_params -= self.embedding.position.weight.numel()
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
        """Configure AdamW optimizer with seperate weight decay and non-decay groups."""
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

    def forward(self, input_ids, target=None, mask=None, past_states=None):
        # Mask [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            
        embed = self.embedding(input_ids)

        decoded = embed
        for layer in self.decoder:
            decoded, new_past_states = layer(decoded, mask, past_states)
            # Send previous layer key/value states to the next layer
            if self.use_cache:
                past_states = new_past_states

        if target is not None:
            # Calculate the loss if targets are provided
            logits = self.lm_head(decoded)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=0)
        else:
            # Mini-optimization for generation - preserve the time dim for lm_head projection on last position
            logits = self.lm_head(decoded[:, [-1], :]) 
            loss = None

        return logits, loss, new_past_states

    def generate_with_cache(self, input_ids, max_length=1024, temp=1.0, top_k=None):
        """This method will generate the next token in sequence based on the intial forward pass
        with the full prompt context window that produces the past key/value cached states.
        The generation sequence is then performed using the last generated token along with the cached states
        from the previous generated output."""
        self.eval()
        # Prepare the input tensor
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        with torch.inference_mode():
            # Generate logits and past states based on the prompt
            logits, _, past_states = self.forward(input_ids, past_states=None)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            generated_ids = torch.multinomial(probs, 1)

            # Start the generation chain passing the next predicted token and the cached key/value states
            self.use_cache = True
            for _ in range(max_length - len(input_ids)): 
                # Do a forward pass with the running list of generated ids using the last one
                logits, _, past_states = self.forward(generated_ids[-1].unsqueeze(0), past_states=past_states)

                # Apply temperature scaling and optionally top_k filtering
                logits = logits[:, -1, :] / temp
                if top_k is not None:
                    values, indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, indices, values)

                # Sample from logits to get probabilities and next token id
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1) 
                generated_ids = torch.cat((generated_ids, next_id), dim=1)

                # Stop generating once a [SEP] token is encountered
                if next_id.item() == self.config.sep_id: 
                    break

        self.use_cache = False # Reset the flag
        return generated_ids.squeeze(0).tolist()
    
    def generate(self, input_ids, max_length=1024, temp=1.0, top_k=None, top_p=None):
        """
        This method will generate with the traditional approach using the full generated
        sequence for each forward pass. It uses the full context window to generate the next token in sequence.
        The lm_head output projection layer will only return the last generated token logits while in
        inference mode (no_grad) and no targets are passed. The generation loop will continually feed
        the generated output back into the model to obtain the next token in sequence. It will optionally
        apply temperature, get the top k logits, and/or get the top p logits before sampling from the filtered logits
        for the next token/word prediction. It will concatenate to the running list of generated tokens and stop
        once a [SEP] token ID has been produced or the max sequence length has been reached.
        Args:\n
            input_ids: list = Any | list of encoded token IDs from a text prompt
            max_length: int = 1,024 | max sequence length (context window) that the model can handle
            temp: float = 1.0 | apply temperature to the logit selection (e.g. lower than 1.0 reduces randomness)
            top_k: int = None | Get the top k logits to sample from (e.g. value of 100 will select the top 100 logits)
            top_p: float = None | Get the top p logits to sample from (e.g. select cumulative probs greater than 0.8 or 80%)
        Return: list of generated token ids to be decoded
        """
        self.eval()
        # Prepare the input tensor
        generated_ids = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        with torch.inference_mode():
            for _ in range(max_length - len(input_ids)): 
                # Do a forward pass with the running list of ids
                logits, _, _ = self.forward(generated_ids)
                logits = logits[:, -1, :] / temp  # Scale logits by temperature

                # Get the top k options from the logits output
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float('-inf')

                # Get the top p options from the logits output
                if top_p is not None:
                    sorted_logits, sorted_idxs = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Remove tokens with cumulative probability above the threshold
                    idxs_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep at least one token
                    idxs_to_remove[..., 1:] = idxs_to_remove[..., :-1].clone()
                    idxs_to_remove[..., 0] = 0
                    idxs = idxs_to_remove.scatter(1, sorted_idxs, idxs_to_remove)
                    logits[idxs] = float('-inf')

                # Sample from logits to get probabilities and next token id
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1) 
                generated_ids = torch.cat((generated_ids, next_id), dim=1)

                # Stop generating once a [SEP] token is encountered
                if next_id.item() == self.config.sep_id: 
                    break

        return generated_ids.squeeze(0).tolist()