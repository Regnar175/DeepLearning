import re
import math
import inspect
import warnings
import torch
from torch import nn, optim
import torch.nn.functional as F
from tokenizers import Tokenizer
from nltk.tokenize import sent_tokenize
import numpy as np
from colorama import Fore

# Suppress the specific FutureWarning related to torch.load
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


ENTITIES = [
    # Can possibly be lower case words
    # States/Provinces/Cities
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks',
    'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny', 'nc', 
    'nd', 'oh', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy',
    'dc', 'nyc', 'pr', 'sf',
    # Organizations/Countries
    'doe', 'ice', 'it', 'swat', 'un', 'us', 'who',
    # Misc
    'ad', 'am', 'da', 'dew', 'er', 'mia', 'vin', 
]

STOPWORDS = ['a', 'an', 'and', 'at', 'be', 'been', 'by', 'for', 'from', 'in', 'into', 'is', 'he', 'her', 'hers', 'him', 'his', 
             'of', 'on', 'or', 'our', 'out', 'over', 'she', 'the' 'to']

ACRONYMS = [
    # Not confused with lower case words
    '1d', '2d', '3d', '4d', '5d', '1g', '2g', '4g', '5g', '6g', 'aba', 'abc', 'ac', 'acp', 'aclu', 'adhd', 'adl', 'adm', 'af', 'afb', 'afc' ,
    'ag', 'agi', 'ai', 'aids', 'aipac', 'ak', 'ama', 'amc', 'amd', 'aoc', 'apec', 'api', 'ar', 'asa', 'asap', 'ascii', 'atf', 'atm', 'att', 'at&t', 'atv', 'au', 
    'awol', 'aws', 'ba', 'bc', 'bce', 'bbc', 'bbq', 'bipoc', 'bj', 'blm', 'bp', 'brics', 'bs', 'btc', 'cbdc', 'cbs', 'cc', 'cd', 'cdc', 'cdr', 'cdt', 'cern',
    'ce', 'ceo', 'cfo', 'cfr', 'ch', 'ccp', 'cct', 'cctv', 'cia', 'cic', 'cis', 'cgi', 'cnbc', 'cnn', 'co2', 'col', 'cp', 'cpac', 'cpl', 'cpr', 'cps', 'cpt', 'cpu', 
    'csi', 'cst', 'ct', 'cte', 'ctrl', 'cvs', 'cya', 'darpa', 'dc', 'dea', 'dei', 'dhs', 'dj', 'dl', 'dm', 'dmt', 'dmv', 'dna', 'dnc', 'dni', 'dod', 'doi', 
    'doj', 'ds', 'dui', 'dvd', 'eas', 'ebs', 'ebt', 'ec', 'efx', 'emf', 'emp', 'emt', 'eo', 'epa', 'esd', 'esg', 'esp', 'espn', 'est', 'et', 'eta', 'etf', 'eu', 'ev', 
    'f1', 'fa', 'faa', 'fbi', 'fcc', 'fda', 'fema', 'fi', 'fisa', 'flir', 'foia', 'fn', 'fng', 'fps', 'ft', 'ftm', 'fx', 'gb', 'gbs', 'gitmo', 'gdp', 'gen', 'gmo', 'gmt', 
    'gop', 'gp', 'gps', 'gpu', 'gpt', 'gui', 'h2o', 'haarp', 'hcq', 'hd', 'hfx', 'hgv', 'hippa', 'hiv', 'hmo', 'hr', 'ic', 'icbm', 'icg', 'icrc', 'icu', 'id', 'idf',
    'ied', 'imf', 'ios', 'ip', 'ipo', 'iq', 'ir', 'ira', 'irs', 'irgc', 'isis', 'isp', 'isr', 'iss', 'ivf', 'jc', 'jd', 'jpg' 'jsoc', 'jsotf', 'jtf', 'kfc', 
    'kgb', 'kjv', 'kpi', 'lan', 'larp', 'lcd', 'lgb', 'lgbt', 'lgbtq', 'lgbtqia', 'lgbtqia+', 'llc', 'llm', 'lsd', 'lsi', 'lt', 'ltc', 'ltg', 'maga', 'maj', 'mba',
    'mc', 'mg', 'mh', 'mi5', 'mi6', 'mic', 'mit', 'mj', 'mk', 'ml', 'mlb', 'mlk', 'mma', 'mmr', 'mos', 'mp', 'mri', 'mrna', 'msnbc', 'mst', 'mtf', 'mtg', 'nas', 'nasa', 
    'nato', 'nba', 'nbc', 'nda', 'ndaa', 'nde', 'nfc', 'nfl', 'nft', 'ngo', 'nhi', 'nhl', 'nhs', 'nih', 'npc', 'npr', 'nra', 'nsa', 'nwo', 'nypd', 'nyt', 'o2', 
    'oan', 'obe', 'ocd', 'od', 'og', 'op', 'opec', 'opsec', 'os', 'oss', 'p2p', 'pac', 'pbs', 'pc', 'pcr', 'pd', 'pdf', 'peta', 'pfc', 'pg', 'pga', 'phd', 'pm', 
    'pms', 'poc', 'potus', 'pow', 'ppe', 'ppi', 'ppo', 'pr', 'ps', 'psa', 'psi', 'psv', 'psyop', 'pto', 'ptsd', 'pvt', 'q', 'q1', 'q2', 'q3', 'q4', 'qb', 'qc', 
    'qcd', 'qed', 'qfd', 'qfs', 'r&d', 'raf', 'rc', 'rfp', 'rico', 'rino', 'r&d', 'rnd', 'roi', 'rn', 'rna', 'rnc', 'rng', 'roe', 'rpg', 'rt', 'rv', 'secdef', 'sa', 
    'sci', 'scif', 'scifi', 'scotus', 'sd', 'sec', 'sf', 'sfc', 'sfx', 'sgt', 'snl', 'sp', 'spc', 'socom', 'sof', 'sop', 'sos', 'sotf', 'sra', 'ss', 'ssg', 'ssn', 'ssp', 
    'ssri', 'std', 'sts', 'suv', 'tb', 'tf', 'ts', 'tsa', 'tv', 'uae', 'uap', 'uas', 'ubi', 'ufo', 'uk', 'url', 'usa', 'usaf', 'usaid', 'usb', 'usd', 'usda', 'usgs', 
    'usmc', 'usn', 'uss', 'ussf', 'ussr', 'uv', 'vfx', 'vip', 'vhs', 'voa', 'vp', 'vpn', 'vr', 'wef', 'wsj', 'wmd', 'ww1', 'ww2', 'ww3', 'wwe', 'wwf', 
    'wi', 'wwi', 'wwii', 'wwiii', 'wifi', 'yt',
    # Roman numerals (process i seperately)
    'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'
]

ACRONYM_REGEX = re.compile(r'(?i)\b(' + '|'.join(map(re.escape, ACRONYMS)) + r")('?s)?\b")

def acronym_mapper(text):
    # Helper function to upper case acronyms if an 's' is attached
    def replace(match):
        # Upper case the acronym
        acronym = match.group(1).upper() 
        # Keep the 's or s as is
        postfix = match.group(2) if match.group(2) else '' 
        return acronym + postfix

    # Compiled regex pattern joining all acronyms in the list, checking if there is a plural or possesive 's'
    return ACRONYM_REGEX.sub(replace, text)


class LayerNorm(nn.Module):
    """Custom LayerNorm with an optional bias. PyTorch doesn't support bias=False"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class BERTEmbedding(nn.Module):
   """Embedding class combines learned token, positional and segment embeddings into one output."""
   def __init__(self, config):
       super().__init__()
       self.token = nn.Embedding(config.ntoken, config.dmodel, padding_idx=0)  # token embedding
       self.position = nn.Embedding(config.seqlen, config.dmodel, padding_idx=0)  # positional embedding
       self.segment = nn.Embedding(3, config.dmodel, padding_idx=0)  # segment(token type) embedding
       self.norm = LayerNorm(config.dmodel, config.bias)
       self.dropout = nn.Dropout(config.dropout)

   def forward(self, input_ids, segments):
       positions = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
       tok_embed = self.token(input_ids)
       pos_embed = self.position(positions)
       seg_embed = self.segment(segments)
       return self.dropout(self.norm(tok_embed + pos_embed + seg_embed))


class MultiHeadAttention(nn.Module):
    """Multi-head attention class for bi-drectional auto-encoder."""
    def __init__(self, config):
        super().__init__()
        self.config = config # Model's configuration parameters
        self.h_size = config.dmodel // config.nhead
        self.n_head = config.nhead
        self.d_model = config.dmodel
        # Query, key, value and output projection layers
        self.q_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Query projection
        self.k_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Key projection
        self.v_proj = nn.Linear(config.dmodel, config.dmodel, bias=config.bias) # Value projection
        self.out_proj = nn.Linear(config.shead * config.nhead, config.dmodel, bias=config.bias) # Output projection
        # Dropout for attention weigths and output projection
        self.attn_drop = nn.Dropout(config.dropout)  
        self.proj_drop = nn.Dropout(config.dropout) 
      
    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        # Reshape and split heads
        x = x.view(batch_size, seq_len, self.n_head, self.h_size)
        return x.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_size]
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        # Q [batch_size, num_heads, seq_len, head_size] @ K [batch_size, seq_len, num_heads, head_size] / sqrt(head_size)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.h_size))
        # attn_scores [batch_size, num_heads, seq_len, seq_len]
        # Apply mask if provided (bring attention to non-padded tokens)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        # Softmax is applied to obtain attention probabilities
        weights = torch.softmax(attn_scores, dim=-1)
        weights = self.attn_drop(weights)
        # Multiply weights by values to obtain the final attention output
        # Weights [batch_size, num_heads, seq_len, seq_len] @ V [batch_size, seq_len, num_heads, head_size]
        context = torch.matmul(weights, V)
        return context
        
    def merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        # Combine heads back to original tensor shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, -1)  # [batch_size, seq_len, d_model]
    
    def forward(self, embed, mask=None):
        # Apply linear projections and split heads
        Q = self.split_heads(self.q_proj(embed))
        K = self.split_heads(self.k_proj(embed))
        V = self.split_heads(self.v_proj(embed))
        # Perform scaled dot-product attention
        context = self.scaled_dot_product_attention(Q, K, V, mask)
        # Output projection: merge heads -> linear transformation -> dropout
        output = self.proj_drop(self.out_proj(self.merge_heads(context)))
        return output


class EncoderBlock(nn.Module):
    """Bi-drectional auto-encoder attention layer including 
    the MLP (Feed Forward Network) projection layer."""
    def __init__(self, config):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.ffwd_net = nn.Sequential(
            nn.Linear(config.dmodel, 4 * config.dmodel, bias=config.bias),
            nn.GELU(), 
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.dmodel, config.dmodel, bias=config.bias),
        )
        self.norm1 = LayerNorm(config.dmodel, config.bias)
        self.norm2 = LayerNorm(config.dmodel, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, embed, mask=None):
        # Embeddings: [batch_size, seq_len, d_model]
        # Attn_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        attn_out = self.self_attn(embed, mask)
        # Residual connection
        attn_out = self.norm1(embed + attn_out)
        # Fully connected (MLP)
        ffwd_out = self.dropout(self.ffwd_net(attn_out))
        return self.norm2(attn_out + ffwd_out)


class BERTModel(nn.Module):
    """
    BERT Model: Bi-directional Encoder Representations from Transformers.\n
    Params (Config):\n 
        ntoken: Size of tokenizer vocabulary
        seqlen: Length of token sequence per mini-batch
        dmodel: Model dimension size (embedding dimension)
        nhead: Number of attention heads (parallel)
        nlayer: Number of transformer blocks (sequential layers)
        shead: dmodel // nhead - Individual head size
        dropout: dropout rate
        bias: Bias True/False in linear/normalization layers
    """
    def __init__(self, config, init_weights=True):
        super().__init__()
        self.config = config
        self.embedding = BERTEmbedding(config)
        self.encoder = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.nlayer)]
        )
        self.mlm_head = nn.Linear(config.dmodel, config.ntoken, bias=False) # Token prediction
        self.nsp_head = nn.Linear(config.dmodel, 2, bias=False) # Is next classification

        # Initialize weights and apply special scaled init to the attn projection layer
        if init_weights:
            self.init_weights()
            for pn, p in self.named_parameters():
                if pn.endswith('out_proj.weight'):
                    nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.nlayer))
        # Tie token embedding weights and lm_head linear layer weights
        self.mlm_head.weight = self.embedding.token.weight

        # Report number of parameters
        print(f"{Fore.MAGENTA}BERT Model Parameters:{Fore.RESET} %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_pos_embed=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_pos_embed:
            n_params -= self.embedding.position.weight.numel()
            n_params -= self.embedding.segment.weight.numel()
        return n_params

    def init_weights(self):
        """
        Initialize weights of the model for all linear layers and embeddings
        using the Glorot (Xavier) uniform initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    @classmethod
    def from_pretrained(cls, config, strict=True):
        model = cls(config, init_weights=False)
        model_dict = torch.load(config.load_path)
        model.load_state_dict(model_dict['model_state_dict'], strict=strict)
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

    def forward(self, input_ids, segments=None, attn_mask=None, hidden=False):
        if segments is None:
            segments = torch.zeros_like(input_ids, dtype=torch.long, device=input_ids.device)
        if attn_mask is not None:
            # Mask [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        embed = self.embedding(input_ids, segments)
        # Iterate through encoder layers manually, passing attn_mask
        encoded = embed
        for layer in self.encoder:
            encoded = layer(encoded, attn_mask)

        # Output hidden states if adding an additional head on top
        if hidden: 
            return encoded

        # Get logits from masked language model head
        mlm = self.mlm_head(encoded)
        # Get the first token id logit for the NSP classifier
        nsp = self.nsp_head(encoded[:, 0])

        return mlm, nsp
    
    def fill_in(self, input_ids, mask_positions):
        self.eval()
        # Ensure input_ids is a tensor and on the correct device
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.config.device)
        
        with torch.inference_mode():
            # Predict logits for the entire sequence
            logits, _ = self.forward(input_tensor) 

            # Get the logits at each masked position and find the most likely token id
            for idx in mask_positions:
                mask_logit = logits[:, idx, :]  # Get logits for the current mask position
                next_id = torch.argmax(mask_logit, dim=-1)  # Get the predicted token ID for this position
                # Replace the masked token in the input tensor
                input_tensor[:, idx] = next_id

        # Convert the tensor back to a list of token IDs
        return input_tensor.squeeze(0).tolist()
    
############################ BERT Classifier ###########################

class BERTClassifier(BERTModel):
    """Extended BERT model for NER and Sentiment tasks."""
    def __init__(self, config):
        super(BERTClassifier, self).__init__(config, init_weights=False)
        self.config = config
        # NER head: Applies to all tokens
        self.ner_head = nn.Linear(config.dmodel, config.nlabels, bias=True)
        # Sentiment head: Applies to the [CLS] token
        self.sent_head = nn.Linear(config.dmodel, 3, bias=True)

        self.init_weights()

    def init_weights(self):
        """Initialize weights for the additional layers."""
        nn.init.xavier_uniform_(self.ner_head.weight)
        nn.init.xavier_uniform_(self.sent_head.weight)
        if self.ner_head.bias is not None:
            nn.init.constant_(self.ner_head.bias, 0)
        if self.sent_head.bias is not None:
            nn.init.constant_(self.sent_head.bias, 0)

    @classmethod
    def from_pretrained(cls, config):
        """Create an instance of BERTClassifier and load pre-trained weights ignoring non-matching keys."""
        model = cls(config)
        # Load pre-trained model state
        pt_dict = torch.load(config.load_path)
        model_dict = model.state_dict()
        # Filter out unnecessary keys from pre-trained state
        pretrained_dict = {k: v for k, v in pt_dict['model_state_dict'].items() 
                           if k in model_dict and model_dict[k].size() == v.size()}
        # Update current model's dict with pre-trained weights
        model_dict.update(pretrained_dict)  
        model.load_state_dict(model_dict)
        
        return model.to(config.device)
    
    @classmethod
    def load_weights(cls, config, strict=True):
        """Load weights from a fine-tuned BERTClassifier model."""
        model = cls(config)
        model_dict = torch.load(config.load_path)
        model.load_state_dict(model_dict['model_state_dict'], strict=strict)

        return model.to(config.device)

    def config_optimizer(self, base_lr, cls_lr, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8):
        """Configure AdamW optimizer for pre-trained param groups and added head for fine-tuning."""
        # Organize parameters into groups with or without weight decay
        decay_params = []
        no_decay_params = []
        classifier_params = []

        for name, param in self.named_parameters():
            # Skip parameters that do not require gradients
            if param.requires_grad:
                # Separate out classifier params with specific learning rate and no weight decay
                if 'ner_head' in name or 'sent_head' in name:
                    classifier_params.append(param)
                # Exclude params less than 2D from weight decay
                # e.g. biases and LayerNorm
                elif param.dim() < 2:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        # Prepare parameter groups for optimizer with specific settings for each group
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': base_lr},
            {'params': no_decay_params, 'weight_decay': 0.0, 'lr': base_lr},
            {'params': classifier_params, 'weight_decay': 0.0, 'lr': cls_lr}
        ]
        # Check if fused version of AdamW is available
        fused_available = 'fused' in inspect.signature(optim.AdamW).parameters
        use_fused = fused_available and self.config.device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        return optim.AdamW(optim_groups, betas=betas, eps=eps, **extra_args)

    def forward(self, input_ids, segments=None, attn_mask=None):
        # Get the base pre-trained BERT model outputs
        hidden_states = super(BERTClassifier, self).forward(input_ids, segments, attn_mask, hidden=True)
        # Apply the NER classification head to all encoder outputs
        ner_logits = self.ner_head(hidden_states)
        # Apply the sentiment classification head to the [CLS] token's output
        sent_logits = self.sent_head(hidden_states[:, 0, :])

        return ner_logits, sent_logits

############################ Uncassed Classifier ###########################

class UncasedClassifier:
    """Uncassed text formatter with a BERT classifier model that tags entities, 
    applies sentiment scores, and reasambles tokenized text during the decoding process."""
    def __init__(self, config):
        self.config = config
        self.model = BERTClassifier.load_weights(config)
        self.tokenizer = Tokenizer.from_file(config.token_path) 
        self.output_text = ""
        self.entities = []
        self.sentiment = {}

    def __call__(self, text, color=False):
        return self.inference(text, color=color)

    def ner_decoder(self, ner_ids, skip_special=False):
        """Convert NER model output ids to their associated tags."""
        decoder = {
            0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: 'O', 4: '##O', 
            5: 'B-PER', 6: '##B-PER', 7: 'I-PER', 8: '##I-PER', 
            9: 'B-LOC', 10: '##B-LOC', 11: 'I-LOC', 12: '##I-LOC', 
            13: 'B-ORG', 14: '##B-ORG', 15: 'I-ORG', 16: '##I-ORG',
            17: 'B-GPE', 18: '##B-GPE', 19: 'I-GPE', 20: '##I-GPE', 
            21: 'B-TIME', 22: '##B-TIME', 23: 'I-TIME', 24: '##I-TIME', 
            25: 'B-MISC', 26: '##B-MISC', 27: 'I-MISC', 28: '##I-MISC', 
        }
        decoded = []
        for id in ner_ids:
            if skip_special:
                if id in [0, 1, 2]:
                    continue
            tag = decoder.get(id, 'O')
            decoded.append(tag)
        return decoded

    def merge_subwords(self, tokens, tags):
        """Merge sub-words in the original tokenized text input and BERT model NER output."""
        merged_tokens = []
        merged_tags = []
        buffer_token = ""

        # Remove special tokens and merge sub-tokens/tags
        for token, tag in zip(tokens, tags):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                if token.startswith("##"):
                    buffer_token += token[2:]  # Remove '##' and merge
                else:
                    if buffer_token:
                        # Append the previous full token and its tag
                        merged_tokens.append(buffer_token)
                        merged_tags.append(current_tag)
                        buffer_token = ""
                    buffer_token = token  # Start new token
                    current_tag = tag  # Update current tag

        # Append the last buffered token
        if buffer_token:
            merged_tokens.append(buffer_token)
            merged_tags.append(current_tag)

        return merged_tokens, merged_tags

    def tag_entities(self, tokens, tags):
        """Tag and capitalize named entities.
        Option to color tagged entities."""

        tagged_ents = []
        token_list = []
        entity = None

        # Tag entities and add to the list
        for token, tag in zip(tokens, tags):
            token = acronym_mapper(token)
            if token == 'i': 
                token = token.upper()
            if tag == 'O' and entity:
                clean_ent = self.format_text(entity[0], False)
                tagged_ents.append((clean_ent, entity[1]))
                token_list.append(clean_ent)
                entity = None
            elif tag.startswith('I') and entity:
                if token in ENTITIES:
                    token = token.upper()
                elif token in STOPWORDS:
                    token = token.lower()
                elif token.islower():
                    token = token.capitalize()
                entity = (entity[0] + ' ' + token, entity[1])
                continue
            elif tag.startswith('B'):
                if entity:
                    clean_ent = self.format_text(entity[0], False)
                    tagged_ents.append((clean_ent, entity[1]))
                    token_list.append(clean_ent)
                    entity = None
                if token in ENTITIES:
                    token = token.upper()
                elif token.islower():
                    token = token.capitalize()
                entity = (token, tag.replace('B-', ''))
                continue

            token_list.append(token)

        output_text = ' '.join(token_list)
        return output_text, tagged_ents

    def format_text(self, text, sentence=True):
        """Format the joined tokenzied text by realigning punctuation
        replicating the decoding process of Hugging Face tokenizers."""
        
        def capitalize(match):
            if match.group(1) == None:
                return match.group(2).upper()
            else:
                return match.group(1) + match.group(2).upper()

        # Close gap between words and hyphen or forward slash
        text = re.sub(r'(\w+)\s(-|/)\s(\w+)', r'\1\2\3', text) 
        # Align punctuation to the left
        text = re.sub(r'(?<=\w)\s([?.!,%:;\]\)}](?:\s|$))', r'\1', text) 
        # Align punctuation to the right
        text = re.sub(r'([{\[\(#$])\s(?=\w)', r'\1', text) 
        # Adjust spacing for punctuation between numbers
        text = re.sub(r'(?<=\d)(,)\s?(\d{3})(?=(,\s?\d{3})*(\D|$))', r'\1\2', text)
        text = re.sub(r'(?<=\d)(\.)\s(\d+)(?=\s)', r'\1\2', text) 
        text = re.sub(r'(?<=\d)(:)\s(\d+)', r'\1\2', text) 
        # Adjust spacing for possesive/conjunction quotation marks
        text = re.sub(r"(?<=\w)\s(')\s(d|m|s|t|ll|re|ve)(?=\s)", r'\1\2', text) 
        # Processing for full sentences
        if sentence:
            # Adjust spacing for double quotes in front and end of the sentence
            text = re.sub(r'(?<=^)(\")\s(?=\w)', r'\1', text) 
            text = re.sub(r'(?<=\S)\s(\")(?=$)', r'\1', text) 
            # Adjust the spacing for content inside double and single quotes inside the sentence
            text = re.sub(r'(?<=\s)(\")\s(.*?)\s(\")(?=\s)', r'\1\2\3', text)
            text = re.sub(r'(?<=\s)(\')\s(.*?)\s(\')(?=\s)', r'\1\2\3', text)
            # Capitalize first word in the sentence
            text = re.sub(r'^(\W)?(\w)', capitalize, text) 

        return text

    def color_tagged_entities(self, text, tagged_ents):
        """Color the tagged entities by type in the output text."""

        def color_entity(entity, tag):
            if tag == 'PER':
                color = Fore.BLUE
            elif tag == 'LOC':
                color = Fore.GREEN
            elif tag == 'ORG':
                color = Fore.MAGENTA
            elif tag == 'GPE':
                color = Fore.RED
            elif tag == 'TIME':
                color = Fore.CYAN
            elif tag == 'MISC':
                color = Fore.YELLOW

            return color + entity + Fore.RESET

        for entity, tag in tagged_ents:
            colored_ent = color_entity(entity, tag)
            text = re.sub(entity, colored_ent, text)

        return text
    
    def format_uncased_text(self, tokens, tags, color=False):
        """
        Format the tokenized input and NER model output 
        aligning input tokens with their associated NER tags.
        Args:\n
            tokens: list of tokens (not IDs) that were passed into the model
            tags: list of NER model IDs decoded into entity representations
            color: True or False | will color the tagged entities\n
                Blue = PER, Green = LOC, Magenta = ORG, Red = GPE, Cyan = TIME, Yellow = MISC
        Return: formatted text and a list of tagged entities
        """
        merged_tokens, merged_tags = self.merge_subwords(tokens, tags)
        text, tagged_ents = self.tag_entities(merged_tokens, merged_tags)
        formatted_text = self.format_text(text)
        if color:
            formatted_text = self.color_tagged_entities(formatted_text, tagged_ents)

        return formatted_text, tagged_ents

    def split_with_spacing(self, original_text):
        """Split sentences and capture spacing in between."""
        sentences = sent_tokenize(original_text)
        # Record the positions and lengths
        positions = [original_text.find(sentence) for sentence in sentences]
        lengths = [len(sentence) for sentence in sentences]
        spacing = []

        end_position = 0
        for start, length in zip(positions, lengths):
            if start > end_position:
                # Capture the \s characters between the end of the 
                # previous sentence and the start of the current one.
                spacing.append(original_text[end_position:start])
            end_position = start + length
        # Capture any trailing content after the last sentence
        spacing.append(original_text[end_position:])

        return sentences, spacing

    def join_with_spacing(self, sentences, spacing):
        """Join sentences in a list with the recorded spacing in between."""
        output_text = ""
        for sentence, spacer in zip(sentences, spacing):
            output_text += sentence + spacer

        return output_text
    
    def average_sentiment(self, sent_scores):
        """Average the sentiment outputs for each sentence."""
        negative, neutral, positive = [], [], []
        for sentiment in sent_scores:
            negative.append(sentiment[0])
            neutral.append(sentiment[1])
            positive.append(sentiment[2])

        return {'negative': np.average(negative), 
                'neutral': np.average(neutral),
                'positive': np.average(positive)
                }

    def inference(self, text, color=False):
        """Convert uncased input text by sampling from the BERTClassifier model. 
        Tags named entities and capitalizes important words in each sentence.
        Reformats the output text with correct spacing, punctuation, and case.
        Provides the average sentiment scores for the output text.
        Args:\n
            text = raw uncased text from a GPT model generation output
            color = True or False | color the tagged entities in the output text"""
        # Split the input text into tokenized sentences and record spacing in between them
        sentences, spacing = self.split_with_spacing(text)

        entities = []
        output_text = []
        sent_scores = []
        # Iterate through NER model inference mode
        for sent in sentences:
            # Encode each sentence in the list
            encoded = self.tokenizer.encode(sent)
            self.model.eval()
            with torch.inference_mode():
                # Conduct a forward pass and get the NER logits
                input_tensor = torch.tensor([encoded.ids], dtype=torch.long).to(self.config.device)
                ner_logits, sent_logits = self.model(input_tensor)

            ner_ids = torch.softmax(ner_logits, dim=-1).argmax(dim=-1)
            sent_preds = torch.softmax(sent_logits, dim=-1).squeeze(0).tolist()
            # Decode the NER model output IDs and return a list of tagged positions
            decoded = self.ner_decoder(ner_ids.squeeze(0).tolist())
            # Match original tokenized text input positions with NER tagged positions and format output text
            formatted_text, tagged_ents = self.format_uncased_text(encoded.tokens, decoded, color=color)

            output_text.append(formatted_text)
            entities.extend(tagged_ents)
            sent_scores.append(sent_preds)

        final_text = self.join_with_spacing(output_text, spacing)
        sentiment = self.average_sentiment(sent_scores)

        self.output_text = final_text
        self.sentiment = sentiment
        self.entities = entities

        return final_text, sentiment, entities

