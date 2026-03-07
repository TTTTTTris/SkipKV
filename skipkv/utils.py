import math
import torch


#################################################################
###################### kv cache utilities #######################
#################################################################
def compute_attention_scores(query_states, key_states, attention_mask=None, pooling="max"):
    batch_size, q_heads, q_len, head_dim = query_states.shape
    kv_heads = key_states.shape[1]
    query_group_size = q_heads // kv_heads


    if attention_mask is not None:  
        # attention_mask: [B, K] with 1=valid, 0=pad
        mask = (~attention_mask) * -1e9
        # expand to [B, 1, 1, K] then broadcast
        mask = mask[:, None, None, None, :]

    if query_group_size == 1:
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = attn_weights + mask
    else:
        # shape: [batch_size, kv_heads, query_group_size, q_len, head_dim]
        query_states = query_states.view(
            batch_size, kv_heads, query_group_size, q_len, head_dim
        )

        # shape: [batch_size, kv_heads, 1, kv_len, head_dim]
        key_states = key_states.unsqueeze(2)

        # shape: [batch_size, kv_heads, query_group_size, q_len, kv_len]
        attn_weights = torch.matmul(
            query_states, key_states.transpose(3, 4)
        ) / math.sqrt(head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            attn_weights = attn_weights + mask

        # apply pooling over query_group_size dimension
        if pooling == "mean":
            attn_weights = attn_weights.mean(dim=2)
        elif pooling == "max":
            attn_weights = attn_weights.max(dim=2).values
        else:
            raise ValueError("Pooling method not supported")

    return attn_weights


def cal_similarity(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    k = key_states[0]
    num_heads = k.shape[0]

    k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

    for h in range(num_heads):
        similarity_cos[h].fill_diagonal_(0.0)

    # shape: [num_heads, seq_len, seq_len]
    similarity_mask = similarity_cos > threshold

    seq_len = similarity_mask.size(-1)
    k = int(seq_len * retain_ratio)

    indices = torch.where(
        similarity_mask,
        torch.arange(similarity_mask.size(-1), device=similarity_mask.device),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )

    # find the last True index in each row
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]

    # find the first True index in each row
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]

    # keep the last_percent% elements
    elif retain_direction == "last_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1)[0][:, :, 0]

    # keep the first_percent% elements
    elif retain_direction == "first_percent":
        similarity_retain = torch.topk(indices, k=k, dim=-1, largest=False)[0][:, :, -1]

    # create indices for zeroing
    batch_idx = (
        torch.arange(num_heads).unsqueeze(1).repeat(1, similarity_retain.size(1))
    )
    seq_idx = torch.arange(similarity_retain.size(1)).unsqueeze(0).repeat(num_heads, 1)

    # zero the specified positions in similarity_cos
    similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

    return similarity_cos.mean(dim=1).softmax(dim=-1)

@torch.no_grad()
def cal_similarity_multi_batch(
    key_states,
    threshold=0.5,
    retain_ratio=0.2,
    retain_direction="last",
):
    """
    Calculate similarity scores for key states with multi-batch support.
    
    Args:
        key_states: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        threshold: Similarity threshold for masking
        retain_ratio: Ratio of tokens to retain for percent-based directions
        retain_direction: Direction for similarity retention ("last", "first", "last_percent", "first_percent")
    
    Returns:
        Tensor of shape (batch_size, seq_len) with similarity scores
    """
    # Input shape: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = key_states.shape
    
    # Normalize keys across the head dimension
    # Shape: (batch_size, num_heads, seq_len, head_dim)
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity: k_norm @ k_norm.T
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    
    # Free memory from k_norm as it's no longer needed
    del k_norm
    torch.cuda.empty_cache()
    
    # Zero out diagonal elements for each head in each batch
    # Create diagonal mask for all batches and heads at once
    diag_mask = torch.eye(seq_len, device=similarity_cos.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
    similarity_cos[diag_mask] = 0.0
    
    # Free diagonal mask
    del diag_mask
    
    # Create similarity mask
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    similarity_mask = similarity_cos > threshold
    
    # Calculate retain count for percent-based methods
    k = int(seq_len * retain_ratio)
    
    # Handle different retention directions with low-memory ops
    if retain_direction == "last":
        # Find the last True index in each row using argmax on the flipped mask
        # Shape: (batch_size, num_heads, seq_len)
        flipped = similarity_mask.flip(dims=[-1])
        last_from_end = torch.argmax(flipped.to(torch.int8), dim=-1)
        del flipped
        any_true = similarity_mask.any(dim=-1)
        similarity_retain = (seq_len - 1 - last_from_end)
        del last_from_end
        # For rows with no True, default to 0
        similarity_retain = torch.where(any_true, similarity_retain, torch.zeros_like(similarity_retain))
        del any_true

    elif retain_direction == "first":
        # Find the first True index per row via argmax on original mask
        first_idx = torch.argmax(similarity_mask.to(torch.int8), dim=-1)
        any_true = similarity_mask.any(dim=-1)
        similarity_retain = torch.where(any_true, first_idx, torch.zeros_like(first_idx))
        del first_idx, any_true

    elif retain_direction == "last_percent":
        # Select among True positions those with largest indices (top-k by index)
        k_eff = min(k, seq_len)
        # Broadcast 1D indices; keep as int32 to reduce memory
        seq_indices_1d = torch.arange(seq_len, device=similarity_mask.device, dtype=torch.int32)
        weights = seq_indices_1d.view(1, 1, 1, -1)
        del seq_indices_1d
        # Mask out False positions with -inf so they don't get selected
        masked = torch.where(
            similarity_mask,
            weights.to(torch.float32),
            torch.full_like(weights, float("-inf")),
        )
        del weights
        # topk over last dim returns indices of selected columns
        similarity_retain = torch.topk(masked, k=k_eff, dim=-1).indices[:, :, :, 0].to(torch.long)
        del masked

    elif retain_direction == "first_percent":
        # Select among True positions those with smallest indices (bottom-k)
        k_eff = min(k, seq_len)
        seq_indices_1d = torch.arange(seq_len, device=similarity_mask.device, dtype=torch.int32)
        weights = seq_indices_1d.view(1, 1, 1, -1)
        del seq_indices_1d
        # For False positions set to very large so they won't be picked in smallest-k
        masked = torch.where(
            similarity_mask,
            weights.to(torch.float32),
            torch.full_like(weights, float("inf")),
        )
        del weights
        similarity_retain = torch.topk(masked, k=k_eff, dim=-1, largest=False).indices[:, :, :, -1].to(torch.long)
        del masked

    else:
        raise ValueError(f"Unknown retain_direction: {retain_direction}")
    
    # Free similarity_mask as it's no longer needed
    del similarity_mask
    torch.cuda.empty_cache()
    
    # Create batch indices for advanced indexing
    # Shape: (batch_size, num_heads, seq_len)
    batch_idx = torch.arange(batch_size, device=similarity_cos.device)
    batch_idx = batch_idx.unsqueeze(1).unsqueeze(2).expand(-1, num_heads, seq_len)
    
    head_idx = torch.arange(num_heads, device=similarity_cos.device)
    head_idx = head_idx.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len)
    
    seq_idx = torch.arange(seq_len, device=similarity_cos.device)
    seq_idx = seq_idx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1)
    
    # Zero the specified positions in similarity_cos
    # Shape remains: (batch_size, num_heads, seq_len, seq_len)
    similarity_cos[batch_idx, head_idx, seq_idx, similarity_retain] = 0
    
    # Free index tensors
    del batch_idx, head_idx, seq_idx, similarity_retain
    
    # Average across heads and apply softmax
    # Shape: (batch_size, num_heads, seq_len, seq_len) -> (batch_size, seq_len)
    similarity_scores = similarity_cos.mean(dim=-2).softmax(dim=-1)
    
    # Free similarity_cos
    del similarity_cos
    torch.cuda.empty_cache()
    
    return similarity_scores

def cal_sentence_similarity(
    key_states,
    completed_punct_ranges,
    aggregation_method="mean"
):
    """
    Calculate cosine similarity among sentences instead of individual tokens.
    
    Args:
        key_states: Key states tensor [batch_size, num_heads, seq_len, head_dim]
        completed_punct_ranges: List of tuples (start_id, end_id) defining sentence boundaries
        threshold: Similarity threshold for determining similar sentences
        retain_ratio: Ratio of sentences to retain
        retain_direction: Direction to retain sentences ("last", "first", "last_percent", "first_percent")
        aggregation_method: How to aggregate tokens within a sentence ("mean", "max", "last")
    
    Returns:
        Token-level similarity scores based on sentence-level similarity
    """
    k = key_states[0]  # [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = k.shape
      
    num_sentences = len(completed_punct_ranges)
    
    # Aggregate tokens within each sentence
    sentence_representations = []
    for i in range(num_sentences):
        start_idx, end_idx = completed_punct_ranges[i]
        
        if start_idx >= end_idx:
            continue
            
        sentence_tokens = k[:, start_idx:end_idx+1, :]  # [num_heads, sentence_len, head_dim]

        # Aggregate tokens within sentence
        if aggregation_method == "mean":
            sentence_repr = sentence_tokens.mean(dim=1)  # [num_heads, head_dim]
        elif aggregation_method == "max":
            sentence_repr = sentence_tokens.max(dim=1)[0]  # [num_heads, head_dim]
        elif aggregation_method == "last":
            sentence_repr = sentence_tokens[:, -1, :]  # [num_heads, head_dim]
        else:
            sentence_repr = sentence_tokens.mean(dim=1)  # Default to mean
        
        sentence_representations.append(sentence_repr)

    if not sentence_representations:
        # Fallback to token-level similarity if no valid sentences
        return torch.zeros(num_heads, seq_len, device=k.device)
    
    # Stack sentence representations
    sentence_reprs = torch.stack(sentence_representations, dim=1)  # [num_heads, num_sentences, head_dim]
    
    # Normalize for cosine similarity
    sentence_reprs_norm = sentence_reprs / (sentence_reprs.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Calculate cosine similarity between sentences
    sentence_similarity = torch.matmul(sentence_reprs_norm, sentence_reprs_norm.transpose(-1, -2))
    # [num_heads, num_sentences, num_sentences]
    
    # Zero out diagonal (self-similarity)
    for h in range(num_heads):
        sentence_similarity[h].fill_diagonal_(0.0)
    
    sentence_similarity_scores = sentence_similarity.max(dim=-1)[0]
    # sentence_similarity_scores = sentence_similarity.mean(dim=1).softmax(dim=-1)


    # Convert sentence-level scores back to token-level scores
    token_similarity_scores = torch.zeros(num_heads, seq_len, device=k.device)
    for i in range(len(sentence_representations)):
        start_idx, end_idx = completed_punct_ranges[i]
        
        if start_idx >= end_idx:
            continue
        # end_idx = seq_len - 1 if end_idx >= seq_len else end_idx

        # Expand final_score[:, i] to match the token span
        sentence_length = end_idx - start_idx + 1
        sentence_score = sentence_similarity_scores[:, i].unsqueeze(-1)  # Shape: [num_heads, 1]
        expanded_score = sentence_score.expand(-1, sentence_length)  # Shape: [num_heads, sentence_length]
        token_similarity_scores[:, start_idx:end_idx+1] = expanded_score

    return token_similarity_scores


def cal_sentence_similarity_head_wise(
    key_states,
    completed_punct_ranges,
    aggregation_method="mean"
):
    """
    Calculate cosine similarity among sentences instead of individual tokens.
    
    Args:
        key_states: Key states tensor [batch_size, num_heads, seq_len, head_dim]
        completed_punct_ranges: List of tuples (start_id, end_id) defining sentence boundaries
        threshold: Similarity threshold for determining similar sentences
        retain_ratio: Ratio of sentences to retain
        retain_direction: Direction to retain sentences ("last", "first", "last_percent", "first_percent")
        aggregation_method: How to aggregate tokens within a sentence ("mean", "max", "last")
    
    Returns:
        Token-level similarity scores based on sentence-level similarity
    """
    k = key_states[0]  # [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = k.shape
      
    token_similarity_scores = torch.zeros(num_heads, seq_len, device=k.device)
    for h in range(num_heads):
        num_sentences = len(completed_punct_ranges[h])
        
        # Aggregate tokens within each sentence
        sentence_representations = []
        for i in range(num_sentences):
            start_idx, end_idx = completed_punct_ranges[h][i]
            
            if start_idx >= end_idx:
                continue
                
            sentence_tokens = k[h, start_idx:end_idx+1, :]  # [sentence_len, head_dim]

            # Aggregate tokens within sentence
            if aggregation_method == "mean":
                sentence_repr = sentence_tokens.mean(dim=0)  # [head_dim]
            elif aggregation_method == "max":
                sentence_repr = sentence_tokens.max(dim=0)[0]  # [head_dim]
            elif aggregation_method == "last":
                sentence_repr = sentence_tokens[-1, :]  # [head_dim]
            else:
                sentence_repr = sentence_tokens.mean(dim=0)  # Default to mean
            
            sentence_representations.append(sentence_repr)

        if not sentence_representations:
            # Fallback to token-level similarity if no valid sentences
            return torch.zeros(num_heads, seq_len, device=k.device)
        
        # Stack sentence representations
        sentence_reprs = torch.stack(sentence_representations, dim=0)  # [num_sentences, head_dim]
        
        # Normalize for cosine similarity
        sentence_reprs_norm = sentence_reprs / (sentence_reprs.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Calculate cosine similarity between sentences
        sentence_similarity = torch.matmul(sentence_reprs_norm, sentence_reprs_norm.transpose(-1, -2))
        # [num_sentences, num_sentences]
        
        # Zero out diagonal (self-similarity)
        sentence_similarity.fill_diagonal_(0.0)
        
        sentence_similarity_scores = sentence_similarity.max(dim=-1)[0]

        # Convert sentence-level scores back to token-level scores
        for i in range(len(sentence_representations)):
            start_idx, end_idx = completed_punct_ranges[h][i]
            
            if start_idx >= end_idx:
                continue
            # end_idx = seq_len - 1 if end_idx >= seq_len else end_idx

            # Expand final_score[:, i] to match the token span
            sentence_length = end_idx - start_idx + 1
            sentence_score = sentence_similarity_scores[i].unsqueeze(-1)  # Shape: [1]
            expanded_score = sentence_score.expand(sentence_length)  # Shape: [sentence_length]
            token_similarity_scores[h, start_idx:end_idx+1] = expanded_score

    return token_similarity_scores

def cal_sentence_similarity_head_wise_pair(
    key_states,
    sentence_ranges,  # List of (start_idx, end_idx) tuples
    similarity_threshold=0.95,
    aggregation_method="mean"
):
    """
    Remove duplicate sentences based on similarity, keeping the latest occurrence.
    
    Args:
        key_states: Key states tensor [num_heads, seq_len, head_dim]
        sentence_ranges: List of (start_idx, end_idx) tuples defining sentence boundaries
        similarity_threshold: Threshold above which sentences are considered duplicates
        aggregation_method: How to aggregate tokens within a sentence
    
    Returns:
        Token-level mask [num_heads, seq_len] where 1=keep, 0=remove
    """
    k = key_states[0]  # [num_heads, seq_len, head_dim]
    num_heads, seq_len, head_dim = k.shape
    
    # Initialize keep mask (1=keep, 0=remove)
    keep_mask = torch.ones(num_heads, seq_len, device=k.device)
    
    for h in range(num_heads):
        num_sentences = len(sentence_ranges[h])
        
        if num_sentences <= 1:
            continue  # Nothing to deduplicate
            
        # Aggregate tokens within each sentence
        sentence_representations = []
        valid_sentence_indices = []
        
        for i, (start_idx, end_idx) in enumerate(sentence_ranges[h]):
            if start_idx >= end_idx or end_idx >= seq_len:
                continue
                
            sentence_tokens = k[h, start_idx:end_idx+1, :]  # [sentence_len, head_dim]

            # Aggregate tokens within sentence
            if aggregation_method == "mean":
                sentence_repr = sentence_tokens.mean(dim=0)  # [head_dim]
            elif aggregation_method == "max":
                sentence_repr = sentence_tokens.max(dim=0)[0]  # [head_dim]
            elif aggregation_method == "last":
                sentence_repr = sentence_tokens[-1, :]  # [head_dim]
            else:
                sentence_repr = sentence_tokens.mean(dim=0)  # Default to mean
            
            sentence_representations.append(sentence_repr)
            valid_sentence_indices.append(i)

        if len(sentence_representations) <= 1:
            continue  # Nothing to compare
        
        # Stack sentence representations
        sentence_reprs = torch.stack(sentence_representations, dim=0)  # [num_valid_sentences, head_dim]
        
        # Normalize for cosine similarity
        sentence_reprs_norm = sentence_reprs / (sentence_reprs.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Calculate cosine similarity between sentences
        sentence_similarity = torch.matmul(sentence_reprs_norm, sentence_reprs_norm.transpose(-1, -2))
        # [num_valid_sentences, num_valid_sentences]
        
        # Zero out diagonal and lower triangle (avoid duplicate comparisons)
        sentence_similarity = torch.triu(sentence_similarity, diagonal=1)
        
        # Find sentence pairs above threshold
        similar_pairs = torch.where(sentence_similarity > similarity_threshold)
        
        # For each similar pair, mark the earlier sentence for removal
        sentences_to_remove = set()
        for i, j in zip(similar_pairs[0].cpu().numpy(), similar_pairs[1].cpu().numpy()):
            # i < j due to upper triangular, so remove sentence i (keep j which is later)
            sentences_to_remove.add(valid_sentence_indices[i])
        
        # Apply removal to token-level mask
        for sentence_idx in sentences_to_remove:
            start_idx, end_idx = sentence_ranges[h][sentence_idx]
            if start_idx < end_idx and end_idx < seq_len:
                keep_mask[h, start_idx:end_idx+1] = 0
    
    return keep_mask
#################################################################
################### visualization utilities #####################
#################################################################

# Visualize the token eviction pattern for a given head
def visualize_token_eviction(
    output_token_ids, kept_token_indices, tokenizer, head_idx=0, kv_budget=0, save_path=None
):
    """
    Visualize which tokens are kept vs evicted for a given head

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices: either a single tensor of shape (num_kv_heads, num_kept_tokens) 
                           or a list of such tensors from multiple compression steps
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
    """
    from IPython.display import HTML

    # Handle both single tensor and list of tensors
    if isinstance(kept_token_indices, list):
        # If it's a list, use the last compression step
        kept_indices_tensor = kept_token_indices[-1]
    else:
        # If it's a single tensor
        kept_indices_tensor = kept_token_indices
    
    # Get the kept indices for the specified head and flatten them
    kept_indices = set(kept_indices_tensor[head_idx].flatten().tolist())

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # set rest of the outputs kept
    print(len(tokens), max(kept_indices))
    if (len(tokens) - max(kept_indices)) <= kv_budget:
        max_kept_idx = max(kept_indices) if kept_indices else -1
        remaining_indices = set(range(max_kept_idx + 1, len(tokens)))
        print(remaining_indices)
        kept_indices.update(remaining_indices)

    # Build HTML with different colors for kept vs evicted tokens
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")  # Remove space marker
            .replace("Ċ", "\n")  # Convert newline marker to actual newline
            .replace("<|begin of sentence|>", "[BOS]")
            .replace("<|end of sentence|>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        if idx in kept_indices:
            # Kept tokens in green with bold
            html_parts.append(
                f'<span style="color: blue; font-weight: bold;">{token}</span>'
            )
        else:
            # Evicted tokens in gray and lighter
            html_parts.append(f'<span style="color: #999999;">{token}</span>')

    # Join without spaces (since we're now handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'
    # Save if a path is given
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Visualization saved to {save_path}")
        
    return HTML(html)

# Visualize the token eviction pattern for a given heads at each compression step
def visualize_multistep_token_eviction(
    output_token_ids, kept_token_indices_list, tokenizer, head_idx=0, step_idx=-1
):
    """
    Visualize which tokens are kept at each compression step with different colors.
    Later steps are shown in more vibrant colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
    """
    from IPython.display import HTML

    # Get the kept indices for each step for the specified head
    kept_indices_by_step = [
        set(indices[head_idx].tolist()) for indices in kept_token_indices_list
    ]
    num_steps = len(kept_indices_by_step) if step_idx == -1 else 1

    # Generate colors using a distinct color spectrum
    def get_color(step):
        # Use a color spectrum for better distinction between steps
        if num_steps <= 1:
            return "#3498db"  # Default blue if only one step

        # Define a set of distinct colors
        colors = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#2ecc71",  # Green
            "#f39c12",  # Orange
            "#9b59b6",  # Purple
            "#1abc9c",  # Teal
            "#d35400",  # Dark Orange
            "#2980b9",  # Dark Blue
            "#27ae60",  # Dark Green
            "#8e44ad",  # Dark Purple
        ]

        if num_steps <= len(colors):
            # If we have fewer steps than colors, use the colors directly
            return colors[step % len(colors)]
        else:
            # For more steps than colors, interpolate between colors
            # Map step to a position in the color spectrum
            position = (step / (num_steps - 1)) * (len(colors) - 1)
            idx1 = int(position)
            idx2 = min(idx1 + 1, len(colors) - 1)
            fraction = position - idx1

            # Get the two colors to interpolate between
            color1 = colors[idx1]
            color2 = colors[idx2]

            # Convert hex to RGB
            r1, g1, b1 = (
                int(color1[1:3], 16),
                int(color1[3:5], 16),
                int(color1[5:7], 16),
            )
            r2, g2, b2 = (
                int(color2[1:3], 16),
                int(color2[3:5], 16),
                int(color2[5:7], 16),
            )

            # Interpolate
            r = int(r1 * (1 - fraction) + r2 * fraction)
            g = int(g1 * (1 - fraction) + g2 * fraction)
            b = int(b1 * (1 - fraction) + b2 * fraction)

            return f"#{r:02x}{g:02x}{b:02x}"

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<|begin of sentence|>", "[BOS]")
            .replace("<|end of sentence|>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        latest_step = -1
        if step_idx == -1:
            # Find the latest step (if any) where this token was kept
            for step, kept_indices in enumerate(kept_indices_by_step[::-1]):
                if idx in kept_indices:
                    latest_step = num_steps - step
                    break

        elif idx in kept_indices_by_step[step_idx]:
            latest_step = num_steps

        # Color the token based on its latest appearance
        if latest_step >= 0:
            color = get_color(latest_step)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)


# Visualize the token eviction pattern for all heads at each compression step
def visualize_multistep_token_eviction_by_head(
    output_token_ids, kept_token_indices_list, tokenizer, step_idx, aggregate=False, save_path=None
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using a distinct color spectrum
    def get_color(idx, aggregate):
        # Define a set of distinct colors
        if not aggregate:
            colors = [
                "#3498db",  # Blue
                "#f39c12",  # Orange
                "#9b59b6",  # Purple
                "#1abc9c",  # Teal
                "#d35400",  # Dark Orange
                "#2980b9",  # Dark Blue
                "#27ae60",  # Dark Green
                "#8e44ad",  # Dark Purple
            ]
        else:
            # colors = [
            #     "#D6EAF8",  # Very Light Blue
            #     "#AED6F1",  # Light Blue
            #     "#85C1E9",  # Medium Light Blue
            #     "#5DADE2",  # Medium Blue
            #     "#3498DB",  # Blue
            #     "#2E86C1",  # Medium Dark Blue
            #     "#2874A6",  # Dark Blue
            #     "#1B4F72",  # Very Dark Blue
            # ]
            colors = [
                "#E65100",  # 深橙色(起点)
                "#D84315",  # 橙红色
                "#C62828",  # 红褐色
                "#B71C1C",  # 深红褐色
                "#A52A00",  # 深橙褐色
                "#8B2500",  # 赭石色
                "#7C2000",  # 深褐色
                "#6B1D00"   # 最深的橙褐色
            ]
        return colors[idx]

    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: set(token_indices_lst[i].tolist()) for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<|begin of sentence|>", "[BOS]")
            .replace("<|end of sentence|>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        color_idx = -1
        for head_idx, kept_token_set in token_indices_dict.items():
            if idx in kept_token_set:
                if aggregate:
                    color_idx += 1
                else:
                    color_idx = head_idx

        # Color the token based on its latest appearance
        if color_idx >= 0:
            color = get_color(color_idx, aggregate)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'
    # Save if a path is given
    if save_path is not None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Visualization saved to {save_path}")
    return HTML(html)


# Visualize the token eviction score for all heads at each compression step
def visualize_multistep_token_eviction_score_by_head(
    output_token_ids, kept_token_indices_list, score_list, tokenizer, step_idx, head_idx
):
    """
    Visualize which tokens are kept by which heads with different colors.

    Args:
        output_token_ids: shape (seq_len, )
        kept_token_indices_list: list of tensors, each with shape (num_kv_heads, num_kept_tokens)
        tokenizer: tokenizer for decoding
        head_idx: which head's eviction pattern to visualize (default 0)
        step: which step to visualize (default -1, visualize all steps)
        aggregate: when set to False, later heads will cover previous heads. when set to `True`, will compute how many times a token are covered by a head.
    """
    from IPython.display import HTML

    # Generate colors using the common blue to yellow heatmap color spectrum
    def get_color(score):
        # Define the blue and yellow colors
        colors = [
                "#E65100",  # 深橙色(起点)
                "#D84315",  # 橙红色
                "#C62828",  # 红褐色
                "#B71C1C",  # 深红褐色
                "#A52A00",  # 深橙褐色
                "#8B2500",  # 赭石色
                "#7C2000",  # 深褐色
                "#6B1D00"   # 最深的橙褐色
        ]

        if score <= 0:
            return colors[0]

        # Calculate the position of the step within the range of colors
        position = score * (len(colors) - 1)

        # Determine the indices for interpolation
        idx1 = int(position)
        idx2 = min(idx1 + 1, len(colors) - 1)
        fraction = position - idx1

        # Get the two colors to interpolate between
        color1 = colors[idx1]
        color2 = colors[idx2]

    # Convert hex to RGB for color1
        r1, g1, b1 = (
            int(color1[1:3], 16),
            int(color1[3:5], 16),
            int(color1[5:7], 16),
        )
        # Convert hex to RGB for color2
        r2, g2, b2 = (
            int(color2[1:3], 16),
            int(color2[3:5], 16),
            int(color2[5:7], 16),
        )

        # Interpolate between the two colors
        r = int(r1 * (1 - fraction) + r2 * fraction)
        g = int(g1 * (1 - fraction) + g2 * fraction)
        b = int(b1 * (1 - fraction) + b2 * fraction)

        # Return the interpolated color as a hex code
        return f"#{r:02x}{g:02x}{b:02x}"


    # Decode all tokens
    tokens = tokenizer.convert_ids_to_tokens(output_token_ids)

    # Get kept token id list
    token_indices_lst = kept_token_indices_list[
        step_idx
    ]  # shape: (kv_head, num_kept_tokens)
    token_indices_dict = {
        i: token_indices_lst[i].tolist() for i in range(token_indices_lst.shape[0])
    }

    # Build HTML with different colors for kept tokens at each step
    html_parts = []
    for idx, token in enumerate(tokens):
        # Clean up special tokens and formatting
        token = (
            token.replace("Ġ", " ")
            .replace("Ċ", "\n")
            .replace("<|begin of sentence|>", "[BOS]")
            .replace("<|end of sentence|>", "[EOS]")
            .replace("<s>", "[BOS]")
            .replace("</s>", "[EOS]")
        )

        score = -1
        # for head_idx, kept_token_set in token_indices_dict.items():
            # if idx in kept_token_set:

        # 定位idx在kept_token_set的index
        if idx in token_indices_dict[head_idx]:
            index = token_indices_dict[head_idx].index(idx)
        else:
            # 处理 idx 不在列表中的情况
            index = -1  # 或者其他合适的值

        if index != -1:
            score = score_list[step_idx][head_idx][index].item()

            # if aggregate:
            #     color_idx += 1
            # else:
            #     color_idx = head_idx

        # Color the token based on its latest appearance
        if score >= 0:
            color = get_color(score)
            html_parts.append(
                f'<span style="color: {color}; font-weight: bold;">{token}</span>'
            )
        else:
            html_parts.append(f'<span style="color: #CCCCCC;">{token}</span>')

    # Join without spaces (since we're handling spaces explicitly)
    html = f'<pre style="font-family: monospace; white-space: pre-wrap; word-wrap: break-word;">{"".join(html_parts)}</pre>'

    return HTML(html)