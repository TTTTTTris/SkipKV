from copy import deepcopy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, cal_similarity_multi_batch, compute_attention_scores, cal_sentence_similarity, cal_sentence_similarity_head_wise, cal_sentence_similarity_head_wise_pair


class SkipKV:
    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        record_kept_token_indices=False,
        **kwargs,
    ):
        assert budget - window_size > 0, "budget must be greater than window_size"
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction

        # for recording kept token indices
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

        self.wait_ranges = None
        self.punct_ranges = {}  # Dictionary to store per-head ranges: {head_idx: [ranges]}
        self.num_sentences = 0
        self.n_remove = 0

    def reset_for_new_batch(self):
        self.wait_ranges = None
        self.punct_ranges = {}  # Reset per-head ranges
        self.num_sentences = 0
        self.n_remove = 0
        logging.info('Reset punct ranges for new sample!')
        
        # Reset kept token indices state if recording is enabled
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def update_punct_ranges_after_compression_head_wise(self, indices, kv_cache_len, punct_ranges):
        """
        Update punctuation ranges for each head separately after compression.
        Each head can have different surviving positions, so ranges are tracked per-head.
        """
        if not punct_ranges:  # If no ranges exist for any head
            return
        
        # Get the position indices - shape: (bsz, num_kv_heads, budget - window_size)
        indices_cpu = indices.cpu()  # Shape: (num_kv_heads, budget - window_size)
        num_heads = indices_cpu.shape[0]
        
        # Process each head separately
        for head_idx in range(num_heads):
            head_indices = indices_cpu[head_idx]  # Shape: (budget - window_size,)

            # # Add window positions
            window_start = kv_cache_len - self.window_size
            window_indices = torch.arange(window_start, kv_cache_len)
            head_indices = torch.cat((head_indices, window_indices), dim=0)
            
            # Update ranges for this head
            updated_punct_ranges = {}
            
            for key, (punct_start, punct_end) in punct_ranges[head_idx].items():
                # head_indices is sorted ascending
                lo = torch.searchsorted(head_indices, punct_start, right=False).item()
                hi = torch.searchsorted(head_indices, punct_end, right=True).item()

                if lo < hi:  # means some indices survived in this range
                    new_start = lo
                    new_end = hi - 1
                    updated_punct_ranges[key] = (new_start, new_end)

            
            # Store updated ranges for this head
            punct_ranges[head_idx] = updated_punct_ranges

    def update_punct_ranges_before_compression_head_wise(self, num_heads, kv_cache_len, current_pos, num_sentences, completed_punct_ranges, punct_ranges):
        for h in range(num_heads):
            if completed_punct_ranges is not None and len(completed_punct_ranges) > 0:
                if not punct_ranges[h]:
                    if completed_punct_ranges[-1][-1] <= kv_cache_len:
                        for j, cwr in enumerate(completed_punct_ranges):
                            if j >= num_sentences:
                                punct_ranges[h][cwr] = (cwr)
                                    
                else:
                    for j, cwr in enumerate(completed_punct_ranges):
                        if j >= num_sentences:
                            last_global = list(punct_ranges[h].keys())[-1]           # (global_start, global_end)
                            last_local = punct_ranges[h][last_global]                # (local_start, local_end)
                            # start_id = punct_ranges[h][-1][-1] + 1
                            start_id = last_local[-1] + 1
                            end_id = kv_cache_len - current_pos + cwr[1]
                            if end_id < kv_cache_len:
                                punct_ranges[h][cwr] = (start_id , end_id)

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        layer_idx,
        sentence_similarity=None,
        completed_punct_ranges=None,
        remove_punct_ranges=None,
        current_pos=0,
        padding_mask=None,
    ):
        batch_size, num_heads, kv_cache_len, head_dim = key_states.shape

        if kv_cache_len < self.budget:
            return key_states, value_states
        else:
            attn_weights = compute_attention_scores(query_states, key_states, padding_mask)

            attn_weights_sum = (
                nn.functional.softmax(
                    attn_weights[:, :, -self.window_size :, : -self.window_size],
                    dim=-1,
                    dtype=torch.float32,
                )
                .mean(dim=-2)
                .to(query_states.dtype)
            )
            # TODO: Softmax then reduce head

            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )

            # add new punct ranges; relative position
            if not self.punct_ranges:
                self.punct_ranges = [{i: {} for i in range(num_heads)} for _ in range(batch_size)]
                self.num_sentences = [0] * batch_size


            for b in range(batch_size):
                self.update_punct_ranges_before_compression_head_wise(num_heads, kv_cache_len, current_pos[b], self.num_sentences[b], completed_punct_ranges[b], self.punct_ranges[b])


            # monitor punct range
            # if not layer_idx:
                # logging.info(f'Input punct ranges: {layer_idx}, {self.punct_ranges}')

            # assert punct range overlapping
            # for i, (start1, end1) in enumerate(self.punct_ranges[0]):
            #     for j, (start2, end2) in enumerate(self.punct_ranges[0][i+1:], i+1):
            #         if start1 < end2 and start2 < end1:
            #             print(f"ranges {i} and {j} overlap!")
            #             print(self.punct_ranges[0])
            #             exit(0)

            # monitor punct range
            # if not layer_idx and self.punct_ranges[0]:
                # last_global = list(self.punct_ranges[0].keys())[-1]           # (global_start, global_end)
                # last_local = self.punct_ranges[0][last_global]                # (local_start, local_end)
                # logging.info(f"{last_global}, {last_local}, {kv_cache_len}, {current_pos}")

            if padding_mask is not None:  
                valid_mask = padding_mask
                valid_mask = padding_mask[:, None, :, None].expand_as(key_states)
                key_states_mask = key_states.masked_fill(valid_mask == 0, float(0))
            else: 
                key_states_mask = key_states

            similarity_cos = cal_similarity_multi_batch(
                key_states_mask,
                retain_ratio=self.retain_ratio,
                retain_direction=self.retain_direction,
            )[:, :, : -self.window_size]  # [B, H, B-W]

            # --------- apply padding mask to final_score ----------
            final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda) # [B, H ,S]

            if padding_mask is not None:
                # padding_mask: [bsz, seq_len]
                valid_mask = padding_mask[:, : -self.window_size]  # ignore last window
                valid_mask = valid_mask.unsqueeze(1).expand_as(final_score)  # [bsz, num_heads, seq_len-window]
                final_score = final_score.masked_fill(valid_mask == 0, float(100))
            # ------------------------------------------------------

            if remove_punct_ranges is not None:
                for b in range(batch_size):
                    for cwr in remove_punct_ranges[b]:
                        for h in range(len(self.punct_ranges[b])):
                            if(cwr in self.punct_ranges[b][h]):
                                start_idx, end_idx = self.punct_ranges[b][h][cwr]
                                if start_idx < end_idx and end_idx < kv_cache_len:
                                    final_score[b, h, start_idx:end_idx+1] -= remove_punct_ranges[b][cwr]

            # shape: (bsz, num_kv_heads, budget - window_size)
            indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
            indices = torch.sort(indices, dim=-1)[0] # this degrades the performance

            for b in range(batch_size):
                self.update_punct_ranges_after_compression_head_wise(indices[b], kv_cache_len, self.punct_ranges[b])
            
            # monitor punct range
            # if not layer_idx:
                # logging.info(f'Updated punct ranges: {self.punct_ranges}')

            #####################################################
            ###### Store evicted token indices start ############
            #####################################################
            # shape: (num_kv_heads, budget - window_size)
            if self.record_kept_token_indices:
                indices_cl = indices.clone().squeeze(0).to("cpu")

                similarity_cos_analysis = cal_similarity(
                    # value_states,
                    key_states,
                    retain_ratio=self.retain_ratio,
                    retain_direction=self.retain_direction,
                )

                attn_weights_sum_analysis = (
                    nn.functional.softmax(
                        attn_weights,
                        dim=-1,
                        dtype=torch.float32,
                    )
                    .mean(dim=-2)
                    .to(query_states.dtype)
                )

                attn_cache_analysis = F.max_pool1d(
                    attn_weights_sum_analysis,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1,
                )

                final_score_analysis = attn_cache_analysis * self.mix_lambda - similarity_cos_analysis * (
                    1 - self.mix_lambda
                )

                recent_window_indices = torch.arange(
                    kv_cache_len - self.window_size, kv_cache_len, device="cpu"
                ).expand(indices_cl.shape[0], -1)
                cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################

                # Gather the scores for the kept tokens
                attn_scores = attn_cache_analysis.clone().squeeze(0).to("cpu")
                sim_scores = similarity_cos_analysis.clone().squeeze(0).to("cpu")
                fin_scores = final_score_analysis.clone().squeeze(0).to("cpu")

                # print(f"cur_indices {cur_indices} attn_cache_analysis {attn_cache_analysis.shape} similarity_cos_analysis {similarity_cos_analysis.shape} final_score_analysis {final_score_analysis.shape}")

                # Gather the scores based on index
                kept_attn = torch.gather(attn_scores, dim=1, index=cur_indices)
                kept_sim = torch.gather(sim_scores, dim=1, index=cur_indices)
                kept_final = torch.gather(fin_scores, dim=1, index=cur_indices)

                #####################################################

                if self.evicted_token_num > 0:
                    prev_indices = self.kept_token_indices[-1]
                    mask = cur_indices < self.budget

                    for i in range(cur_indices.shape[0]):
                        positions = torch.where(mask[i])[0]

                        # For each position, get the value and use it as an index into prev_indices
                        for pos in positions:
                            val = cur_indices[i, pos].item()
                            cur_indices[i, pos] = prev_indices[i, val]

                    # For values >= self.budget, add the evicted token count
                    cur_indices[~mask] += self.evicted_token_num

                #####################################################
                ### Store final scores, attention and similarity ####
                #####################################################
                self.kept_attention_scores.append(kept_attn)
                self.kept_similarity_scores.append(kept_sim)
                self.kept_final_scores.append(kept_final)
                #####################################################

                self.kept_token_indices.append(cur_indices)
                self.evicted_token_num += kv_cache_len - self.budget
            ######################################################

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
                            
            # record the num sentences in the last step, monitor if there are new setences in the next step
            if completed_punct_ranges is not None:
                for b in range(batch_size):
                # self.n_remove += kv_cache_len - self.budget
                    self.num_sentences[b] = len(completed_punct_ranges[b])

            return key_states, value_states
