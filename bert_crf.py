import logging
import math

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

from compressed_embedding import create_compressed_embedding
from crf import CRF
from utils_maven import to_crf_pad, unpad_crf

logger = logging.getLogger(__name__)


def _make_attention_stack(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
    )


def _resolve_attention_mask(last_hidden_state, attention_mask):
    if attention_mask is None:
        return torch.ones(
            last_hidden_state.shape[:2],
            dtype=torch.bool,
            device=last_hidden_state.device,
        )
    return attention_mask.to(device=last_hidden_state.device).bool()


def _masked_softmax(scores, mask, dim):
    mask = mask.to(device=scores.device, dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    all_masked = ~mask.any(dim=dim, keepdim=True)
    masked_scores = torch.where(all_masked, torch.zeros_like(masked_scores), masked_scores)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-9)


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        if num_heads <= 0:
            raise ValueError("attention_pooling_heads must be positive")
        self.num_heads = num_heads
        self.attention = _make_attention_stack(hidden_size, num_heads)
        self.fuse = nn.Linear(hidden_size * num_heads, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, last_hidden_state, attention_mask):
        mask = _resolve_attention_mask(last_hidden_state, attention_mask)
        scores = self.attention(last_hidden_state).float()
        weights = _masked_softmax(scores, mask.unsqueeze(-1), dim=1)
        contexts = torch.einsum("blh,bld->bhd", weights, last_hidden_state)
        pooled = self.fuse(contexts.reshape(contexts.size(0), -1))
        expanded = pooled.unsqueeze(1).expand(-1, last_hidden_state.size(1), -1)
        gate = torch.sigmoid(self.gate(torch.cat([last_hidden_state, expanded], dim=-1)))
        return last_hidden_state + gate * expanded


def build_attention_pooling(config):
    variant = getattr(config, "attention_pooling_variant", "multihead")
    if variant != "multihead":
        raise ValueError("TBEF only supports attention_pooling_variant='multihead'.")
    return MultiHeadAttentionPooling(
        config.hidden_size,
        num_heads=getattr(config, "attention_pooling_heads", 4),
    )


def _save_attention_pooling_config(model):
    model.config.use_attention_pooling = model.use_attention_pooling
    model.config.attention_pooling_variant = model.attention_pooling_variant
    model.config.attention_pooling_heads = model.attention_pooling_heads


def _save_sentence_event_fusion_config(model):
    model.config.use_sentence_event_fusion = getattr(model, "use_sentence_event_fusion", False)
    model.config.sentence_event_fusion_gate_bias = getattr(
        model, "sentence_event_fusion_gate_bias", 3.0
    )


def _masked_max_pool(hidden_states, attention_mask):
    mask = _resolve_attention_mask(hidden_states, attention_mask)
    masked_hidden = hidden_states.masked_fill(
        ~mask.unsqueeze(-1),
        torch.finfo(hidden_states.dtype).min,
    )
    pooled = masked_hidden.max(dim=1).values
    all_masked = ~mask.any(dim=1, keepdim=True)
    return torch.where(all_masked, torch.zeros_like(pooled), pooled)


class GateFusion(nn.Module):
    def __init__(self, hidden_size, init_bias=0.0):
        super().__init__()
        self.init_bias = init_bias
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, self.init_bias)

    def forward(self, first, second, mask=None):
        gate = torch.sigmoid(self.gate(torch.cat([first, second], dim=-1)))
        fused = gate * first + (1.0 - gate) * second
        if mask is not None:
            fused = fused * mask.unsqueeze(-1).to(dtype=fused.dtype)
        return fused


class SentenceEventFusionLayer(nn.Module):
    """Sentence-level event fusion with event-type attention and two gates."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_event_types = getattr(config, "num_maven_types", max(1, (config.num_labels - 1) // 2))
        self.identity_gate_bias = getattr(config, "sentence_event_fusion_gate_bias", 3.0)

        self.event_type_embeddings = nn.Embedding(self.num_event_types, self.hidden_size)
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.global_event_gate = GateFusion(self.hidden_size, init_bias=self.identity_gate_bias)
        self.target_event_gate = GateFusion(self.hidden_size, init_bias=self.identity_gate_bias)

        self.token_type_projection = nn.Linear(self.hidden_size, self.num_event_types)
        self.sentence_type_projection = nn.Linear(self.hidden_size, self.num_event_types)
        self._reset_event_fusion_parameters()

    def _reset_event_fusion_parameters(self):
        nn.init.zeros_(self.event_type_embeddings.weight)
        nn.init.zeros_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.token_type_projection.weight)
        nn.init.zeros_(self.token_type_projection.bias)
        nn.init.zeros_(self.sentence_type_projection.weight)
        nn.init.zeros_(self.sentence_type_projection.bias)
        self.global_event_gate.reset_parameters()
        self.target_event_gate.reset_parameters()

    def forward(self, sequence_output, attention_mask):
        if sequence_output.dim() != 3:
            raise ValueError(
                "SentenceEventFusionLayer expects [batch, seq_len, hidden], "
                f"got {tuple(sequence_output.shape)}"
            )

        token_mask = _resolve_attention_mask(sequence_output, attention_mask)
        type_embeddings = self.event_type_embeddings.weight

        query = self.query(sequence_output)
        key = self.key(type_embeddings)
        value = self.value(type_embeddings)

        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.hidden_size)
        global_event_weights = torch.softmax(scores, dim=-1)
        global_event_repr = torch.matmul(global_event_weights, value)
        global_event_repr = global_event_repr * token_mask.unsqueeze(-1).to(dtype=global_event_repr.dtype)

        fused_context = self.global_event_gate(sequence_output, global_event_repr, mask=token_mask)

        sentence_repr = _masked_max_pool(fused_context, token_mask)
        token_type_logits = self.token_type_projection(fused_context)
        sentence_type_logits = self.sentence_type_projection(sentence_repr).unsqueeze(1)

        mixed_type_probs = torch.softmax(token_type_logits + sentence_type_logits, dim=-1)
        target_event_repr = torch.matmul(mixed_type_probs, type_embeddings)
        target_event_repr = target_event_repr * token_mask.unsqueeze(-1).to(dtype=target_event_repr.dtype)

        return self.target_event_gate(fused_context, target_event_repr, mask=token_mask)


def _run_crf_decode(model, logits, attention_mask, labels, pad_token_label_id, extra_loss=None):
    if labels is not None:
        pad_mask = labels != pad_token_label_id
        loss_mask = ((attention_mask == 1) & pad_mask) if attention_mask is not None else pad_mask
        valid_rows = loss_mask.any(dim=1)
        best_path = labels.clone().detach()

        if not valid_rows.any():
            loss = logits.new_zeros(())
            if extra_loss is not None:
                loss = loss + extra_loss
            return loss, best_path

        loss = None
        valid_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
        for start in range(0, valid_indices.numel(), 4):
            chunk = valid_indices[start : start + 4]
            crf_labels, crf_mask = to_crf_pad(labels[chunk], loss_mask[chunk], pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits[chunk], loss_mask[chunk], pad_token_label_id)
            chunk_loss = model.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            loss = chunk_loss if loss is None else loss + chunk_loss
            best_path[chunk] = unpad_crf(
                model.crf(crf_logits, crf_mask),
                crf_mask,
                labels[chunk],
                loss_mask[chunk],
            )

        if loss is None:
            loss = logits.new_zeros(())
        if extra_loss is not None:
            loss = loss + extra_loss
        return loss, best_path

    mask = attention_mask == 1 if attention_mask is not None else torch.ones(
        logits.shape[:2], dtype=torch.bool, device=logits.device
    )
    valid_rows = mask.any(dim=1)
    best_path = torch.full(mask.shape, pad_token_label_id, dtype=torch.long, device=logits.device)
    valid_indices = torch.nonzero(valid_rows, as_tuple=False).flatten()
    for start in range(0, valid_indices.numel(), 4):
        chunk = valid_indices[start : start + 4]
        crf_logits, crf_pad = to_crf_pad(logits[chunk], mask[chunk], pad_token_label_id)
        crf_mask = crf_pad.sum(dim=2) == crf_pad.shape[2]
        temp_labels = torch.full(mask[chunk].shape, pad_token_label_id, dtype=torch.long, device=logits.device)
        best_path[chunk] = unpad_crf(
            model.crf(crf_logits, crf_mask),
            crf_mask,
            temp_labels,
            mask[chunk],
        )
    return None, best_path


class BertCompressedCRFForTokenClassification(BertPreTrainedModel):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.compression_method = getattr(config, "compression_method", None)
        self.tsvd_dim = getattr(config, "tsvd_dim", None)
        self.is_decomposed = getattr(config, "is_decomposed", False)
        self.gate_sparsity_lambda = getattr(config, "gate_sparsity_lambda", 0.0)

        self.use_attention_pooling = getattr(config, "use_attention_pooling", False)
        self.attention_pooling_variant = getattr(config, "attention_pooling_variant", "multihead")
        self.attention_pooling_heads = getattr(config, "attention_pooling_heads", 4)
        self.use_sentence_event_fusion = getattr(config, "use_sentence_event_fusion", False)
        self.sentence_event_fusion_gate_bias = getattr(config, "sentence_event_fusion_gate_bias", 3.0)
        self.pruned_intermediate_sizes = getattr(config, "pruned_intermediate_sizes", None)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.compressed_embedding = None
        if self.is_decomposed and self.compression_method is not None:
            self._rebuild_compressed_embedding_from_config(config)

        self.attention_pooling = None
        if self.use_attention_pooling:
            self.attention_pooling = build_attention_pooling(config)
        self.sentence_event_fusion = None
        if self.use_sentence_event_fusion:
            self.sentence_event_fusion = SentenceEventFusionLayer(config)

        if self.pruned_intermediate_sizes is not None:
            self._restore_pruned_layers(self.pruned_intermediate_sizes)

        self._intermediate_outputs = {}
        self._intermediate_grads = {}
        self._ffn_hook_handles = []
        self._hooks_registered = False

        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, CRF):
            module.reset_parameters()
            return
        if isinstance(module, GateFusion):
            module.reset_parameters()
            return
        if isinstance(module, SentenceEventFusionLayer):
            module._reset_event_fusion_parameters()
            return
        super()._init_weights(module)

    def _rebuild_compressed_embedding_from_config(self, config):
        if config.compression_method != "gate":
            raise ValueError("TBEF checkpoints only support compression_method='gate'.")

        self.compressed_embedding = create_compressed_embedding(
            "gate", config.vocab_size, config.hidden_size
        )
        rank = config.tsvd_dim
        self.compressed_embedding.embedding = nn.Embedding(config.vocab_size, rank)
        self.compressed_embedding.projection = nn.Linear(rank, config.hidden_size, bias=False)
        self.compressed_embedding.gate = nn.Embedding(config.vocab_size, rank)
        self.compressed_embedding.gate_sparsity_lambda = getattr(config, "gate_sparsity_lambda", 0.0)
        self.compressed_embedding.is_decomposed = True
        self.tsvd_dim = rank
        logger.info("Rebuilt gated compressed embedding: tsvd_dim=%d", rank)

    def init_compression(self, method, tsvd_dim, gate_init_bias=2.197, gate_sparsity_lambda=1e-4):
        if method != "gate":
            raise ValueError("TBEF only supports compression_method='gate'.")

        self.compression_method = method
        self.tsvd_dim = tsvd_dim
        self.gate_sparsity_lambda = gate_sparsity_lambda

        weight = self.bert.embeddings.word_embeddings.weight.data.detach().float()
        self.compressed_embedding = create_compressed_embedding(
            method, self.config.vocab_size, self.config.hidden_size
        )
        self.compressed_embedding.decompose(
            weight,
            tsvd_dim,
            gate_init_bias=gate_init_bias,
            gate_sparsity_lambda=gate_sparsity_lambda,
        )
        self.to(weight.device)
        self.is_decomposed = True
        logger.info("Compression initialized: method=%s, tsvd_dim=%d", method, tsvd_dim)

    def _clear_ffn_hooks(self):
        for handle in self._ffn_hook_handles:
            handle.remove()
        self._ffn_hook_handles = []
        self._hooks_registered = False

    def _register_ffn_hooks(self):
        if self._hooks_registered:
            return

        self._intermediate_outputs.clear()
        self._intermediate_grads.clear()
        self._ffn_hook_handles = []

        for layer_idx, layer in enumerate(self.bert.encoder.layer):
            def make_hook(idx):
                def forward_hook(module, inputs, output):
                    self._intermediate_outputs[idx] = output.detach()
                    if output.requires_grad:
                        output.register_hook(
                            lambda grad, hook_idx=idx: self._intermediate_grads.__setitem__(
                                hook_idx, grad.detach()
                            )
                        )
                return forward_hook

            self._ffn_hook_handles.append(layer.intermediate.register_forward_hook(make_hook(layer_idx)))
        self._hooks_registered = True

    def compute_ffn_importance(self):
        scores = []
        for layer_idx in range(len(self.bert.encoder.layer)):
            if layer_idx not in self._intermediate_outputs or layer_idx not in self._intermediate_grads:
                scores.append(None)
                continue
            hidden = self._intermediate_outputs[layer_idx]
            grad = self._intermediate_grads[layer_idx]
            scores.append(torch.abs((hidden * grad).sum(dim=(0, 1))))
        self._intermediate_outputs.clear()
        self._intermediate_grads.clear()
        return scores

    def prune_ffn(self, prune_ratio):
        logger.info("Starting FFN pruning with prune_ratio=%.4f", prune_ratio)
        importance_scores = self.compute_ffn_importance()
        pruned_sizes = []

        for layer_idx, layer in enumerate(self.bert.encoder.layer):
            intermediate = layer.intermediate
            output = layer.output

            old_inter_weight = intermediate.dense.weight.data
            old_inter_bias = intermediate.dense.bias.data
            old_out_weight = output.dense.weight.data
            old_out_bias = output.dense.bias.data

            intermediate_size = old_inter_weight.shape[0]
            num_to_keep = max(1, int(intermediate_size * (1 - prune_ratio)))

            importance = importance_scores[layer_idx]
            if importance is None or importance.numel() != intermediate_size:
                importance = torch.rand(intermediate_size, device=old_inter_weight.device)

            _, keep_indices = torch.topk(importance, num_to_keep)
            keep_indices, _ = torch.sort(keep_indices)

            new_intermediate = nn.Linear(old_inter_weight.shape[1], num_to_keep)
            new_intermediate.weight = nn.Parameter(old_inter_weight[keep_indices].clone())
            new_intermediate.bias = nn.Parameter(old_inter_bias[keep_indices].clone())

            new_output = nn.Linear(num_to_keep, old_out_weight.shape[0])
            new_output.weight = nn.Parameter(old_out_weight[:, keep_indices].clone())
            new_output.bias = nn.Parameter(old_out_bias.clone())

            intermediate.dense = new_intermediate
            output.dense = new_output
            pruned_sizes.append(num_to_keep)
            logger.info("Layer %d: %d -> %d neurons", layer_idx, intermediate_size, num_to_keep)

        self.pruned_intermediate_sizes = pruned_sizes
        self.config.pruned_intermediate_sizes = pruned_sizes
        self._clear_ffn_hooks()

    def _restore_pruned_layers(self, pruned_sizes):
        for layer_idx, new_size in enumerate(pruned_sizes):
            layer = self.bert.encoder.layer[layer_idx]
            current_size = layer.intermediate.dense.out_features
            if new_size >= current_size:
                continue
            hidden_size = layer.intermediate.dense.in_features
            layer.intermediate.dense = nn.Linear(hidden_size, new_size)
            layer.output.dense = nn.Linear(new_size, hidden_size)

    def _get_features(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if self.compressed_embedding is not None and input_ids is not None and inputs_embeds is None:
            word_embeds = self.compressed_embedding(input_ids)
            outputs = self.bert(
                inputs_embeds=word_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )

        sequence_output = outputs[0]
        if self.use_attention_pooling:
            sequence_output = self.attention_pooling(sequence_output, attention_mask)
        if self.use_sentence_event_fusion:
            sequence_output = self.sentence_event_fusion(sequence_output, attention_mask)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits, outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pad_token_label_id=None,
    ):
        logits, bert_outputs = self._get_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        extra_loss = None
        if labels is not None and self.compressed_embedding is not None:
            aux_loss = self.compressed_embedding.get_extra_loss()
            if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                extra_loss = aux_loss

        loss, best_path = _run_crf_decode(
            self,
            logits,
            attention_mask,
            labels,
            pad_token_label_id,
            extra_loss=extra_loss,
        )

        outputs = (logits,) + bert_outputs[2:]
        if loss is not None:
            return (loss,) + outputs + (best_path,)
        return outputs + (best_path,)

    def save_pretrained(self, save_directory, **kwargs):
        self.config.tsvd_dim = self.tsvd_dim
        self.config.is_decomposed = self.is_decomposed
        self.config.compression_method = self.compression_method
        self.config.gate_sparsity_lambda = self.gate_sparsity_lambda
        _save_attention_pooling_config(self)
        _save_sentence_event_fusion_config(self)
        if self.pruned_intermediate_sizes is not None:
            self.config.pruned_intermediate_sizes = self.pruned_intermediate_sizes
        super().save_pretrained(save_directory, **kwargs)
