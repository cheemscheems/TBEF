import torch
import torch.nn as nn


class CRF(nn.Module):
    """Linear-chain CRF with START/STOP states stored as the final two tags."""

    def __init__(self, tagset_size):
        super().__init__()
        self.START_TAG = -2
        self.STOP_TAG = -1
        self.tagset_size = tagset_size

        transitions = torch.empty(tagset_size + 2, tagset_size + 2)
        self.transitions = nn.Parameter(transitions)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.transitions.zero_()
            self.transitions[:, self.START_TAG] = -10000.0
            self.transitions[self.STOP_TAG, :] = -10000.0

    def _constrain_emissions(self, feats):
        feats = feats.clone()
        feats[..., self.START_TAG] = -10000.0
        feats[..., self.STOP_TAG] = -10000.0
        return feats

    def _calculate_partition(self, feats, mask):
        feats = self._constrain_emissions(feats)
        batch_size, seq_len, tag_size = feats.shape
        assert tag_size == self.tagset_size + 2

        mask = mask.bool()
        alpha = self.transitions[self.START_TAG].unsqueeze(0) + feats[:, 0]
        for idx in range(1, seq_len):
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + feats[:, idx].unsqueeze(1)
            )
            next_alpha = torch.logsumexp(scores, dim=1)
            alpha = torch.where(mask[:, idx].unsqueeze(1), next_alpha, alpha)

        terminal_scores = alpha + self.transitions[:, self.STOP_TAG].unsqueeze(0)
        return torch.logsumexp(terminal_scores, dim=1).sum()

    def _score_sentence(self, feats, mask, tags):
        feats = self._constrain_emissions(feats)
        batch_size, seq_len, _ = feats.shape
        mask = mask.bool()

        score = feats.new_zeros(batch_size)
        first_tags = tags[:, 0]
        score += self.transitions[self.START_TAG, first_tags]
        score += feats[torch.arange(batch_size, device=feats.device), 0, first_tags]

        for idx in range(1, seq_len):
            active = mask[:, idx]
            prev_tags = tags[:, idx - 1]
            curr_tags = tags[:, idx]
            step_score = self.transitions[prev_tags, curr_tags]
            step_score += feats[torch.arange(batch_size, device=feats.device), idx, curr_tags]
            score += step_score * active.to(feats.dtype)

        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score += self.transitions[last_tags, self.STOP_TAG]
        return score.sum()

    def _viterbi_decode(self, feats, mask):
        feats = self._constrain_emissions(feats)
        batch_size, seq_len, tag_size = feats.shape
        assert tag_size == self.tagset_size + 2

        mask = mask.bool()
        alpha = self.transitions[self.START_TAG].unsqueeze(0) + feats[:, 0]
        history = []

        for idx in range(1, seq_len):
            scores = (
                alpha.unsqueeze(2)
                + self.transitions.unsqueeze(0)
                + feats[:, idx].unsqueeze(1)
            )
            next_alpha, backpointers = torch.max(scores, dim=1)
            alpha = torch.where(mask[:, idx].unsqueeze(1), next_alpha, alpha)
            history.append(backpointers)

        terminal_scores = alpha + self.transitions[:, self.STOP_TAG].unsqueeze(0)
        _, best_last_tags = torch.max(terminal_scores, dim=1)

        decoded = torch.zeros(batch_size, seq_len, dtype=torch.long, device=feats.device)
        lengths = mask.long().sum(dim=1).clamp_min(1)
        for batch_idx in range(batch_size):
            seq_length = int(lengths[batch_idx].item())
            pointer = best_last_tags[batch_idx]
            decoded[batch_idx, seq_length - 1] = pointer
            for pos in range(seq_length - 1, 0, -1):
                pointer = history[pos - 1][batch_idx, pointer]
                decoded[batch_idx, pos - 1] = pointer
        return None, decoded

    def forward(self, feats, mask):
        _, best_path = self._viterbi_decode(feats, mask)
        return best_path

    def neg_log_likelihood(self, feats, mask, tags):
        partition = self._calculate_partition(feats, mask)
        gold_score = self._score_sentence(feats, mask, tags)
        return partition - gold_score
