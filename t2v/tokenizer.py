"""Flat parameter vector <-> sequence of fixed-size tokens (RPG's layer-wise tokenisation).

Body tensors are normalised per tensor (mean/std over the whole zoo) and cut into `token_size`
chunks. The classifier is tokenised differently: one token per class row ([W_i, b_i]), so each
row can be tied to the text embedding of its class. That makes class permutation (Tina's key
augmentation) a permutation of tokens.
"""
import math

import torch

from .target import HEAD


class ParamTokenizer:
    def __init__(self, spec, token_size, stats, max_classes):
        self.spec, self.token_size, self.stats, self.K = spec, token_size, stats, max_classes
        offsets, o = {}, 0
        for name, shape in spec:
            offsets[name] = (o, shape)
            o += int(torch.Size(shape).numel())
        self.num_params = o

        idx, mean, std, layer_ids, intra_ids, slots = [], [], [], [], [], []

        elem_mean = stats.get("__elem_mean__")  # per-element zoo mean for body weights (norm="element"/"zscore")
        elem_std = stats.get("__elem_std__")    # per-element zoo spread (norm="zscore")
        if elem_mean is not None:
            elem_mean = elem_mean.cpu()  # checkpoints may be loaded straight onto the GPU
        if elem_std is not None:
            elem_std = elem_std.cpu()

        def add(positions, name, layer, intra, slot):
            pad = token_size - len(positions)
            idx.append(torch.cat([positions, torch.full((pad,), -1, dtype=torch.long)]))
            if elem_mean is not None and slot < 0:
                mean.append(torch.cat([elem_mean[positions].float(), torch.zeros(pad)]))
            else:
                mean.append(torch.full((token_size,), stats[name][0]))
            if elem_std is not None and slot < 0:
                std.append(torch.cat([elem_std[positions].float(), torch.ones(pad)]))
            else:
                std.append(torch.full((token_size,), stats[name][1]))
            layer_ids.append(layer), intra_ids.append(intra), slots.append(slot)

        body = [n for n, _ in spec if n not in HEAD]
        for li, name in enumerate(body):
            start, shape = offsets[name]
            n = int(torch.Size(shape).numel())
            for j in range(math.ceil(n / token_size)):
                lo, hi = j * token_size, min(n, (j + 1) * token_size)
                add(torch.arange(start + lo, start + hi), name, li, j, -1)

        (w0, (k, feat)), (b0, _) = offsets[HEAD[0]], offsets[HEAD[1]]
        assert k == max_classes and feat + 1 <= token_size, "classifier row must fit in one token"
        head_pos = []
        for i in range(k):
            head_pos.append(len(slots))
            row = torch.cat([torch.arange(w0 + i * feat, w0 + (i + 1) * feat), torch.tensor([b0 + i])])
            add(row, HEAD[0], len(body), 0, i)
            # the bias entry uses the bias statistics
            mean[-1][feat], std[-1][feat] = stats[HEAD[1]]

        self.idx = torch.stack(idx)                 # (T, D) index into the flat vector, -1 = padding
        self.valid = self.idx >= 0                  # (T, D)
        self.mean, self.std = torch.stack(mean), torch.stack(std)
        self.layer_ids = torch.tensor(layer_ids)
        self.intra_ids = torch.tensor(intra_ids)
        self.slots = torch.tensor(slots)            # class slot for head rows, -1 for body tokens
        self.head_pos = torch.tensor(head_pos)      # token positions of the K head rows
        self.num_tokens = len(slots)

    @staticmethod
    def compute_stats(flat, spec, norm="tensor", eps=1e-6):
        """norm="tensor": one mean/std per tensor (RPG). norm="element": body weights are expressed as a
        deviation from the zoo's mean model (scaled per tensor), so the generator spends its capacity on
        what differs between tasks. norm="zscore": the same deviation, scaled by each weight's own spread
        across the zoo, so a generation error of a given size is equally harmless for every weight (deep
        BatchNorm nets such as ResNet collapse once errors exceed ~2x the zoo spread). The classifier keeps
        per-tensor stats because its rows follow the classes, not fixed positions."""
        stats, o = {}, 0
        mean_model = flat.float().mean(0)
        if norm in ("element", "zscore"):
            stats["__elem_mean__"] = mean_model
        if norm == "zscore":
            stats["__elem_std__"] = (flat.float() - mean_model).std(0).clamp_min(1e-8)
        for name, shape in spec:
            n = int(torch.Size(shape).numel())
            t = flat[:, o:o + n].float()
            if norm in ("element", "zscore") and name not in HEAD:
                stats[name] = (0.0, max((t - mean_model[o:o + n]).std().item(), eps))
            else:
                stats[name] = (t.mean().item(), max(t.std().item(), eps))
            o += n
        return stats

    def to(self, device):
        for k in ("idx", "valid", "mean", "std", "layer_ids", "intra_ids", "slots", "head_pos"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def encode(self, flat):
        """(B, P) -> (B, T, D) normalised tokens, zero in padding."""
        x = flat.float()[:, self.idx.clamp_min(0)]
        return torch.where(self.valid, (x - self.mean) / self.std, torch.zeros((), device=x.device))

    def decode(self, tokens, num_classes=None):
        """(B, T, D) -> (B, P). Head rows beyond `num_classes` are zeroed."""
        x = tokens.float() * self.std + self.mean
        if num_classes is not None:
            unused = (self.slots >= num_classes)[None, :, None]
            x = torch.where(unused, torch.zeros((), device=x.device), x)
        flat = torch.zeros(tokens.shape[0], self.num_params, device=tokens.device)
        flat[:, self.idx[self.valid]] = x[:, self.valid]
        return flat

    def state(self):
        return {"spec": self.spec, "token_size": self.token_size, "stats": self.stats, "max_classes": self.K}

    @classmethod
    def from_state(cls, s):
        return cls(s["spec"], s["token_size"], s["stats"], s["max_classes"])
