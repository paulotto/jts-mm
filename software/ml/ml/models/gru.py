# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""GRU sequence classifier (PyTorch) on sub-frame features ``(n, S, d)``.

Torch is imported lazily: the registry lists ``gru`` without torch installed and
``ModelSpec.available()`` reports it as unavailable. Training uses AdamW,
class-weighted cross-entropy and early stopping on validation macro-F1 with the
validation rows split by groups (subjects/sessions/blocks), restoring the best
weights. The fitted model pickles (``model.joblib``) with its weights on the CPU.
"""

from __future__ import annotations

import io
import sys
from typing import Any

import numpy as np

from ml.config.classes import CLASS_COUNT
from ml.evaluation.metrics import macro_f1
from ml.fusion.early import BlockScaler
from ml.models.base import _Base, align_proba, grouped_val_split


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("the GRU needs PyTorch: pip install -e './software/ml[torch]'") from exc
    return torch


def resolve_device(name: str = "auto"):
    torch = require_torch()
    name = (name or "auto").lower()
    if name == "auto" or (name.startswith("cuda") and not torch.cuda.is_available()):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def balanced_class_weights(y: np.ndarray, n_classes: int = CLASS_COUNT) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(float)
    w = np.zeros(n_classes)
    present = counts > 0
    if present.any():
        w[present] = counts.sum() / (present.sum() * counts[present])
    return w


def _module(input_size: int, hidden: int, layers: int, dropout: float, bidirectional: bool,
            readout: str, n_classes: int):
    torch = require_torch()

    class GruNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.readout = readout
            self.bidirectional = bidirectional
            self.gru = torch.nn.GRU(input_size, hidden, num_layers=layers, batch_first=True,
                                    dropout=dropout if layers > 1 else 0.0,
                                    bidirectional=bidirectional)
            self.fc = torch.nn.Linear(hidden * (2 if bidirectional else 1), n_classes)

        def forward(self, x):
            out, h = self.gru(x)
            if self.readout == "last":
                feat = torch.cat([h[-2], h[-1]], dim=-1) if self.bidirectional else h[-1]
            else:
                feat = out.mean(dim=1)
            return self.fc(feat)

    return GruNet()


class GruClassifier(_Base):
    input = "sequences"

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2,
                 bidirectional: bool = False, readout: str = "mean_pool", batch_size: int = 128,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 100,
                 patience: int = 10, grad_clip: float = 1.0, device: str = "cpu",
                 progress: bool = False, seed: int = 42, slices: list[slice] | None = None,
                 probability: bool = False) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.readout = readout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device
        self.progress = progress
        self.seed = seed
        self.slices = slices
        self.probability = probability

    # -- training ------------------------------------------------------------------------
    def fit(self, X, y, groups=None):
        torch = require_torch()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"GRU expects sequences (n, S, d), got shape {X.shape}")
        if self.readout not in ("mean_pool", "last"):
            raise ValueError(f"readout must be mean_pool|last, got {self.readout!r}")
        torch.manual_seed(self.seed)
        y_raw = np.asarray(y, dtype=np.int64)
        # The network outputs only the classes present in training (sklearn convention);
        # predict_proba maps them back onto the 7-class axis.
        self.classes_ = np.unique(y_raw)
        y = np.searchsorted(self.classes_, y_raw)
        k = int(self.classes_.size)
        self.scaler_ = BlockScaler(self.slices).fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        split = grouped_val_split(groups, self.seed)
        tr, va = split if split is not None else (np.arange(len(y)), None)
        dev = resolve_device(self.device)
        net = _module(Xs.shape[-1], int(self.hidden_size), int(self.num_layers),
                      float(self.dropout), bool(self.bidirectional), self.readout, k).to(dev)
        crit = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(
            balanced_class_weights(y[tr], k), dtype=torch.float32, device=dev))
        opt = torch.optim.AdamW(net.parameters(), lr=float(self.learning_rate),
                                weight_decay=float(self.weight_decay))
        gen = torch.Generator().manual_seed(self.seed)
        Xt = torch.as_tensor(Xs[tr])
        yt = torch.as_tensor(y[tr])
        bs = max(1, min(int(self.batch_size), len(tr)))
        best, best_state, best_epoch, wait, history = -np.inf, None, 0, 0, []
        for epoch in range(1, int(self.max_epochs) + 1):
            net.train()
            perm = torch.randperm(len(tr), generator=gen)
            loss_sum = 0.0
            for i in range(0, len(tr), bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = crit(net(Xt[b].to(dev)), yt[b].to(dev))
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), float(self.grad_clip))
                opt.step()
                loss_sum += float(loss.item()) * len(b)
            score = (macro_f1(y[va], self._logits(net, Xs[va], dev).argmax(1)) if va is not None
                     else -loss_sum / len(tr))
            history.append({"epoch": epoch, "train_loss": loss_sum / len(tr), "val": float(score)})
            if self.progress:
                sys.stderr.write(f"\r  gru epoch {epoch}/{self.max_epochs} val={score:.3f}")
            if score > best + 1e-6:
                best, best_epoch, wait = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= int(self.patience):
                    break
        if self.progress:
            sys.stderr.write("\n")
        net.load_state_dict(best_state)
        self.net_ = net.cpu().eval()
        self.fit_info_ = {"early_stopping": "grouped_val_macro_f1" if va is not None
                          else "train_loss", "best_epoch": best_epoch, "epochs": len(history),
                          "best_score": float(best)}
        return self

    @staticmethod
    def _logits(net, X: np.ndarray, dev, batch: int = 1024) -> np.ndarray:
        torch = require_torch()
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                out.append(net(torch.as_tensor(X[i:i + batch], dtype=torch.float32).to(dev))
                           .cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, net.fc.out_features))

    def predict_proba(self, X):
        Xs = self.scaler_.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)
        z = self._logits(self.net_, Xs, resolve_device("cpu"))
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return align_proba(e / e.sum(axis=1, keepdims=True), self.classes_)

    # -- pickling: keep the network as a CPU state dict -----------------------------------
    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        net = state.pop("net_", None)
        if net is not None:
            torch = require_torch()
            buf = io.BytesIO()
            torch.save(net.state_dict(), buf)
            state["_net_bytes"] = buf.getvalue()
            state["_net_input"] = int(net.gru.input_size)
            state["_net_output"] = int(net.fc.out_features)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        blob = state.pop("_net_bytes", None)
        d_in = state.pop("_net_input", None)
        d_out = state.pop("_net_output", CLASS_COUNT)
        self.__dict__.update(state)
        if blob is not None:
            torch = require_torch()
            net = _module(d_in, int(self.hidden_size), int(self.num_layers), float(self.dropout),
                          bool(self.bidirectional), self.readout, d_out)
            net.load_state_dict(torch.load(io.BytesIO(blob), map_location="cpu"))
            self.net_ = net.eval()
