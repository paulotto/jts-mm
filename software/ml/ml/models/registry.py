# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Model registry: the only place that names models.

Every model is a scikit-learn compatible classifier with

* ``fit(X, y, groups=None)``: ``groups`` (subjects under LOSO, sessions under
  leave-one-session-out, blocks under within-subject CV) are used for every
  internal split -- SVM probability calibration and MLP/GRU early stopping --
  so overlapping windows never sit on both sides of a split;
* ``predict(X)`` and ``predict_proba(X)``, the latter on the fixed 7-class axis;
* a :class:`~ml.fusion.early.BlockScaler` fit on the training data, so scaling
  is always per fold.

``get(name)`` also builds late-fusion models: ``late_mean:<base>``,
``late_weighted:<base>`` and ``late_stack:<base>`` (:mod:`ml.fusion.late`).
"""

from __future__ import annotations

import copy
import importlib.util
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ml.evaluation.metrics import macro_f1
from ml.fusion.early import BlockScaler
from ml.models import fast_svm
from ml.models.base import _Base, align_proba, grouped_val_split

# libsvm kernel cache per fit (MB; sklearn default 200): speed only, same solution.
SVC_CACHE_MB = 1000.0


class LdaClassifier(_Base):
    """Shrinkage LDA. ``priors="uniform"`` (default) gives every class of the training
    data the same prior, so the REST-dominated window counts do not bias decisions
    towards REST (the balanced counterpart of the SVM's ``class_weight="balanced"``);
    ``"empirical"`` keeps sklearn's class frequencies."""

    def __init__(self, shrinkage: str | float = "auto", priors: str = "uniform", seed: int = 42,
                 slices: list[slice] | None = None, probability: bool = False) -> None:
        self.shrinkage = shrinkage
        self.priors = priors
        self.seed = seed
        self.slices = slices
        self.probability = probability

    def fit(self, X, y, groups=None):
        if self.priors not in ("uniform", "empirical"):
            raise ValueError(f"LDA priors must be 'uniform' or 'empirical', got {self.priors!r}")
        k = np.unique(y).size
        priors = np.full(k, 1.0 / k) if self.priors == "uniform" else None
        self.pipe_ = Pipeline([("scale", BlockScaler(self.slices)),
                               ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                                  shrinkage=self.shrinkage,
                                                                  priors=priors))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipe_.fit(X, y)
        self.classes_ = self.pipe_.classes_
        return self

    def predict(self, X):
        return self.pipe_.predict(X)

    def predict_proba(self, X):
        return align_proba(self.pipe_.predict_proba(X), self.classes_)


class SvmClassifier(_Base):
    """RBF SVM. Probabilities (only when ``probability``) come from sigmoid calibration
    on *grouped* out-of-fold decision values; with fewer than two groups there is no
    leakage-free calibration and a softmax of the decision function is used instead
    (``proba_method_`` records which).

    ``gamma_mult`` scales the kernel width: with ``gamma="scale"`` the fitted
    ``gamma_ = gamma_mult / (n_features * Var(X_scaled))`` is computed on the training
    fold after :class:`BlockScaler` (``gamma_mult=1`` is sklearn's ``"scale"``); with a
    numeric ``gamma`` it multiplies that value.

    ``fit(..., sample_weight=w)`` passes per-row weights to the SVC (they multiply the
    ``class_weight``); weighted fits use no grouped calibration (``predict_proba`` is then
    the softmax of the decision function). Used by the few-shot variant
    (:mod:`ml.protocols.variants`)."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale", gamma_mult: float = 1.0,
                 kernel: str = "rbf", class_weight: str | None = "balanced", seed: int = 42,
                 slices: list[slice] | None = None, probability: bool = False) -> None:
        self.C = C
        self.gamma = gamma
        self.gamma_mult = gamma_mult
        self.kernel = kernel
        self.class_weight = class_weight
        self.seed = seed
        self.slices = slices
        self.probability = probability

    def _gamma(self, X) -> float | str:
        k = float(self.gamma_mult)
        if self.gamma == "auto":
            return k / X.shape[-1] if k != 1.0 else "auto"
        if self.gamma == "scale":
            Xs = BlockScaler(self.slices).fit_transform(X)
            v = float(Xs.var())
            return k / (Xs.shape[-1] * v) if v > 0 else k
        return k * float(self.gamma)

    def _pipe(self) -> Pipeline:
        return Pipeline([("scale", BlockScaler(self.slices)),
                         ("clf", SVC(C=float(self.C), gamma=self.gamma_, kernel=self.kernel,
                                     class_weight=self.class_weight, random_state=self.seed,
                                     cache_size=SVC_CACHE_MB))])

    def fit(self, X, y, groups=None, sample_weight=None):
        self.gamma_ = self._gamma(np.asarray(X, dtype=np.float64))
        self.proba_method_ = "none"
        self.calibrator_ = None
        n_groups = 0 if groups is None else np.unique(groups).size
        if self.probability and n_groups >= 2 and sample_weight is None:
            splits = list(GroupKFold(n_splits=min(3, n_groups)).split(X, y, groups))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cal = CalibratedClassifierCV(self._pipe(), method="sigmoid", cv=splits,
                                                 ensemble=False).fit(X, y)
                self.calibrator_ = cal
                self.pipe_ = cal.calibrated_classifiers_[0].estimator
                self.proba_method_ = "sigmoid_grouped_cv"
            except ValueError:                       # a class missing from a calibration fold
                self.calibrator_ = None
        if self.calibrator_ is None:
            kw = {} if sample_weight is None else {
                "clf__sample_weight": np.asarray(sample_weight, dtype=np.float64)}
            self.pipe_ = self._pipe().fit(X, y, **kw)
            if self.probability:
                self.proba_method_ = "softmax_decision"
        self.classes_ = self.pipe_.classes_
        return self

    def predict(self, X):
        # Batched exact path (ml.models.fast_svm): the labels of SVC.predict, faster.
        if fast_svm.enabled() and len(X) >= fast_svm.MIN_ROWS:
            svc = self.pipe_.steps[-1][1]
            if fast_svm.supported(svc):
                return fast_svm.predict(svc, self.pipe_.steps[0][1].transform(X))
        return self.pipe_.predict(X)

    def predict_proba(self, X):
        if self.calibrator_ is not None:
            return align_proba(self.calibrator_.predict_proba(X), self.calibrator_.classes_)
        s = np.asarray(self.pipe_.decision_function(X), dtype=np.float64)
        if s.ndim == 1:
            s = np.column_stack([-s, s])
        e = np.exp(s - s.max(axis=1, keepdims=True))
        return align_proba(e / e.sum(axis=1, keepdims=True), self.classes_)


class MlpClassifier(_Base):
    """Feed-forward network (sklearn). Early stopping on validation macro-F1 with the
    validation rows split by *groups* (sklearn's own ``early_stopping`` splits windows at
    random, which leaks overlapping windows); without groups it trains ``max_epochs``
    epochs and stops on the training loss plateau."""

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (64, 32), alpha: float = 1e-4,
                 learning_rate_init: float = 1e-3, batch_size: int = 200, max_epochs: int = 300,
                 patience: int = 10, seed: int = 42, slices: list[slice] | None = None,
                 probability: bool = False) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.slices = slices
        self.probability = probability

    def fit(self, X, y, groups=None):
        self.scaler_ = BlockScaler(self.slices).fit(X)
        Xs = self.scaler_.transform(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        clf = MLPClassifier(hidden_layer_sizes=tuple(int(h) for h in self.hidden_layer_sizes),
                            alpha=float(self.alpha),
                            learning_rate_init=float(self.learning_rate_init),
                            batch_size=min(int(self.batch_size), len(y)), random_state=self.seed,
                            max_iter=int(self.max_epochs), n_iter_no_change=int(self.patience))
        split = grouped_val_split(groups, self.seed)
        self.fit_info_ = {"early_stopping": "none"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if split is None:
                clf.fit(Xs, y)
                self.fit_info_ = {"early_stopping": "train_loss", "epochs": int(clf.n_iter_)}
            else:
                tr, va = split
                best, best_state, best_epoch, wait = -np.inf, None, 0, 0
                for epoch in range(1, int(self.max_epochs) + 1):
                    clf.partial_fit(Xs[tr], y[tr], classes=self.classes_)
                    s = macro_f1(y[va], clf.predict(Xs[va]))
                    if s > best + 1e-6:
                        best, best_epoch, wait = s, epoch, 0
                        best_state = (copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_))
                    else:
                        wait += 1
                        if wait >= int(self.patience):
                            break
                clf.coefs_, clf.intercepts_ = best_state
                self.fit_info_ = {"early_stopping": "grouped_val_macro_f1",
                                  "best_epoch": best_epoch,
                                  "best_val_macro_f1": float(best), "epochs": epoch}
        self.clf_ = clf
        return self

    def predict(self, X):
        return self.clf_.predict(self.scaler_.transform(X))

    def predict_proba(self, X):
        return align_proba(self.clf_.predict_proba(self.scaler_.transform(X)), self.clf_.classes_)


def _gru_factory(**kw: Any):
    from ml.models.gru import GruClassifier

    return GruClassifier(**kw)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    grid: dict[str, list[Any]] = field(default_factory=dict)
    input: str = "features"                 # features (n, d) | sequences (n, S, d)
    requires: tuple[str, ...] = ()

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.requires)

    def make(self, seed: int = 42, *, slices: list[slice] | None = None,
             probability: bool = False, **params: Any):
        if not self.available():
            raise ImportError(f"model {self.name!r} needs {', '.join(self.requires)}; "
                              "install with: pip install -e './software/ml[torch]'")
        return self.factory(seed=seed, slices=slices, probability=probability, **params)


REGISTRY: dict[str, ModelSpec] = {
    "lda": ModelSpec("lda", LdaClassifier),
    # C: 1 (then the lowest value) won 59/60 folds; gamma was never tuned. Gamma is a
    # multiple of the training fold's "scale" value (see SvmClassifier).
    "svm": ModelSpec("svm", SvmClassifier, {"C": [0.01, 0.1, 1.0, 10.0],
                                            "gamma_mult": [0.25, 1.0, 4.0]}),
    "mlp": ModelSpec("mlp", MlpClassifier, {"alpha": [1e-4, 1e-2]}),
    "gru": ModelSpec("gru", _gru_factory, {}, input="sequences", requires=("torch",)),
}
LATE_METHODS = ("late_mean", "late_weighted", "late_stack")


def names(*, available_only: bool = False) -> list[str]:
    """Registered base model names (late-fusion names are built on demand)."""
    return [n for n, s in REGISTRY.items() if s.available() or not available_only]


def get(name: str) -> ModelSpec:
    if name in REGISTRY:
        return REGISTRY[name]
    method, _, base = name.partition(":")
    if method in LATE_METHODS and base in REGISTRY:
        from ml.fusion.late import LateFusionClassifier

        b = REGISTRY[base]

        def factory(**kw: Any):
            return LateFusionClassifier(base=base, method=method.split("_", 1)[1], **kw)

        return ModelSpec(name, factory, {}, input=b.input, requires=b.requires)
    raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)} "
                   f"or {'|'.join(LATE_METHODS)}:<model>")
