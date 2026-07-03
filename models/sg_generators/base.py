"""Abstract base class for scene-graph generators.

Design contract
===============
An SGGenerator is a self-contained image -> scene-graph module. Its
interface to the rest of SSG-VQA-Net v2 is deliberately SYMBOLIC ONLY --
no appearance features cross the boundary. This is what makes the
detector genuinely swappable: the downstream scene-graph encoder consumes
only entity ids, region ids, positiveness ids, normalised bboxes, a
soft relation matrix, and an object count.

Forward contract
================

.. code-block:: python

    generator = get_sg_generator("txrv_detr", num_entities=22, num_regions=30)
    result = generator(images)          # images: (B, 3, H, W) float in [0, 1]
    # result: {
    #     "raw": {                       # For SG loss during training.
    #         "bbox_preds":         (B, N, 4)   in [0, 1], xyxy
    #         "entity_logits":      (B, N, num_entities)
    #         "region_logits":      (B, N, num_regions)
    #         "positiveness_logits": (B, N, 3)   {neg, pos, unknown}
    #         "relationship_logits": (B, N, N, num_relations)
    #         "objectness_scores":  (B, N)       in [0, 1]
    #     },
    #     "dicts": List[Dict[str, Any]] # For SceneGraphEncoderV2. See below.
    # }

Each dict in ``result["dicts"]`` has (all cpu/numpy where noted, GPU
tensor for ``relations`` because it retains grad in sg_only mode):

- ``bboxes``:        ndarray (n, 4), xyxy normalised in [0, 1]
- ``entity_ids``:    ndarray (n,)  int64 in [0, num_entities)
- ``region_ids``:    ndarray (n,)  int64 in [0, num_regions)
- ``positiveness``:  ndarray (n,)  int64 in {0=neg, 1=pos, 2=unknown}
- ``relations``:     tensor (n, n, num_relations)  softmaxed
- ``num_objects``:   int

Why the split? ``raw`` keeps GPU tensors + grad for MultiTaskLoss.
``dicts`` is decoded / thresholded / mostly-CPU numpy so downstream
non-tensor code paths work. Both come from the same forward pass so
they're consistent.

Standalone training
===================
When ``forward_train=True`` is passed with GT targets (``gt_bboxes``,
``gt_entities``, ``gt_regions``), the generator can additionally return
its own training loss so :file:`scripts/train_sg_generator.py` can
optimise it in isolation without instantiating any of the VLM code path.
"""
from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SGGenerator(nn.Module, abc.ABC):
    """Abstract scene-graph generator.

    Subclasses must implement:
    - ``forward(images, ...) -> {"raw": Dict, "dicts": List[Dict]}``
    - ``num_entities`` / ``num_regions`` properties (they may be inferred
      from ``__init__`` args, but must be exposed for the encoder's
      collapsed vocab and for cross-checks at load time).
    """

    num_entities: int
    num_regions: int
    num_relations: int

    def __init__(
        self,
        num_entities: int,
        num_regions: int,
        num_relations: int = 10,
    ) -> None:
        super().__init__()
        self.num_entities = int(num_entities)
        self.num_regions = int(num_regions)
        self.num_relations = int(num_relations)

    @abc.abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        *,
        objectness_threshold: float = 0.3,
        return_dicts: bool = True,
    ) -> Dict[str, Any]:
        """Run one forward pass on a batch of images."""

    # ------------------------------------------------------------------
    # Standalone-training helpers. Default impls raise so subclasses
    # opt in explicitly; a generator that only supports inference can
    # skip these.
    # ------------------------------------------------------------------
    def compute_training_loss(
        self,
        raw_outputs: Dict[str, torch.Tensor],
        gt_bboxes: List[torch.Tensor],
        gt_entities: List[torch.Tensor],
        gt_regions: List[torch.Tensor],
        gt_positiveness: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the SG training loss.

        Default: delegate to training.loss._compute_scene_graph_loss so
        every generator gets the Hungarian-matched DETR loss for free.
        Subclasses can override if they want a different objective.
        """
        # Delayed import to avoid a training.loss <-> models cycle.
        from training.loss import compute_scene_graph_loss_hungarian

        return compute_scene_graph_loss_hungarian(
            raw_outputs=raw_outputs,
            gt_bboxes=gt_bboxes,
            gt_entities=gt_entities,
            gt_regions=gt_regions,
            gt_positiveness=gt_positiveness,
            num_entities=self.num_entities,
            num_regions=self.num_regions,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers -- so the pluggable loader in SSGVQANetV2 can
    # load pretrained generator weights without knowing the class.
    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "num_entities": self.num_entities,
            "num_regions": self.num_regions,
            "num_relations": self.num_relations,
            "class_name": type(self).__name__,
        }, path)

    def load_weights(self, path: str, strict: bool = True) -> None:
        # weights_only=True is safe here -- we only save tensors + a handful of
        # ints and a short class-name string. weights_only=False would allow
        # arbitrary code execution during unpickling.
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Older checkpoints may not be weights_only compatible. Fall
            # back to the unsafe path ONLY with an explicit env opt-in.
            import os
            if os.environ.get("SG_GEN_TRUST_CHECKPOINT", "0") != "1":
                raise RuntimeError(
                    f"torch.load(..., weights_only=True) rejected {path}. "
                    "If you trust the origin of this checkpoint (e.g. your "
                    "own training output), re-run with "
                    "SG_GEN_TRUST_CHECKPOINT=1 to permit weights_only=False."
                )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Sanity-check shapes: a checkpoint trained on collapsed vocab
        # (22 entities) MUST NOT silently load into a 237-entity head.
        for k in ("num_entities", "num_regions", "num_relations"):
            saved = int(ckpt.get(k, getattr(self, k)))
            live = int(getattr(self, k))
            if saved != live:
                raise ValueError(
                    f"SG-generator checkpoint mismatch on {k}: "
                    f"saved={saved} vs live={live}. Set the correct "
                    f"vocab in the model config before loading."
                )
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Strip DDP prefix if present.
        if state and next(iter(state.keys())).startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
        self.load_state_dict(state, strict=strict)


# ==========================================================================
# Registry
# ==========================================================================
_REGISTRY: Dict[str, Callable[..., SGGenerator]] = {}


def register_sg_generator(name: str) -> Callable[[type], type]:
    """Decorator: @register_sg_generator("txrv_detr")."""
    def _wrap(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"SG generator '{name}' already registered")
        if not issubclass(cls, SGGenerator):
            raise TypeError(
                f"{cls.__name__} must subclass SGGenerator to be registered"
            )
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_sg_generator(name: str, **kwargs: Any) -> SGGenerator:
    """Instantiate a registered SG generator by name."""
    if name not in _REGISTRY:
        # Trigger imports so decorators run. Kept lazy so importing base
        # doesn't drag in torchxrayvision on machines that don't need it.
        _ensure_builtins_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown SG generator '{name}'. Registered: "
            f"{sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_sg_generators() -> List[str]:
    _ensure_builtins_loaded()
    return sorted(_REGISTRY.keys())


_BUILTINS_LOADED = False


def _ensure_builtins_loaded() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Import known implementations so their @register_sg_generator runs.
    # Wrapped in try/except because the modules have optional deps
    # (torchxrayvision, transformers Qwen). We register stubs for the
    # ones that aren't built yet so users can see them in error messages.
    try:
        from . import torchxrayvision_detr  # noqa: F401
    except Exception:
        pass
    try:
        from . import qwen_vit_rpn  # noqa: F401
    except Exception:
        pass
    # Stubs -- registered even without an implementation module so users
    # who try `--generator chest_imagenome` get a clear NotImplementedError
    # instead of "unknown name".
    for stub_name in ("chest_imagenome", "detr_sg", "dino_sg"):
        if stub_name not in _REGISTRY:
            _REGISTRY[stub_name] = _make_stub(stub_name)
    _BUILTINS_LOADED = True


def _make_stub(name: str) -> Callable[..., SGGenerator]:
    def _stub(**_kwargs: Any) -> SGGenerator:
        raise NotImplementedError(
            f"SG generator '{name}' is registered but not implemented "
            f"in this branch. Add models/sg_generators/{name}.py and "
            f"decorate a subclass of SGGenerator with "
            f"@register_sg_generator('{name}')."
        )
    return _stub
