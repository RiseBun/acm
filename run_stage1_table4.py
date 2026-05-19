"""Stage 1 minimum imagination-RL pipeline.

Trains + evaluates one condition (or ``all`` / ``pilot``) on the same
nuScenes split as Stage 0 and emits Table 4 metrics (latent return,
imagined collision rate, rollout stability).

Condition -> representation mapping (from docs/stage1_design.md section 3):

    bc           : object_only       , no imagination, action MSE
    ac1          : object_only       , 1-step TD AC (WM detached)
    wm_holistic  : holistic_16slot   , K-step imagination AC
    wm_object    : object_only       , K-step imagination AC
    wm_naive     : object_relation   , K-step imagination AC
    wm_decoupled : object_relation_decoupled_visibility, K-step AC
    wm_decoupled_rel_to_critic_only:
                   same decoupled slots, dyn-only actor, dyn+rel critic

Each condition can optionally warm-start from the matching Stage 0
checkpoint under ``--stage0-root``, so we don't have to re-learn
representation from scratch.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.evaluation.stage1_metrics import evaluate_stage1
from doorrl.imagination.task_reward import TaskRewardCfg
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.training.losses_stage1 import Stage1LossCfg
from doorrl.training.trainer_stage1 import ImaginationTrainer, Stage1Cfg
from doorrl.utils import set_seed


_CONDITIONS = {
    "bc":                  {"variant": "object_only",                          "cond": "bc"},
    "ac1":                 {"variant": "object_only",                          "cond": "ac1"},
    "wm_holistic":         {"variant": "holistic_16slot",                      "cond": "wm"},
    "wm_set_pma":          {"variant": "set_transformer_pma",                  "cond": "wm"},
    "wm_object":           {"variant": "object_only",                          "cond": "wm"},
    "wm_naive":            {"variant": "object_relation",                      "cond": "wm"},
    # With visibility weighting on the dynamic path (default decoupled variant
    # reported in Stage 0).
    "wm_decoupled":        {"variant": "object_relation_decoupled_visibility", "cond": "wm"},
    # Ablation for the Stage-1 "Y" question: is visibility weighting the root
    # cause of the decoupled variant's higher imagined-collision rate? Same
    # decoupled abstraction, no visibility multiplicaiton on the dyn path.
    "wm_decoupled_no_vis": {"variant": "object_relation_decoupled",            "cond": "wm"},
    # DOOR+: same typed dynamic/relation budgets, but relation selector reads
    # optional confidence/uncertainty token fields and gates low-confidence
    # relation evidence. Warm-starts from the decoupled no-vis Stage-0 model.
    "wm_doorplus_uncertainty": {
        "variant": "object_relation_decoupled_uncertainty",
        "cond": "wm",
        "warm_start_variant": "object_relation_decoupled",
    },
    # Stage-1 typed-budget ablation. Same architecture as wm_decoupled, but
    # with --top-k-dyn 14 --top-k-rel 2 (vs default 12+4). The aim is to
    # test whether the rel budget is the destabiliser revealed by X.
    "wm_decoupled_14_2":   {"variant": "object_relation_decoupled_visibility", "cond": "wm"},
    # Fusion ablation: keep relation slots in the world model and critic
    # value input, but keep the actor policy mean on the dynamic branch only.
    "wm_decoupled_rel_to_critic_only": {
        "variant": "object_relation_decoupled_visibility",
        "cond": "wm",
        "rel_to_critic_only": True,
    },
}

_PILOT_CONDITIONS = ["bc", "wm_object", "wm_decoupled"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--dataset", type=str, choices=["nuscenes", "nuplan"],
                        default="nuscenes",
                        help="Data source. 'nuplan' uses preprocessed NPZ.")
    parser.add_argument("--nuscenes-root", type=str,
                        default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument(
        "--token-cache-dir", type=str, default=str(ROOT / "experiments" / "_token_cache"),
        help="Where to cache pre-tokenised nuScenes scenes (sha1-keyed). "
             "Re-uses 18 min of single-threaded tokenisation across runs. "
             "Pass '' to disable.",
    )
    parser.add_argument("--nuplan-root", type=str,
                        default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split")
    parser.add_argument("--nuplan-num-samples", type=int, default=5000)
    parser.add_argument("--nuplan-index-json", type=str, default=None)
    parser.add_argument(
        "--nuplan-workers", type=int, default=0,
        help="Parallel workers for nuPlan NPZ tokenisation. 0/1 = serial.",
    )
    parser.add_argument(
        "--nuplan-lazy", action="store_true",
        help="Do not pre-materialise nuPlan samples; convert NPZ in DataLoader workers.",
    )
    parser.add_argument(
        "--loader-workers", type=int, default=2,
        help="PyTorch DataLoader workers. Increase for --nuplan-lazy.",
    )
    parser.add_argument(
        "--condition", type=str, default="pilot",
        choices=list(_CONDITIONS.keys()) + ["pilot", "all"],
        help="pilot = {bc, wm_object, wm_decoupled}; all = all six.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-scenes", type=int, default=700)
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-scale", type=float, default=4.0)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--output-dir", type=str,
                        default=str(ROOT / "experiments" / "stage1"))
    parser.add_argument(
        "--stage0-root", type=str,
        default=str(ROOT / "experiments" / "table3_fair_fix2_seed7"),
        help="Per-variant warm-start checkpoint dir. Set to '' to disable.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; only run evaluation on the saved Stage 1 ckpt.",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze encoder + abstraction; train only WM + policy.",
    )
    # --- Policy-shape knobs for v3 ablation: tame the `wm_object` actor ---
    parser.add_argument(
        "--entropy-beta", type=float, default=0.01,
        help="Entropy bonus weight in the actor loss. v2 pilot used 0.01.",
    )
    parser.add_argument(
        "--action-sample-clip", type=float, default=8.0,
        help="Hard cap on sampled action magnitude. v2 pilot used 8.0.",
    )
    # --- Capacity and typed-budget overrides ----------------------------
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Override config.model.top_k for capacity-scaling experiments.",
    )
    parser.add_argument(
        "--top-k-dyn", type=int, default=None,
        help="Override config.model.top_k_dyn (decoupled variants only). "
             "Default keeps the value from the config file (12).",
    )
    parser.add_argument(
        "--top-k-rel", type=int, default=None,
        help="Override config.model.top_k_rel (decoupled variants only). "
             "Default keeps the value from the config file (4). Note: must "
             "satisfy top_k_dyn + top_k_rel == config.model.top_k.",
    )
    parser.add_argument(
        "--doorplus-train-relfp-rate", type=float, default=0.0,
        help="For wm_doorplus_uncertainty only: inject low-confidence false-positive "
             "relation tokens during Stage-1 training.",
    )
    parser.add_argument(
        "--doorplus-fp-confidence", type=float, default=0.05,
        help="Confidence written to injected false-positive relation tokens.",
    )
    parser.add_argument(
        "--doorplus-safe-confidence", type=float, default=1.0,
        help="Confidence written to non-corrupted relation tokens during augmentation.",
    )
    return parser.parse_args()


def _relation_mask(batch: SceneBatch) -> torch.Tensor:
    return (batch.token_types == int(TokenType.RELATION)) & batch.token_mask


def _copy_batch(batch: SceneBatch, tokens: torch.Tensor) -> SceneBatch:
    return SceneBatch(
        tokens=tokens,
        token_mask=batch.token_mask,
        token_types=batch.token_types,
        actions=batch.actions,
        next_tokens=batch.next_tokens,
        rewards=batch.rewards,
        continues=batch.continues,
    )


def _apply_doorplus_train_relfp(
    batch: SceneBatch,
    rate: float,
    fp_confidence: float,
    safe_confidence: float,
) -> SceneBatch:
    if rate <= 0.0:
        return batch
    tokens = batch.tokens.clone()
    rel_mask = _relation_mask(batch)
    fp = (torch.rand(batch.token_mask.shape, device=tokens.device) < rate) & rel_mask
    tokens[..., 15] = torch.where(rel_mask, torch.full_like(tokens[..., 15], safe_confidence), tokens[..., 15])
    tokens[..., 15] = torch.where(fp, torch.full_like(tokens[..., 15], fp_confidence), tokens[..., 15])
    tokens[..., 16] = torch.where(fp, torch.ones_like(tokens[..., 16]), torch.zeros_like(tokens[..., 16]))
    # Same adversarial relation semantics as the post-hoc reliability probe.
    tokens[..., 6] = torch.where(fp, torch.ones_like(tokens[..., 6]), tokens[..., 6])
    tokens[..., 8] = torch.where(fp, torch.zeros_like(tokens[..., 8]), tokens[..., 8])
    tokens[..., 9] = torch.where(fp, torch.ones_like(tokens[..., 9]), tokens[..., 9])
    tokens[..., 10] = torch.where(fp, torch.ones_like(tokens[..., 10]), tokens[..., 10])
    return _copy_batch(batch, tokens)


class _DoorPlusAugmentedLoader:
    def __init__(self, loader, args: argparse.Namespace) -> None:
        self.loader = loader
        self.args = args

    def __iter__(self):
        for batch in self.loader:
            yield _apply_doorplus_train_relfp(
                batch,
                rate=float(self.args.doorplus_train_relfp_rate),
                fp_confidence=float(self.args.doorplus_fp_confidence),
                safe_confidence=float(self.args.doorplus_safe_confidence),
            )

    def __len__(self):
        return len(self.loader)


def _build_loaders(args: argparse.Namespace, config: DoorRLConfig):
    if args.dataset == "nuplan":
        full = NuPlanPreprocessedDataset(
            config=config,
            data_root=args.nuplan_root,
            num_samples=args.nuplan_num_samples,
            index_json=args.nuplan_index_json,
            seed=args.seed,
            num_workers=getattr(args, "nuplan_workers", 0),
            materialize_cache=not getattr(args, "nuplan_lazy", False),
        )
    else:
        full = NuScenesSceneDataset(
            config=config,
            nuscenes_root=args.nuscenes_root,
            num_scenes=args.num_scenes,
            version="v1.0-trainval",
            cache_dir=args.token_cache_dir or None,
        )
    names = sorted(set(full.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * n_total))) if n_total > 1 else 0
    n_train = n_total - n_val
    val_scenes = set(shuffled[n_train:])
    train_scenes = set(shuffled[:n_train])
    train_idx = full.indices_for_scenes(train_scenes)
    val_idx = full.indices_for_scenes(val_scenes)
    print(f"scene split: {n_train} train / {n_val} val -> "
          f"{len(train_idx)} train samples / {len(val_idx)} val samples")

    loader_workers = int(getattr(args, "loader_workers", 2))
    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=loader_workers, pin_memory=True,
    )
    if loader_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(Subset(full, train_idx), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(full, val_idx), shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _maybe_warm_start(model: DoorRLModelVariant, stage0_root: str, variant_name: str) -> bool:
    if not stage0_root:
        return False
    ckpt = Path(stage0_root) / variant_name / "model.pt"
    if not ckpt.exists():
        print(f"[warm-start] no ckpt at {ckpt}, training from scratch")
        return False
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    print(f"[warm-start] loaded {ckpt} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    return True


def _freeze_backbone(model: DoorRLModelVariant) -> None:
    """Freeze encoder + abstraction; keep WM + policy trainable."""
    for p in model.encoder.parameters():
        p.requires_grad = False
    # top-k variants have ``abstraction`` directly; decoupled has abstraction_dyn/_rel.
    if hasattr(model, "abstraction") and isinstance(model.abstraction, torch.nn.Module):
        for p in model.abstraction.parameters():
            p.requires_grad = False
    for attr in ("abstraction_dyn", "abstraction_rel"):
        mod = getattr(model, attr, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = False
    # Holistic-16Slot has learned queries + cross_attn; leave those trainable
    # since they're part of "how the representation aggregates" and we said
    # freeze=False is default. The CLI user chose to freeze -> freeze these too.
    for attr in ("holistic_queries", "holistic_cross_attn", "holistic_slot_norm"):
        mod = getattr(model, attr, None)
        if isinstance(mod, torch.nn.Module):
            for p in mod.parameters():
                p.requires_grad = False
        elif isinstance(mod, torch.nn.Parameter):
            mod.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable {trainable:,} / {total:,} params")


def run_condition(
    args: argparse.Namespace,
    condition_name: str,
    loaders=None,
) -> Dict:
    spec = _CONDITIONS[condition_name]
    variant_name = spec["variant"]
    cond = spec["cond"]

    print("\n" + "=" * 80)
    print(f"Stage 1 condition: {condition_name}  (variant={variant_name}, cond={cond})")
    print("=" * 80)

    config = DoorRLConfig.from_json(args.config)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.seed = args.seed
    if args.top_k is not None:
        config.model.top_k = int(args.top_k)
    if args.top_k_dyn is not None:
        config.model.top_k_dyn = int(args.top_k_dyn)
    if args.top_k_rel is not None:
        config.model.top_k_rel = int(args.top_k_rel)
    if (args.top_k_dyn is not None) or (args.top_k_rel is not None):
        if config.model.top_k_dyn + config.model.top_k_rel != config.model.top_k:
            raise ValueError(
                f"Typed-budget override violates top_k constraint: "
                f"top_k_dyn ({config.model.top_k_dyn}) + top_k_rel "
                f"({config.model.top_k_rel}) != top_k ({config.model.top_k})."
            )
        print(f"[budget] top_k_dyn={config.model.top_k_dyn}, "
              f"top_k_rel={config.model.top_k_rel}")
    elif args.top_k is not None:
        print(f"[budget] top_k={config.model.top_k}")
    base_lr = config.training.learning_rate
    if args.lr is not None:
        config.training.learning_rate = float(args.lr)
    else:
        config.training.learning_rate = base_lr * float(args.lr_scale)
    if config.training.learning_rate != base_lr:
        print(f"[lr] {base_lr:.2e} -> {config.training.learning_rate:.2e}")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if loaders is None:
        train_loader, val_loader = _build_loaders(args, config)
    else:
        train_loader, val_loader = loaders

    model = DoorRLModelVariant(config.model, ModelVariant(variant_name))
    if spec.get("rel_to_critic_only", False):
        if not hasattr(model, "rel_to_critic_only"):
            raise ValueError("rel_to_critic_only requires a decoupled variant")
        model.rel_to_critic_only = True
        print("[fusion] rel_to_critic_only=True (actor=dyn, critic=dyn+rel)")
    _maybe_warm_start(model, args.stage0_root, spec.get("warm_start_variant", variant_name))
    if args.freeze_backbone:
        _freeze_backbone(model)
    model.to(device)

    exp_dir = Path(args.output_dir) / f"seed{args.seed}" / condition_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_dir / "model.pt"

    stage1_cfg = Stage1Cfg(
        condition=cond,
        horizon=args.horizon,
        loss_cfg=Stage1LossCfg(entropy_beta=args.entropy_beta),
        reward_cfg=TaskRewardCfg(),
        action_sample_clip=args.action_sample_clip,
    )

    if not args.eval_only:
        train_for_fit = train_loader
        if condition_name == "wm_doorplus_uncertainty" and args.doorplus_train_relfp_rate > 0.0:
            train_for_fit = _DoorPlusAugmentedLoader(train_loader, args)
            print(
                f"[doorplus-train] relfp_rate={args.doorplus_train_relfp_rate} "
                f"fp_conf={args.doorplus_fp_confidence}"
            )
        trainer = ImaginationTrainer(
            model=model, config=config.training,
            device=device, stage1_cfg=stage1_cfg,
        )
        trainer.fit(train_for_fit, val_loader=val_loader)
        torch.save({
            "condition": condition_name,
            "variant": variant_name,
            "model_state_dict": model.state_dict(),
            "stage1_cfg": {
                "condition": stage1_cfg.condition,
                "horizon": stage1_cfg.horizon,
            },
            "seed": args.seed,
            "epochs": args.epochs,
        }, ckpt_path)
        print(f"saved {ckpt_path}")
    else:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)

    metrics = evaluate_stage1(
        model, val_loader, device=device,
        horizon=args.horizon, reward_cfg=stage1_cfg.reward_cfg,
    )
    result = {
        "condition": condition_name,
        "variant": variant_name,
        "seed": args.seed,
        "horizon": args.horizon,
        "metrics": metrics.to_dict(),
    }
    out_json = exp_dir / "stage1_metrics.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"\n{condition_name}: {metrics.to_dict()}")
    print(f"wrote {out_json}")
    return result


def main():
    args = parse_args()
    if args.condition == "pilot":
        conditions = _PILOT_CONDITIONS
    elif args.condition == "all":
        conditions = list(_CONDITIONS.keys())
    else:
        conditions = [args.condition]

    # Share loaders across conditions in the same call (tokenisation once).
    set_seed(args.seed)
    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    loaders = _build_loaders(args, config)

    all_results: Dict[str, Dict] = {}
    for cname in conditions:
        try:
            all_results[cname] = run_condition(args, cname, loaders=loaders)
        except Exception as e:
            print(f"\n[X] condition {cname} failed: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate table.
    out_path = Path(args.output_dir) / f"seed{args.seed}" / "stage1_all.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\naggregate -> {out_path}")

    print("\n" + "=" * 90)
    print(f"{'condition':<16} {'return_mean':>12} {'coll_rate':>11} "
          f"{'coll_mean':>11} {'stability':>11}")
    print("-" * 90)
    for cname, res in all_results.items():
        m = res["metrics"]
        print(f"{cname:<16} {m['latent_return_mean']:>12.3f} "
              f"{m['imagined_collision_rate']:>11.3f} {m['collision_mean']:>11.3f} "
              f"{m['rollout_stability']:>11.4f}")


if __name__ == "__main__":
    main()
