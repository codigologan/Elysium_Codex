from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils import seed_everything, get_device, ensure_dir, save_json, TrainConfig
from src.data import make_synthetic_classification, make_loaders
from src.model import build_mlp


logger = logging.getLogger("lad.train")


def safe_torch_load(path: str, map_location=None, weights_only: bool = True):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    try:
        torch.save(payload, path)
    except Exception:
        logger.exception("Failed to save checkpoint %s", path)


class Trainer:
    def __init__(self, cfg: TrainConfig, *, runs_dir: Optional[str] = None, save_dir: Optional[str] = None, use_amp: Optional[bool] = None, save_opt: bool = False):
        self.cfg = cfg
        self.device = get_device()
        self.use_amp = use_amp if use_amp is not None else getattr(cfg, "amp", False)
        self.save_opt = save_opt

        self.runs_dir = runs_dir or getattr(cfg, "runs_dir", "runs")
        self.save_dir = save_dir or getattr(cfg, "save_dir", "models")

        self.run_dir = os.path.join(self.runs_dir, cfg.run_name)
        self.ckpt_dir = os.path.join(self.save_dir, cfg.run_name)
        ensure_dir(self.run_dir)
        ensure_dir(self.ckpt_dir)

        save_json(os.path.join(self.run_dir, "config.json"), cfg.to_dict())

        # data
        train, val = make_synthetic_classification(cfg.train_size, cfg.val_size, cfg.input_dim)
        self.dl_train, self.dl_val = make_loaders(train, val, cfg.batch_size)

        # model / optimizer / loss
        self.model = build_mlp(cfg.input_dim, cfg.hidden_dim, cfg.num_classes).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=getattr(cfg, "weight_decay", 0.0))
        self.loss_fn = nn.CrossEntropyLoss()

        # scaler: support different torch versions; provide noop scaler when amp not available
        try:
            GradScaler = torch.amp.GradScaler
        except Exception:
            GradScaler = getattr(torch.cuda.amp, "GradScaler", None)

        if GradScaler is None:
            # fallback noop scaler (no automatic mixed precision)
            self.use_amp = False
            class _NoopScaler:
                def scale(self, v):
                    return v
                def unscale_(self, optimizer):
                    pass
                def step(self, optimizer):
                    optimizer.step()
                def update(self):
                    pass
                def state_dict(self):
                    return {}
                def load_state_dict(self, d):
                    return
            self.scaler = _NoopScaler()
        else:
            try:
                self.scaler = GradScaler(device_type=("cuda" if self.device.type == "cuda" else "cpu"), enabled=self.use_amp)
            except TypeError:
                try:
                    self.scaler = GradScaler(enabled=self.use_amp)
                except TypeError:
                    self.scaler = GradScaler()

        # logging
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
        self.writer = SummaryWriter(self.run_dir)

    def load_ckpt(self, path: str) -> Dict[str, Any]:
        ckpt = safe_torch_load(path, map_location=self.device)
        if not isinstance(ckpt, dict):
            raise ValueError("Invalid checkpoint format")
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        if self.save_opt and ckpt.get("opt_state") is not None:
            try:
                self.opt.load_state_dict(ckpt["opt_state"])
            except Exception:
                logger.exception("Failed to load optimizer state from %s", path)
        if ckpt.get("scaler_state") is not None and self.use_amp:
            try:
                self.scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                logger.exception("Failed to load scaler state from %s", path)
        return ckpt.get("meta", {})

    def save_ckpt(self, name: str, epoch: int, best_val_loss: float) -> None:
        meta = {"epoch": epoch, "best_val_loss": best_val_loss, "config": self.cfg.to_dict()}
        payload = {"model_state": self.model.state_dict(), "meta": meta}
        if self.save_opt:
            payload.update({"opt_state": self.opt.state_dict(), "scaler_state": self.scaler.state_dict() if self.use_amp else None})
        save_payload(os.path.join(self.ckpt_dir, name), payload)

    def fit(self, resume: str = "") -> float:
        start_epoch = 0
        best_val_loss = float("inf")
        if resume:
            meta = self.load_ckpt(resume)
            start_epoch = int(meta.get("epoch", 0) + 1)
            best_val_loss = float(meta.get("best_val_loss", best_val_loss))
            logger.info("RESUME: epoch=%s best_val_loss=%s", start_epoch, best_val_loss)

        # early stopping params from config (optional)
        patience = int(getattr(self.cfg, "patience", 2))
        early_stopping = bool(getattr(self.cfg, "early_stopping", False))
        no_improve = 0

        t0 = time.time()
        for epoch in range(start_epoch, self.cfg.epochs):
            self.model.train()
            ep_loss, ep_acc = 0.0, 0.0

            for xb, yb in self.dl_train:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=("cuda" if self.device.type == "cuda" else "cpu"), enabled=self.use_amp):
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)

                self.scaler.scale(loss).backward()

                grad_clip_val = getattr(self.cfg, "grad_clip", 0.0)
                if grad_clip_val and grad_clip_val > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)

                self.scaler.step(self.opt)
                self.scaler.update()

                ep_loss += loss.item()
                ep_acc += accuracy(out.detach(), yb)

            tr_loss = ep_loss / len(self.dl_train)
            tr_acc = ep_acc / len(self.dl_train)

            # eval
            va_loss, va_acc = 0.0, 0.0
            with torch.no_grad():
                for xb, yb in self.dl_val:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    out = self.model(xb)
                    loss = self.loss_fn(out, yb)
                    va_loss += loss.item()
                    va_acc += accuracy(out, yb)
            va_loss /= len(self.dl_val)
            va_acc /= len(self.dl_val)

            self.writer.add_scalar("loss/train", tr_loss, epoch)
            self.writer.add_scalar("loss/val", va_loss, epoch)
            self.writer.add_scalar("acc/train", tr_acc, epoch)
            self.writer.add_scalar("acc/val", va_acc, epoch)

            logger.info("epoch %03d | train loss %.4f acc %.4f | val loss %.4f acc %.4f", epoch, tr_loss, tr_acc, va_loss, va_acc)

            # checkpoints + early stopping
            self.save_ckpt("last.pt", epoch, best_val_loss)

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                no_improve = 0
                self.save_ckpt("best.pt", epoch, best_val_loss)
            else:
                no_improve += 1

            if early_stopping and no_improve >= patience:
                logger.info("early stopping triggered at epoch %s (no_improve=%s, patience=%s)", epoch, no_improve, patience)
                break

        self.writer.close()
        logger.info("done in %.2fs | best_val_loss=%.6f", time.time() - t0, best_val_loss)
        return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/mlp_default.json")
    parser.add_argument("--resume", type=str, default="", help="path to checkpoint to resume (optional)")
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    seed_everything(cfg.seed)
    trainer = Trainer(cfg)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
