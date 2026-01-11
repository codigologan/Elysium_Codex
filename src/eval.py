from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

from src.utils import ensure_dir, get_device, seed_everything, load_json
from src.data import make_synthetic_classification, make_loaders
from src.model import build_mlp


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, dl, loss_fn, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        loss = loss_fn(out, yb)
        total_loss += loss.item()
        total_acc += accuracy(out, yb)
    return total_loss / len(dl), total_acc / len(dl)


def load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if "model_state" not in ckpt:
        raise ValueError("Checkpoint inválido: não encontrei 'model_state'.")
    meta = ckpt.get("meta", {})
    return {"state": ckpt["model_state"], "meta": meta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="lad_mlp_v1", help="nome do run (pasta em runs/ e models/)")
    parser.add_argument("--ckpt", type=str, default="best", choices=["best", "last"], help="qual checkpoint avaliar")
    parser.add_argument("--save_csv", action="store_true", help="salva um CSV em reports/")
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    device = get_device()

    run_dir = os.path.join("runs", args.run_name)
    ckpt_dir = os.path.join("models", args.run_name)
    ckpt_path = os.path.join(ckpt_dir, f"{args.ckpt}.pt")
    cfg_path = os.path.join(run_dir, "config.json")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Não achei config.json em {cfg_path}. Rode o treino primeiro.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Não achei checkpoint em {ckpt_path}. Verifique run_name e ckpt.")

    cfg = load_json(cfg_path)
    seed_everything(int(cfg.get("seed", 42)))

    # Dados (mesmo gerador do treino, com sizes/dims do config)
    input_dim = int(cfg.get("input_dim", 20))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    num_classes = int(cfg.get("num_classes", 2))
    train_size = int(cfg.get("train_size", 20000))
    val_size = int(cfg.get("val_size", 5000))
    batch_size = int(cfg.get("batch_size", 256))

    train, val = make_synthetic_classification(train_size, val_size, input_dim)
    _, dl_val = make_loaders(train, val, batch_size)

    # Modelo
    model = build_mlp(input_dim, hidden_dim, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Carrega checkpoint
    ckpt = load_ckpt(ckpt_path, device)
    model.load_state_dict(ckpt["state"])

    val_loss, val_acc = evaluate(model, dl_val, loss_fn, device)

    meta = ckpt.get("meta", {})
    epoch = meta.get("epoch", None)
    best_val_loss = meta.get("best_val_loss", None)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": args.run_name,
        "ckpt": args.ckpt,
        "device": str(device),
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "best_val_loss_in_ckpt_meta": best_val_loss,
    }

    print("\n[LAD][EVAL] Resultado")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.save_csv:
        ensure_dir(args.reports_dir)
        out_csv = os.path.join(args.reports_dir, "eval_results.csv")
        write_header = not os.path.exists(out_csv)

        # fixed, explicit schema to avoid column-order/length mismatches
        fieldnames = [
            "timestamp",
            "run_name",
            "ckpt",
            "device",
            "epoch",
            "val_loss",
            "val_acc",
            "best_val_loss_in_ckpt_meta",
        ]

        row = {
            "timestamp": result["timestamp"],
            "run_name": result["run_name"],
            "ckpt": result["ckpt"],
            "device": result["device"],
            "epoch": result["epoch"],
            "val_loss": result["val_loss"],
            "val_acc": result["val_acc"],
            "best_val_loss_in_ckpt_meta": result.get("best_val_loss_in_ckpt_meta"),
        }

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerow(row)

        print(f"\n[LAD][EVAL] CSV atualizado: {out_csv}")


if __name__ == "__main__":
    main()
