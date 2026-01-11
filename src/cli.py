"""Simple CLI with subcommands: train, eval."""

import argparse
import os
from datetime import datetime

from src.train import Trainer
from src.utils import load_json


def cmd_train(args):
    cfg_path = args.config
    cfg = Trainer.__annotations__.get('cfg')
    # instantiate Trainer via config file path
    from src.utils import TrainConfig
    cfg = TrainConfig.from_json(cfg_path)
    trainer = Trainer(cfg, runs_dir=args.runs_dir, save_dir=args.save_dir, use_amp=args.amp, save_opt=args.save_opt)
    trainer.fit(resume=args.resume)


def cmd_eval(args):
    from src.utils import TrainConfig
    cfg = TrainConfig.from_json(args.config)
    trainer = Trainer(cfg, runs_dir=args.runs_dir, save_dir=args.save_dir, use_amp=args.amp, save_opt=args.save_opt)
    ckpt_name = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(trainer.ckpt_dir, args.ckpt + (".pt" if not args.ckpt.endswith('.pt') else ""))
    meta = trainer.load_ckpt(ckpt_name)

    # evaluate on validation set
    model = trainer.model
    device = trainer.device
    loss_fn = trainer.loss_fn
    val_loader = trainer.dl_val

    total_loss, total_acc = 0.0, 0.0
    import torch
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            total_loss += loss.item()
            total_acc += (out.argmax(dim=1) == yb).float().mean().item()
    total_loss /= len(val_loader)
    total_acc /= len(val_loader)

    ts = datetime.now().isoformat()
    print("[LAD][EVAL] Resultado")
    print(f"  timestamp: {ts}")
    print(f"  run_name: {cfg.run_name}")
    print(f"  ckpt: {args.ckpt}")
    print(f"  device: {device}")
    print(f"  val_loss: {total_loss}")
    print(f"  val_acc: {total_acc}")
    print(f"  best_val_loss_in_ckpt_meta: {meta.get('best_val_loss')}")

    if args.save_csv:
        import csv
        os.makedirs('reports', exist_ok=True)
        path = os.path.join('reports', 'eval_results.csv')
        existed = os.path.exists(path)

        # align CLI eval CSV output with canonical schema used by src.eval
        fieldnames = [
            'timestamp', 'run_name', 'ckpt', 'device', 'epoch', 'val_loss', 'val_acc', 'best_val_loss_in_ckpt_meta'
        ]

        row = {
            'timestamp': ts,
            'run_name': cfg.run_name,
            'ckpt': args.ckpt,
            'device': str(device),
            'epoch': meta.get('epoch', None),
            'val_loss': float(total_loss),
            'val_acc': float(total_acc),
            'best_val_loss_in_ckpt_meta': meta.get('best_val_loss'),
        }

        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not existed:
                writer.writeheader()
            writer.writerow(row)
        print(f"[LAD][EVAL] CSV atualizado: {path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_train = sub.add_parser('train')
    p_train.add_argument('--config', default='configs/mlp_default.json')
    p_train.add_argument('--runs-dir', default=None)
    p_train.add_argument('--save-dir', default=None)
    p_train.add_argument('--amp', action='store_true')
    p_train.add_argument('--save-opt', action='store_true')
    p_train.add_argument('--resume', default='')

    p_eval = sub.add_parser('eval')
    p_eval.add_argument('--config', default='configs/mlp_default.json')
    p_eval.add_argument('--ckpt', default='best')
    p_eval.add_argument('--save-csv', action='store_true')
    p_eval.add_argument('--runs-dir', default=None)
    p_eval.add_argument('--save-dir', default=None)
    p_eval.add_argument('--amp', action='store_true')
    p_eval.add_argument('--save-opt', action='store_true')

    p_rl_train = sub.add_parser('rl-train')
    p_rl_train.add_argument('--episodes', type=int, default=200)
    p_rl_train.add_argument('--config', default='configs/rl_dqn.json')
    p_rl_train.add_argument('--dream-every', type=int, default=25, help='a cada N episodios roda uma noite de sonhos (0 desliga)')
    p_rl_train.add_argument('--dream-steps', type=int, default=800, help='quantos TD updates por noite')
    p_rl_train.add_argument('--dream-batch', type=int, default=128, help='batch size do sonho')
    p_rl_train.add_argument('--dream-sigma', type=float, default=0.02, help='ruido gaussiano no estado durante sonhos')
    p_rl_train.add_argument('--dream-mix-prob', type=float, default=0.20, help='probabilidade de mixar estados no sonho')

    p_rl_eval = sub.add_parser('rl-eval')
    p_rl_eval.add_argument('--model', default='models/rl_logan_v1/best.pt')

    p_rl_lb = sub.add_parser('rl-leaderboard')
    p_rl_lb.add_argument('--csv', default='reports/rl_results.csv')
    p_rl_lb.add_argument('--metric', default='mean_reward_last_50')
    p_rl_lb.add_argument('--include-rl', action='store_true', help='also include per-episode RL files when summarizing')

    p_lb = sub.add_parser('leaderboard')
    p_lb.add_argument('--csv', default='reports/eval_results.csv')
    p_lb.add_argument('--metric', default='val_loss', choices=['val_loss', 'val_acc', 'mean_reward'])
    p_lb.add_argument('--include-rl', action='store_true')

    args = parser.parse_args()
    if args.cmd == 'train' or args.cmd is None:
        cmd_train(args)
    elif args.cmd == 'eval':
        cmd_eval(args)
    elif args.cmd == 'rl-train':
        # run RL training
        from src.rl.train_rl import train_loop
        train_loop(
            config_path=args.config,
            episodes=args.episodes,
            dream_every=args.dream_every,
            dream_steps=args.dream_steps,
            dream_batch=args.dream_batch,
            dream_sigma=args.dream_sigma,
            dream_mix_prob=args.dream_mix_prob,
        )
    elif args.cmd == 'rl-eval':
        from src.rl.eval_rl import evaluate
        evaluate(model_path=args.model)
    elif args.cmd == 'leaderboard':
        # delegate to the leaderboard module; construct argv for its parser
        import sys
        lb_argv = ['leaderboard', '--csv', args.csv, '--metric', args.metric]
        if getattr(args, 'include_rl', False):
            lb_argv.append('--include-rl')
        old_argv = sys.argv
        try:
            sys.argv = lb_argv
            from src.leaderboard import main as _lb_main
            _lb_main()
        finally:
            sys.argv = old_argv
    elif args.cmd == 'rl-leaderboard':
        # run the RL-specific leaderboard (ensure its parser sees only RL args)
        import sys
        rl_argv = ['rl-leaderboard', '--csv', args.csv, '--metric', args.metric]
        old_argv = sys.argv
        try:
            sys.argv = rl_argv
            from src.leaderboard_rl import main as _rl_lb
            _rl_lb()
        finally:
            sys.argv = old_argv
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
