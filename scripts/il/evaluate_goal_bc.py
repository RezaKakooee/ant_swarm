"""Evaluate a repaired-goal BC checkpoint on new full-maze random starts.

Saves the actual start poses/goals for replay. --repeat checks that every
per-episode result repeats exactly. This never loads the geodesic field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ant_swarm import AntSwarmEnv, load_config
from train_goal_bc import evaluate_suite
from train_chunked_bc import ChunkPolicy


@torch.no_grad()
def render_episode(config, dataset, checkpoint, episode, output):
    """Render a full deterministic rollout from a saved evaluation start."""
    from PIL import Image
    torch.set_num_threads(1)
    env = AntSwarmEnv(config=load_config(config), seed=0)
    with np.load(dataset) as z:
        obs, _ = env.reset(options={'init_pose': z['init_pose'][episode],
                                    'goal': z['goal'][episode]})
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    net = ChunkPolicy(obs.size, 1)
    net.load_state_dict(saved['state_dict'])
    net.eval()
    frames = [Image.fromarray(env.render())]
    for step in range(env.max_steps):
        action = net(torch.from_numpy(obs.reshape(1, -1)))[0, 0].numpy()
        obs, _, term, trunc, info = env.step(action.reshape(env.action_space.shape))
        if step % 2 == 1 or term or trunc:
            frames.append(Image.fromarray(env.render()))
        if term or trunc:
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   loop=0, duration=80, optimize=True)
    env.close()
    return {'success': bool(info['is_success']), 'distance_m': info['object_distance']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='storage_local/ant__20260905_1322__240805__train_goal_bc/bc_goal_fixed.pt')
    p.add_argument('--config', default='storage_local/ant__20260905_1322__240805__train_goal_bc/config.yaml')
    p.add_argument('--out', default='storage_local/ant__20260905_1322__240805__train_goal_bc/fresh_starts')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=10000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--repeat', action='store_true')
    p.add_argument('--render', action='store_true', help='save success and failure rollouts as GIFs')
    p.add_argument('--render-count', type=int, default=1,
                   help='how many successes and how many failures to render')
    args = p.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        p.error('--episodes and --workers must be positive')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved.get('goal_observation_version') != 2:
        p.error('checkpoint was not trained with repaired goal observations')
    cfg = load_config(args.config)
    if not cfg.spawn.resample_each_reset or cfg.spawn.get('fixed_pose') is not None:
        p.error('fresh-start evaluation requires randomized full-maze spawns')
    if cfg.env.reward_mode != 'sparse':
        p.error('requires sparse reward; no geodesic field is used')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        p.error('results exist; choose a new --out')
    env = AntSwarmEnv(config=cfg, seed=args.seed)
    poses, goals = [], []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        poses.append([*env.init_center, env.init_angle])
        goals.append(env.layout.goal.copy())
    env.close()
    poses = np.asarray(poses, dtype=np.float32)
    if len(np.unique(poses, axis=0)) != args.episodes:
        raise ValueError('fresh-start sampler produced repeated poses')
    starts = out / 'starts.npz'
    np.savez(starts, init_pose=poses, goal=np.asarray(goals, dtype=np.float32))
    kwargs = dict(config=args.config, dataset=str(starts),
                  checkpoints={'policy': args.checkpoint},
                  eval_ids=np.arange(args.episodes), workers=args.workers,
                  include_expert=False)
    summary, rows = evaluate_suite(**kwargs)
    print(json.dumps(summary, indent=2), flush=True)
    repeat_matches = None
    if args.repeat:
        repeat_summary, repeat_rows = evaluate_suite(**kwargs)
        repeat_matches = summary == repeat_summary and rows == repeat_rows
        if not repeat_matches:
            raise AssertionError('identical evaluation did not repeat exactly')
        print('Repeated evaluation is identical for every episode.', flush=True)
    report = {'args': vars(args), 'summary': summary, 'episodes': rows,
              'repeat_identical': repeat_matches}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.render:
        for success, tag in ((True, 'success'), (False, 'failure')):
            picked = [r for r in rows if r['success'] == success][:args.render_count]
            for k, row in enumerate(picked, 1):
                name = f'{tag}.gif' if args.render_count == 1 else f'{tag}_{k}.gif'
                res = render_episode(args.config, str(starts), args.checkpoint,
                                     row['episode'], out / name)
                print(f"{name}: episode {row['episode']} "
                      f"distance={res['distance_m']:.4f} m", flush=True)
            if len(picked) < args.render_count:
                print(f'only {len(picked)} {tag} episodes available', flush=True)


if __name__ == '__main__':
    main()
