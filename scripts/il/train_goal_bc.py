"""Re-run the unchanged H=1 BC baseline after fixing stale goal observations.

No geodesic field, curriculum, action noise, or privileged policy inputs.
Hold out entire start-pose groups, and evaluate each selected pose only once.
The report records the exact held-out episode IDs and all per-episode results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ant_swarm import AntSwarmEnv, load_config
from ant_swarm.run_id import build_run_id
from repair_replay_goals import require_correct_goals
from train_chunked_bc import ChunkPolicy, train


def split_by_start_pose(init_poses, n_eval, seed):
    """All repeats of an evaluation start are excluded from training."""
    _, first, groups = np.unique(init_poses, axis=0, return_index=True, return_inverse=True)
    if not 0 < n_eval < len(first):
        raise ValueError('need more distinct start poses than evaluation episodes')
    selected = np.random.default_rng(seed).choice(len(first), n_eval, replace=False)
    eval_ids = np.sort(first[selected])
    train_episodes = ~np.isin(groups, selected)
    return train_episodes, eval_ids


_WORKER = {}


def _eval_init(config, dataset, checkpoints):
    torch.set_num_threads(1)
    cfg = load_config(config)
    if cfg.env.reward_mode != 'sparse':
        raise ValueError('this experiment requires sparse reward')
    with np.load(dataset) as z:
        _WORKER.update(poses=z['init_pose'], goals=z['goal'])
        if 'actions' in z:
            _WORKER.update(offsets=z['offsets'], actions=z['actions'])
    _WORKER['env'] = AntSwarmEnv(config=cfg, seed=0)
    policies = {}
    for tag, path in checkpoints.items():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        policy = ChunkPolicy(int(np.prod(_WORKER['env'].observation_space.shape)), 1)
        policy.load_state_dict(saved['state_dict'])
        policy.eval()
        policies[tag] = policy
    _WORKER['policies'] = policies


@torch.no_grad()
def _evaluate_one(task):
    tag, ep = task
    env = _WORKER['env']
    obs, _ = env.reset(seed=int(ep), options={
        'init_pose': _WORKER['poses'][ep], 'goal': _WORKER['goals'][ep]})
    policy = _WORKER['policies'].get(tag)
    actions = (_WORKER['actions'][_WORKER['offsets'][ep]:_WORKER['offsets'][ep+1]]
               if tag == 'expert' else None)
    initial_goal = env.layout.goal.copy()
    trail = []
    max_tip_x = float(env.state.obj.world_corners()[:, 0].max())
    max_center_x = float(env.state.obj.center[0])
    info = {'is_success': False, 'object_distance': env.state.distance_to_goal()}
    limit = len(actions) if tag == 'expert' else env.max_steps
    for step in range(limit):
        if policy is None:
            action = actions[step]
        else:
            action = policy(torch.from_numpy(obs.reshape(1, -1)))[0, 0].numpy()
        obs, _, terminated, truncated, info = env.step(
            np.asarray(action, dtype=np.float32).reshape(env.action_space.shape))
        trail.append(env.state.obj.center.copy())
        max_tip_x = max(max_tip_x, float(env.state.obj.world_corners()[:, 0].max()))
        max_center_x = max(max_center_x, float(env.state.obj.center[0]))
        if terminated or truncated:
            break
    np.testing.assert_array_equal(env.layout.goal, initial_goal)
    jammed = len(trail) >= 50 and float(np.linalg.norm(
        np.diff(np.asarray(trail[-50:]), axis=0), axis=1).max()) < 1e-7
    return {'variant': tag, 'episode': int(ep), 'success': bool(info['is_success']),
            'distance_m': float(info['object_distance']), 'steps': len(trail),
            'max_tip_x': max_tip_x, 'max_center_x': max_center_x, 'jammed': jammed}


def evaluate_suite(config, dataset, checkpoints, eval_ids, workers, *, include_expert=True):
    tags = (['expert'] if include_expert else []) + list(checkpoints)
    tasks = [(tag, int(ep)) for tag in tags for ep in eval_ids]
    with get_context('spawn').Pool(workers, initializer=_eval_init,
                                    initargs=(config, dataset, checkpoints)) as pool:
        rows = list(pool.imap(_evaluate_one, tasks, chunksize=1))
    summaries = {}
    for tag in tags:
        selected = [row for row in rows if row['variant'] == tag]
        summaries[tag] = {
            'episodes': len(selected),
            'successes': sum(row['success'] for row in selected),
            'success_rate_pct': 100 * np.mean([row['success'] for row in selected]),
            'mean_distance_m': float(np.mean([row['distance_m'] for row in selected])),
            'jammed': sum(row['jammed'] for row in selected),
        }
    return summaries, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/il/il_augmented_bc.yaml')
    p.add_argument('--dataset', default='storage_local/datasets/successes_v1/dataset.npz')
    p.add_argument('--cache', default='storage_local/cache/replay_full_goals_v2.npz')
    # run dirs live directly under storage_local/ and are named by run id
    # (see ant_swarm/run_id.py); ANT_SWARM_RUN_ID from ops/sb_train.sh wins
    p.add_argument('--out', default=None,
                   help='default: storage_local/<run id>')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--old-checkpoint', default='storage_local/checkpoints/bc_chunk_H1.pt')
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out) if args.out else (
        Path('storage_local') / build_run_id('train_goal_bc'))
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    if cfg.env.reward_mode != 'sparse':
        p.error('requires sparse reward')
    with np.load(args.dataset) as z:
        poses, goals = z['init_pose'], z['goal']
    train_episodes, eval_ids = split_by_start_pose(poses, args.eval_episodes, args.seed)
    checkpoint = out / 'bc_goal_fixed.pt'
    metadata = {'args': vars(args), 'eval_episode_ids': eval_ids.tolist(),
                'held_out_episode_count': int((~train_episodes).sum())}
    if args.evaluate_only:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved.get('eval_episode_ids') != eval_ids.tolist():
            p.error('requested evaluation split differs from the checkpoint holdout')
        metadata = json.loads((out / 'training.json').read_text())
    if not args.evaluate_only:
        if checkpoint.exists():
            p.error(f'{checkpoint} exists; choose a new --out')
        with np.load(args.cache) as z:
            obs, act, epid = z['obs'], z['act'], z['epid']
        metadata['cache_audit'] = require_correct_goals(obs, epid, goals, cfg)
        keep = train_episodes[epid]
        obs, act = obs[keep], act[keep]
        metadata['training_transitions'] = len(obs)
        from omegaconf import OmegaConf
        OmegaConf.save(cfg, out / 'config.yaml')
        print(json.dumps(metadata, indent=2), flush=True)
        start = time.time()
        idx = np.arange(len(obs), dtype=np.int32)[:, None]
        mask = np.ones((len(obs), 1), dtype=bool)
        net = train(obs, act, idx, mask, 1, args.epochs, args.batch_size,
                    args.lr, args.device, seed=args.seed)
        torch.save({'state_dict': {k: v.cpu() for k, v in net.state_dict().items()},
                    'horizon': 1, 'goal_observation_version': 2,
                    'eval_episode_ids': eval_ids.tolist()}, checkpoint)
        metadata['training_seconds'] = time.time() - start
        (out / 'training.json').write_text(json.dumps(metadata, indent=2) + '\n')
        del net, obs, act, idx, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    checkpoints = {'corrected_bc': str(checkpoint)}
    if args.old_checkpoint:
        checkpoints['old_bc_correct_goal'] = args.old_checkpoint
    summaries, rows = evaluate_suite(args.config, args.dataset, checkpoints,
                                      eval_ids, args.workers)
    report = {'metadata': metadata, 'summary': summaries, 'episodes': rows}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
