"""Push a finished run's results.json into W&B after the fact.

For runs that started before tracking existed. Reads the run dir, logs the
per-round / per-eval history as steps and the final numbers as summary.

    python scripts/tools/wandb_backfill.py storage_local/ant__*__dagger_shared_n5_h4
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ant_swarm.tracking import Tracker  # noqa: E402


def main():
    for arg in sys.argv[1:]:
        d = Path(arg); res = d / "results.json"
        if not res.exists():
            print(f"skip {d}: no results.json"); continue
        r = json.loads(res.read_text())
        name = d.name
        if "dagger" in r:
            tr = Tracker(name, group="dagger_shared_ant", tags=["marl", "dagger", "backfill"], config=r.get("args", {}))
            for row in r["dagger"]:
                tr.log({"dagger/student_sr": row["sr"], "dagger/mean_dist": row["dist"],
                        "dagger/mse": row["mse"], "dagger/samples": row["samples"],
                        **({"dagger/collect_sr": row["collect_sr"]} if "collect_sr" in row else {})},
                       step=row["it"])
            for u in r.get("unseen", []):
                tr.summary({f"unseen_layout_{u['layout_seed']}/sr": u["sr"]})
            tr.summary({"final/student_sr": r["dagger"][-1]["sr"]})
        elif "eval_history" in r:
            tr = Tracker(name, group="finetune_bc", tags=["finetune", r.get("mode", ""), "backfill"], config=r.get("finetune", {}))
            for h in r["eval_history"]:
                tr.log({"eval/success_rate_pct": h["success_rate_pct"], "eval/mean_distance_m": h["mean_distance_m"]}, step=h["timesteps"])
            tr.summary({f"final/{k}": v for k, v in r["after"].items()})
        elif "history" in r:
            tr = Tracker(name, group="marl_ppo", tags=["marl", "ppo", "backfill"], config=r.get("args", {}))
            for h in r["history"]:
                tr.log({"eval/success_rate_pct": h["success_rate_pct"], "eval/mean_distance_m": h["mean_distance_m"]}, step=h["steps"])
            tr.summary({f"final/n{x['ants']}_sr": x["success_rate_pct"] for x in r.get("final", [])})
        else:
            print(f"skip {d}: unknown results.json layout"); continue
        ok = tr.run is not None
        tr.finish(); print(f"{'pushed' if ok else 'NOT pushed (tracking off)'} {name}")


if __name__ == "__main__":
    main()
