"""
命令行入口。

用法
    python run.py                          读 data/sample.csv，打印终端报告
    python run.py --csv 我的数据.csv        换数据源
    python run.py --html report.html        生成自包含 HTML 报告
    python run.py --evaluate               在合成数据上跑量化评测
    python run.py --synthesize             重新生成示例数据
    python run.py --json out.json          把判定结果落成 JSON

配置从 config.yaml 读（不存在就用内置默认值）。
配置文件只覆盖**参数**，不覆盖数据 —— 数据永远走命令行或默认路径，
免得"我改了配置但报告没变"这种难查的事。
"""

import argparse
import json
import os
import sys


def _load_config(path="config.yaml"):
    """读配置文件。文件不存在不是错误，直接用默认值。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except ImportError:
        print("提示：装了 PyYAML 才能读 config.yaml，本次用默认参数。"
              f"（pip install PyYAML）", file=sys.stderr)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} 的顶层应该是一个映射（key: value），实际是 {type(cfg).__name__}")
    return cfg


def _build_parser():
    p = argparse.ArgumentParser(
        prog="fitxray",
        description="体成分变化归因：先确认这个变化是不是真的，再谈它意味着什么",
    )
    p.add_argument("--csv", default=None, help="测量记录（CSV 或 JSONL），默认 data/sample.csv")
    p.add_argument("--profile", default=None, help="profile 文件（YAML 或 JSON），可选")
    p.add_argument("--device", default=None, help="设备 ID；不填就用 profile 里的")
    p.add_argument("--html", nargs="?", const="report.html", default=None,
                   help="生成自包含 HTML 报告，可指定路径（默认 report.html）")
    p.add_argument("--json", nargs="?", const="result.json", default=None,
                   help="把判定结果落成 JSON，可指定路径（默认 result.json）")
    p.add_argument("--evaluate", action="store_true", help="在合成数据上跑量化评测")
    p.add_argument("--evaluate-seeds", type=int, default=None, help="评测用多少个随机种子")
    p.add_argument("--synthesize", action="store_true", help="重新生成示例数据到 data/")
    p.add_argument("--seed", type=int, default=7, help="生成示例数据用的随机种子")
    p.add_argument("--contamination", type=float, default=None,
                   help="生成示例数据时注入污染的比例（默认 0.10）")
    p.add_argument("--alpha", type=float, default=None, help="多重比较校正的 FDR 水平（默认 0.05）")
    p.add_argument("--config", default="config.yaml", help="配置文件路径")
    return p


def _synth(args, cfg):
    from .synth import DEFAULT_SEGMENTS, generate, write_csv, write_truth

    out_dir = cfg.get("out_dir", "data")
    contamination = (args.contamination if args.contamination is not None
                     else cfg.get("contamination", 0.10))
    records, truth = generate(segments=DEFAULT_SEGMENTS, seed=args.seed,
                              contamination_rate=contamination)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sample.csv")
    truth_path = os.path.join(out_dir, "ground_truth.json")
    write_csv(records, csv_path)
    write_truth(truth, truth_path)

    print(f"生成 {len(records)} 条测量记录（{truth['total_days']} 天，seed={args.seed}）")
    print(f"  真实轨迹：{' → '.join(s['note'] for s in DEFAULT_SEGMENTS)}")
    print(f"  注入污染：{len(truth['contaminated_indices'])} 条"
          f"（{contamination:.0%} 的测量是「不该拿来比」的）")
    print(f"  注入的体重日间波动 TE = {truth['te_weight']} kg")
    print(f"  注入的体脂率方法误差   = {truth['method_bf_pct']} 个百分点")
    print(f"  → {csv_path}")
    print(f"  → {truth_path}")
    return 0


def _evaluate(args, cfg):
    from .evaluate import evaluate, format_evaluation

    seeds = args.evaluate_seeds or cfg.get("evaluate_seeds", 30)
    print(format_evaluation(evaluate(n_seeds=seeds)))
    return 0


def _report(args, cfg):
    from .loaders import load_measurements
    from .verdict import format_verdicts, verdicts

    csv_path = args.csv or cfg.get("measurements", os.path.join("data", "sample.csv"))
    if not os.path.exists(csv_path):
        print(f"找不到测量记录：{csv_path}", file=sys.stderr)
        print("先跑一次 `python run.py --synthesize` 生成示例数据，"
              f"或用 --csv 指定你自己的文件。", file=sys.stderr)
        return 2

    profile = None
    profile_path = args.profile or cfg.get("profile")
    if profile_path and os.path.exists(profile_path):
        from .loaders import load_profile

        profile = load_profile(profile_path)

    device = args.device or cfg.get("device_id")
    if not device and profile:
        device = profile.get("device_id")

    records = load_measurements(csv_path, device_id=device)
    if not records:
        print(f"{csv_path} 里没有可用的测量记录", file=sys.stderr)
        return 2

    alpha = args.alpha if args.alpha is not None else cfg.get("alpha", 0.05)
    result = verdicts(records, alpha=alpha)

    print(format_verdicts(result))

    if args.html:
        from .report import write_html

        write_html(args.html, result)
        print(f"\nHTML 报告 → {args.html}")

    if args.json:
        # 记录里带 _date / _day 这种内部字段，落盘前去掉
        payload = {k: v for k, v in result.items() if k != "records"}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"判定结果 → {args.json}")

    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)

    if args.synthesize:
        return _synth(args, cfg)
    if args.evaluate:
        return _evaluate(args, cfg)
    return _report(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
