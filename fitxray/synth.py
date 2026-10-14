"""
合成数据生成器 —— 造一段**已知真相**的体成分序列。

为什么需要这个？两个原因：

**① 评测要有标准答案。**
没有 ground truth，你只能说「跑起来看着挺合理」。
有了它，才能问出真问题：估出来的噪声地板离真实值有多远？
注入的坏点抓出来了几个？喂一条**纯噪声**序列，它会不会误报「真实变化」？
最后那个问题（假阳性率）是最有说服力的 —— 一个只会说"变了"的工具没有价值。

**② 演示不能拿真人的数据。**
体成分是健康数据，任何人都不该往开源仓库里放。
合成数据可以反复生成、参数可调、还能顺便把真相一起输出。

---- 观测模型 ----

真相和读数之间隔着一层，这一层是本项目全部立论的来源：

    体重读数  = 真实体重 + 日间波动            ← 误差小，主要是随机的
    体脂率读数 = 真实体脂率 + **方法误差**     ← 误差大，±3 个百分点

第二项是关键。它不是噪声，是 BIA 方法本身的偏差 —— 而偏差**依赖测量时的
身体状态**（水合、血流量、体温）。每天的状态不一样，偏差就不一样，
所以你没有办法通过"多测几次取平均"把它消掉：每一次测量对应一个
**无法重复的状态**，没有可平均的对象。

这直接决定了：两个时间点之间的「减脂」结论，门槛是
    √2 × 3 个百分点 × 体重 ≈ 4.2% × 78 ≈ 3.3 kg
而不是百分比算出来的那零点几公斤。
"""

import datetime as _dt
import json

import numpy as np

from .loaders import attach_day

# 默认轨迹：减脂 → 平台 → 继续减。
# 平台期是刻意加的 —— 真实的减脂从来不是一条直线，
# 而"平台"恰恰是最容易被误读成"我失败了"的一段。
DEFAULT_SEGMENTS = (
    {"days": 45, "fat_rate": -0.035, "note": "减脂期"},
    {"days": 30, "fat_rate": 0.000, "note": "平台期"},
    {"days": 45, "fat_rate": -0.020, "note": "继续减"},
)

START_DATE = _dt.date(2026, 1, 1)


def generate(
    segments=DEFAULT_SEGMENTS,
    seed=0,
    start_weight=80.0,
    start_fat_pct=25.0,
    te_weight=0.5,
    method_bf_pct=3.0,
    measure_prob=0.75,
    contamination_rate=0.10,
    first_measure_day=0,
    lean_cut_rate=-0.006,
    lean_gain_rate=0.004,
):
    """生成一段序列，返回 (records, truth)。

    records 是可以直接写成 CSV 的测量记录；
    truth 是同一段时期的**真实值**，评测时当标准答案用。

    参数
        segments           分段线性的真实脂肪变化，见 DEFAULT_SEGMENTS
        te_weight          体重的日间波动标准差（kg）—— 随机的
        method_bf_pct      体脂率的**方法误差**标准差（百分点）—— 系统性的
        measure_prob       每天真的去称的概率（模拟不规则采样）
        contamination_rate 有多大比例的测量是"不该拿来比"的（晚上/练后/酒后）
    """
    rng = np.random.default_rng(seed)

    total_days = sum(s["days"] for s in segments)

    # ---------------------------------------------------------- 真实轨迹
    true_fat, true_lean = [], []
    fat = start_weight * start_fat_pct / 100.0
    lean = start_weight - fat
    changepoints = []
    day = 0
    for seg in segments:
        changepoints.append({"day": day, "note": seg["note"]})
        for _ in range(seg["days"]):
            true_fat.append(fat)
            true_lean.append(lean)
            fat += seg["fat_rate"]
            # 减脂期会带一点瘦体重流失，平台期略有回补 —— 这是常见现象。
            # 做「真实变化为零」的对照组时把这两个速率都设成 0。
            if seg["fat_rate"] < 0:
                lean += lean_cut_rate
            else:
                lean += lean_gain_rate
            day += 1

    true_fat = np.array(true_fat)
    true_lean = np.array(true_lean)
    true_weight = true_fat + true_lean
    true_pct = true_fat / true_weight * 100.0

    # ---------------------------------------------------------- 采样与观测
    records, truth_rows = [], []
    contaminated = []

    for d in range(first_measure_day, total_days):
        if d > first_measure_day and rng.random() > measure_prob:
            continue

        # 体重的观测：真实值 + 随机日间波动
        w_obs = true_weight[d] + rng.normal(0.0, te_weight)
        # 体脂率的观测：真实值 + 方法误差（每次独立，因为每次的身体状态不同）
        pct_obs = true_pct[d] + rng.normal(0.0, method_bf_pct)

        is_contaminated = rng.random() < contamination_rate
        hour = 7.0 + rng.random() * 1.0
        trained = False
        alcohol = False
        note = ""
        date = START_DATE + _dt.timedelta(days=d)

        if is_contaminated:
            # 换一个"晚上练完刚吃完"的状态：水分少、阻抗低、体脂率假性偏低
            hour = 19.0 + rng.random() * 2.0
            trained = True
            w_obs -= 0.7 + rng.random() * 0.5
            pct_obs -= 1.0 + rng.random() * 1.0
            note = "晚上练后测"
            contaminated.append(len(records))

        # 骨骼肌 / 体水分：从瘦体重推，带上各自的误差。
        # 注意它们和体脂率来自**同一次阻抗测量**，所以在这个模型里
        # 它们的误差是相关的 —— 这也是为什么本项目**不做**"自洽校验"。
        smm_obs = true_lean[d] * 0.47 + rng.normal(0.0, 0.3)
        water_pct_obs = true_fat[d] * 0.0 + (true_lean[d] * 0.73) / w_obs * 100.0
        water_pct_obs += rng.normal(0.0, 0.8)

        records.append(
            {
                "date": date.isoformat(),
                "_date": date,
                "time": f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
                "_hour": hour,
                "weight_kg": round(float(w_obs), 2),
                "body_fat_pct": round(float(pct_obs), 1),
                "skeletal_muscle_kg": round(float(smm_obs), 2),
                "body_water_pct": round(float(water_pct_obs), 1),
                "fasted": (not is_contaminated),
                "trained_8h": trained,
                "alcohol_8h": alcohol,
                "note": note,
            }
        )
        truth_rows.append(
            {
                "day": d,
                "date": date.isoformat(),
                "true_weight_kg": round(float(true_weight[d]), 3),
                "true_fat_mass_kg": round(float(true_fat[d]), 3),
                "true_lean_mass_kg": round(float(true_lean[d]), 3),
                "true_body_fat_pct": round(float(true_pct[d]), 3),
            }
        )

    truth = {
        "seed": seed,
        "total_days": total_days,
        "te_weight": te_weight,
        "method_bf_pct": method_bf_pct,
        "segments": [dict(s) for s in segments],
        "changepoints": changepoints,
        "contaminated_indices": contaminated,
        "contamination_rate": contamination_rate,
        "n_records": len(records),
        "rows": truth_rows,
    }
    attach_day(records)
    return records, truth


# ---------------------------------------------------------------- 落盘
CSV_COLUMNS = (
    "date",
    "time",
    "weight_kg",
    "body_fat_pct",
    "skeletal_muscle_kg",
    "body_water_pct",
    "fasted",
    "trained_8h",
    "alcohol_8h",
    "note",
)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_csv(records, path):
    """把测量记录写成 CSV。没有第三方依赖，因为 csv 模块够用。"""
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for r in records:
            w.writerow([_cell(r.get(c)) for c in CSV_COLUMNS])
    return path


def write_truth(truth, path):
    """写出真相文件（评测的标准答案）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    return path


def main(argv=None):
    """命令行入口：python -m fitxray.synth

    生成示例数据到 data/ 下。
    """
    import argparse
    import os

    p = argparse.ArgumentParser(description="生成带 ground truth 的合成体成分数据")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--days", type=int, default=None, help="总天数（覆盖默认轨迹）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contamination", type=float, default=0.10)
    args = p.parse_args(argv)

    segments = DEFAULT_SEGMENTS
    if args.days:
        per = args.days // 3
        segments = tuple(
            {"days": (args.days - 2 * per if i == 2 else per),
             "fat_rate": s["fat_rate"], "note": s["note"]}
            for i, s in enumerate(DEFAULT_SEGMENTS)
        )

    records, truth = generate(
        segments=segments, seed=args.seed, contamination_rate=args.contamination
    )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "sample.csv")
    truth_path = os.path.join(args.out_dir, "ground_truth.json")
    write_csv(records, csv_path)
    write_truth(truth, truth_path)

    print(f"生成 {len(records)} 条测量记录（{truth['total_days']} 天）")
    print(f"  真实轨迹：{' → '.join(s['note'] for s in segments)}")
    print(f"  注入污染：{len(truth['contaminated_indices'])} 条"
          f"（{args.contamination:.0%} 的测量是「不该拿来比」的）")
    print(f"  注入的体重日间波动 TE = {truth['te_weight']} kg")
    print(f"  注入的体脂率方法误差     = {truth['method_bf_pct']} 个百分点")
    print(f"  → {csv_path}")
    print(f"  → {truth_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
