"""
在合成数据上量化评测。

一个只会说「变了」的工具没有价值，一个只会说「没变」的工具也没有。
所以这里不做「跑起来看着挺合理」的自证，而是拿**已知真相**的合成数据
去问四个能证伪的问题：

    ① 噪声估得准吗？
       估出来的 TE 和实际注入的 TE 差多少。

    ② 该降级的点抓出来了吗？
       注入的「晚上练后测」的点，召回率和精确率各是多少。

    ③ **会不会误报？**
       喂一段**真实变化为零**的纯噪声序列（体重和脂肪都没动），
       看它会不会报出「真实变化」。这一项最有说服力 ——
       一个把噪声当成果的工具，比一个什么都没检出的工具更糟。

    ④ 真的变了，检得出来吗？
       给一段足够大的真实变化，看它认不认。
       顺便验证反向命题：**不够大的变化，它应该老实说检不出来**。

四项合起来，才能说清楚这个工具的能力边界在哪。

跑法：
    python run.py --evaluate
"""

import numpy as np

from .comparability import screen
from .noise import noise_floor, series_points
from .schema import METRICS
from .synth import generate
from .verdict import METRIC_ORDER, VERDICT_REAL, verdicts

# 「什么算抓对了」的容差：估出来的 TE 落在实际值的 ±25% 之内
TE_TOLERANCE = 0.25

# 对照组：真实变化为零的轨迹
FLAT_SEGMENTS = (
    {"days": 60, "fat_rate": 0.0, "note": "不动"},
    {"days": 60, "fat_rate": 0.0, "note": "不动"},
)


def _realized_te(records, truth, obs_key, true_key):
    """这批数据里**实际实现**的噪声标准差。

    注意对比的是这个，不是名义上的注入值 —— 名义值只是生成参数，
    有限样本下真正落进数据里的是另一回事。拿名义值当标准答案，
    会把随机波动错记成估计器的偏差。
    """
    tt = {r["date"]: r for r in truth["rows"]}
    dev = [
        r[obs_key] - tt[r["date"]][true_key]
        for r in records
        if not r["note"] and r["date"] in tt
    ]
    if len(dev) < 3:
        return None
    return float(np.std(dev, ddof=1))


# ---------------------------------------------------------------- ① 噪声估计
def eval_noise(n_seeds=30, metrics=("weight_kg", "body_fat_pct")):
    """估出来的 TE 准不准。"""
    out = {}
    for metric in metrics:
        ratios = []
        for seed in range(n_seeds):
            recs, truth = generate(seed=seed)
            obs_key = metric
            true_key = "true_" + metric
            if true_key not in truth["rows"][0]:
                continue
            realized = _realized_te(recs, truth, obs_key, true_key)
            if not realized:
                continue
            s = screen(recs)
            clean = [recs[i] for i in s["clean_idx"]]
            nf = noise_floor(series_points(clean, metric), metric)
            if nf.get("te"):
                ratios.append(nf["te"] / realized)

        a = np.array(ratios)
        if not len(a):
            out[metric] = None
            continue
        out[metric] = {
            "label": METRICS[metric]["label"],
            "n": len(a),
            "mean_ratio": float(a.mean()),
            "sd_ratio": float(a.std()),
            "within_tolerance": float(np.mean(np.abs(a - 1.0) <= TE_TOLERANCE)),
        }
    return out


# ---------------------------------------------------------------- ② 门控
def eval_gating(n_seeds=30):
    """注入的污染点，抓出来多少（召回），抓错多少（精确）。"""
    hits = misses = false_alarms = 0
    for seed in range(n_seeds):
        recs, truth = generate(seed=seed)
        injected = set(truth["contaminated_indices"])
        s = screen(recs)
        got = {f["index"] for f in s["flagged"]}

        hits += len(got & injected)
        misses += len(injected - got)
        false_alarms += len(got - injected)

    total_injected = hits + misses
    total_flagged = hits + false_alarms
    return {
        "n_injected": total_injected,
        "n_flagged": total_flagged,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "recall": hits / total_injected if total_injected else None,
        "precision": hits / total_flagged if total_flagged else None,
    }


# ---------------------------------------------------------------- ③ 假阳性
def eval_false_positive(n_seeds=30):
    """真实变化为零时，有多少指标被判成「真实变化」。

    这是整套评测里最关键的一项。一个会把噪声当成成果的工具，
    比一个什么都不报的工具更糟 —— 后者只是没用，前者会让人
    照着假信号去改训练计划。

    同时给出「校正前 / 校正后」两个数字，让多重比较校正的效果可见。
    """
    raw_fp = corrected_fp = tests = 0
    per_metric = {m: 0 for m in METRIC_ORDER}

    for seed in range(n_seeds):
        recs, _ = generate(
            segments=FLAT_SEGMENTS,
            seed=seed,
            lean_cut_rate=0.0,
            lean_gain_rate=0.0,
        )
        res = verdicts(recs, alpha=0.05)
        for row in res["metrics"]:
            p = row.get("p_value")
            if p is None:
                continue
            tests += 1
            # 校正前：只看 p 值
            if p < 0.05 and row["change"] is not None:
                raw_fp += 1
                per_metric[row["metric"]] += 1
            # 校正后：看最终判定
            if row["verdict"] == VERDICT_REAL:
                corrected_fp += 1

    return {
        "n_seeds": n_seeds,
        "tests": tests,
        "raw_false_positives": raw_fp,
        "raw_rate": raw_fp / tests if tests else None,
        "corrected_false_positives": corrected_fp,
        "corrected_rate": corrected_fp / tests if tests else None,
        "per_metric": per_metric,
    }


# ---------------------------------------------------------------- ④ 检出
def eval_detection(n_seeds=20):
    """真实变化足够大 / 不够大时，工具分别怎么说。

    期望的结果是**双向**的：
        · 变化足够大  →  该认的认（真阳性）
        · 变化不够大  →  老实说检不出来（这是诚实，不是无能）

    如果第二行它也报「真实变化」，说明工具在过度声称。
    """
    cases = {
        "大幅减脂（约 12 kg，180 天）": (
            (
                {"days": 90, "fat_rate": -0.08, "note": "大幅减脂"},
                {"days": 90, "fat_rate": -0.08, "note": "大幅减脂"},
            ),
            "should_detect",
        ),
        "小幅减脂（约 2 kg，180 天）": (
            (
                {"days": 90, "fat_rate": -0.012, "note": "小幅减脂"},
                {"days": 90, "fat_rate": -0.012, "note": "小幅减脂"},
            ),
            "should_not_detect",
        ),
    }

    out = {}
    for name, (segments, expectation) in cases.items():
        detected = 0
        weights_detected = 0
        # 「体重说变了、脂肪说没变」的种子数 —— 这正是本项目要演示的那一幕
        split = 0
        for seed in range(n_seeds):
            recs, _ = generate(segments=segments, seed=seed)
            res = verdicts(recs, alpha=0.05)
            by = {r["metric"]: r for r in res["metrics"]}
            fat_real = by["fat_mass_kg"]["verdict"] == VERDICT_REAL
            w_real = by["weight_kg"]["verdict"] == VERDICT_REAL
            if fat_real:
                detected += 1
            if w_real:
                weights_detected += 1
            if w_real and not fat_real:
                split += 1
        out[name] = {
            "expectation": expectation,
            "n_seeds": n_seeds,
            "fat_mass_detected": detected,
            "fat_mass_rate": detected / n_seeds,
            "weight_detected": weights_detected,
            "weight_rate": weights_detected / n_seeds,
            "split": split,
        }
    return out


# ---------------------------------------------------------------- 汇总
def evaluate(n_seeds=30):
    """跑完四项，返回结构化结果。"""
    return {
        "n_seeds": n_seeds,
        "noise": eval_noise(n_seeds),
        "gating": eval_gating(n_seeds),
        "false_positive": eval_false_positive(n_seeds),
        "detection": eval_detection(max(10, n_seeds // 2)),
    }


def format_evaluation(rep):
    """把评测结果排成终端报告。"""
    L = []
    L.append("=" * 78)
    L.append("fitxray —— 量化评测（在已知真相的合成数据上）")
    L.append("=" * 78)
    L.append(f"每个场景跑 {rep['n_seeds']} 个随机种子")
    L.append("")

    # ①
    L.append("-" * 78)
    L.append("① 噪声估计准不准")
    L.append("-" * 78)
    L.append("   （「估计值 / 实际实现值」的分布，越接近 1.000 越好）")
    L.append("")
    L.append(f"   {'指标':<10}{'均值':>10}{'标准差':>10}{'落在±25%内':>14}")
    for metric, r in rep["noise"].items():
        if not r:
            continue
        L.append(
            f"   {r['label']:<10}{r['mean_ratio']:>10.3f}"
            f"{r['sd_ratio']:>10.3f}{r['within_tolerance']:>13.0%}"
        )
    L.append("")

    # ②
    g = rep["gating"]
    L.append("-" * 78)
    L.append("② 该降级的测量点，抓出来了吗")
    L.append("-" * 78)
    L.append(f"   注入 {g['n_injected']} 个污染点，标记了 {g['n_flagged']} 个")
    L.append(f"   召回率 {g['recall']:.0%}   精确率 {g['precision']:.0%}")
    if g["false_alarms"]:
        L.append(
            f"   （{g['false_alarms']} 个误标 —— 其中包含体重突跳判据的副产品，"
            f"那部分不一定是真误报）"
        )
    L.append("")

    # ③
    fp = rep["false_positive"]
    L.append("-" * 78)
    L.append("③ 真实变化为零时，会不会误报")
    L.append("-" * 78)
    L.append(f"   {fp['n_seeds']} 个「什么都没动」的序列 × "
             f"{len(METRIC_ORDER)} 个指标 = {fp['tests']} 次检验")
    L.append("")
    L.append(f"   只看 p 值（不校正）  {fp['raw_false_positives']:>3} 次误报"
             f"   误报率 {fp['raw_rate']:.1%}")
    L.append(f"   多重比较校正之后    {fp['corrected_false_positives']:>3} 次误报"
             f"   误报率 {fp['corrected_rate']:.1%}")
    L.append("")
    if fp["raw_false_positives"] > fp["corrected_false_positives"]:
        L.append("   → 校正把误报压下去了。这就是为什么判定模块里那一步不能省。")
    L.append("")

    # ④
    L.append("-" * 78)
    L.append("④ 真的变了，检得出来吗")
    L.append("-" * 78)
    for name, r in rep["detection"].items():
        L.append("")
        L.append(f"   {name}")
        L.append(f"     脂肪质量被判「真实变化」：{r['fat_mass_rate']:.0%}"
                 f"（{r['fat_mass_detected']}/{r['n_seeds']}）")
        L.append(f"     体重被判「真实变化」：    {r['weight_rate']:.0%}"
                 f"（{r['weight_detected']}/{r['n_seeds']}）")
        if r["expectation"] == "should_detect" and r["fat_mass_rate"] < 0.5:
            L.append("     ⚠ 大幅变化都没检出来，说明阈值过严")
        if r["expectation"] == "should_not_detect" and r["fat_mass_rate"] > 0.2:
            L.append("     ⚠ 小幅变化被判成真的了，说明工具在过度声称")

    # 这一段是整份评测的落点：同一批数据，体重说「变了」，
    # 脂肪质量说「没变」。两个结论都对 —— 它们量的不是同一个东西。
    split = max(
        (r for r in rep["detection"].values() if r["expectation"] == "should_not_detect"),
        key=lambda r: r["split"],
        default=None,
    )
    if split and split["split"]:
        L.append("")
        L.append("   " + "·" * 74)
        L.append("   最该看的一行：")
        L.append(
            f"   小幅减脂那组里，有 {split['split']}/{split['n_seeds']} 个种子出现"
            f"「体重判真实变化、脂肪质量判在噪声内」。"
        )
        L.append(
            "   两个结论都是对的 —— 体重那 2 kg 里主要是水和瘦体重，"
            "真正想减的脂肪还埋在噪声底下。"
        )
        L.append(
            "   这正是普通体脂秤曲线骗人的地方：它只画体重，"
            "你就把整条下行的线都当成了减脂成果。"
        )
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="在合成数据上量化评测 fitxray")
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args(argv)
    print(format_evaluation(evaluate(n_seeds=args.seeds)))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
