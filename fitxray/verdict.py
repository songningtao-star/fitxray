"""
综合判定 —— 把噪声地板、可比性、误差传播三样东西合成一句话。

对每个指标回答同一个问题：**这个变化，是真的吗？**

---- 判定的算法 ----

    ① 取首尾各三个**可比点**的中位数当起点和终点
       （不取单个点 —— 一个坏天气就能把结论带偏）

    ② 算变化量和它的典型误差
       实测指标：σ(Δ) = √2 × TE
       派生指标：TE 先从输入量按公式传过来，再 √2
                 （见 schema.derived_te 里那段关于「稳定偏差会抵消」的推理）

    ③ 拿变化量和阈值比
       |Δ| > 1.96·σ  →  真实变化
       |Δ| > 1.28·σ  →  临界（比典型误差大，但还没到 95% 那条线）
       否则           →  在噪声内

    ④ 多重比较校正（BH-FDR）
       要在好几个指标上同时做判定。每个都用 α=0.05，跑得多了
       总会撞上一个假阳性 —— 这是**统计上的**「数字太多」问题，
       而「数字太多」恰恰是这个项目要解决的问题本身。
       所以这里显式校正一遍，报告里也会写出来。

---- 一个自指的讽刺，顺手做成了卖点 ----

    项目的立论是「别再给我一堆数字了」。可是判定模块自己就引入了
    一堆假设检验，于是又冒出多重比较问题。**显式处理它，本身就是
    论点的一部分** —— 一个连自己的统计偏差都不检查的工具，
    没资格教别人怎么看待数字。
"""

import math

from .comparability import LEVEL_BAD, screen
from .noise import noise_floor, series_points
from .schema import IGNORE_METRICS, METRICS, derived_te, value_and_uncertainty

# ---------------------------------------------------------------- 判定等级
VERDICT_REAL = "真实变化"
VERDICT_BORDER = "临界"
VERDICT_NOISE = "在噪声内"
VERDICT_IGNORE = "该忽略"
VERDICT_UNKNOWN = "数据不足"

# 阈值的两个倍数
Z_95 = 1.96  # 95% —— 越过它才叫「真实变化」
Z_80 = 1.28  # 80% —— 越过它叫「临界」

# 首尾各取几个点的中位数
EDGE_N = 3

# 报告里指标的显示顺序
METRIC_ORDER = (
    "weight_kg",
    "body_fat_pct",
    "fat_mass_kg",
    "lean_mass_kg",
    "skeletal_muscle_kg",
    "body_water_pct",
)

# 需要估噪声地板的（派生量的 TE 是从这些传过去的）
DIRECTLY_MEASURED = ("weight_kg", "body_fat_pct", "skeletal_muscle_kg", "body_water_pct")

DEFAULT_ALPHA = 0.05


# ---------------------------------------------------------------- 小工具
def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _two_sided_p(z):
    """标准正态的双侧 p 值。不用 scipy，math.erf 就够了。"""
    z = abs(z)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _bh(pvals):
    """Benjamini-Hochberg FDR 校正，返回 q 值（顺序与输入一致）。

    做法：把 p 值从小到大排，第 k 小的 p 值乘以 n/k，然后从大到小
    取累计最小值。这样得到的 q 值是「在控制假发现率 5% 的前提下，
    这个假设还能不能站住」。
    """
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    running = 1.0
    for pos in range(n - 1, -1, -1):
        i = order[pos]
        running = min(running, pvals[i] * n / (pos + 1))
        q[i] = min(1.0, running)
    return q


# ---------------------------------------------------------------- 首尾取值
def _edges(records, metric, n=EDGE_N):
    """取首尾各 n 个可比点的中位数，作为起点和终点。

    返回 None 表示这个指标的可比点不够，做不了判定。
    """
    rows = []
    for r in records:
        val, unc = value_and_uncertainty(r, metric)
        if val is None:
            continue
        rows.append({"day": r["_day"], "value": val, "unc": unc, "record": r})

    if len(rows) < 2:
        return None

    rows.sort(key=lambda x: x["day"])
    k = min(n, max(1, len(rows) // 3))
    head, tail = rows[:k], rows[-k:]

    return {
        "start": {
            "value": _median([x["value"] for x in head]),
            "day": head[0]["day"],
            "date": head[0]["record"]["date"],
            "n": len(head),
        },
        "end": {
            "value": _median([x["value"] for x in tail]),
            "day": tail[-1]["day"],
            "date": tail[-1]["record"]["date"],
            "n": len(tail),
        },
        "ref_record": tail[-1]["record"],  # 传播系数用最近的一条，代表当前体型
        "gap_days": tail[-1]["day"] - head[0]["day"],
        "n_points": len(rows),
    }


# ---------------------------------------------------------------- 单指标判定
def verdict_for(metric, records, te_map, alpha=DEFAULT_ALPHA, edge_n=EDGE_N):
    """给一个指标做判定。

    参数
        records  可比点（已筛过的干净记录）
        te_map   {指标: 日间典型误差}，由 noise.noise_floor 估出来
    """
    spec = METRICS.get(metric)
    if spec is None:
        raise ValueError(f"不认识的指标：{metric}")

    label, unit = spec["label"], spec["unit"]
    base = {
        "metric": metric,
        "label": label,
        "unit": unit,
        "kind": spec["kind"],
    }

    e = _edges(records, metric, n=edge_n)
    if e is None:
        base.update(
            {
                "verdict": VERDICT_UNKNOWN,
                "change": None,
                "te": None,
                "threshold": None,
                "reason": "可比点太少（少于 2 个），做不了判定",
            }
        )
        return base

    change = e["end"]["value"] - e["start"]["value"]

    # 典型误差：实测直接取，派生的从输入量按公式传过来
    te = derived_te(e["ref_record"], metric, te_map)
    if te is None or te <= 0:
        base.update(
            {
                "verdict": VERDICT_UNKNOWN,
                "change": change,
                "te": None,
                "threshold": None,
                "start": e["start"],
                "end": e["end"],
                "gap_days": e["gap_days"],
                "n_points": e["n_points"],
                "reason": "估不出这个指标的典型误差（输入量的噪声地板缺失），"
                "无法判定",
            }
        )
        return base

    # 变化量的标准差：两个测量值相减，方差加倍
    sigma_change = math.sqrt(2.0) * te
    thr_95 = Z_95 * sigma_change
    thr_80 = Z_80 * sigma_change
    magnitude = abs(change)

    if magnitude > thr_95:
        verdict = VERDICT_REAL
    elif magnitude > thr_80:
        verdict = VERDICT_BORDER
    else:
        verdict = VERDICT_NOISE

    # p 值和阈值是同一套数学的两种说法，两个都给出来，
    # 让人既能看懂「过没过线」，也能看懂「过了多少」
    p = _two_sided_p(magnitude / sigma_change)

    # 绝对值那套不确定度（方法误差）—— 和判定无关，但报告里要提，
    # 因为它决定了「你那个 25% 到底是不是 25%」这个我们永远不回答的问题
    _, absolute_unc = value_and_uncertainty(e["ref_record"], metric)

    base.update(
        {
            "verdict": verdict,
            "change": change,
            "change_abs": magnitude,
            "direction": "up" if change > 0 else ("down" if change < 0 else "flat"),
            "te": te,
            "sigma_change": sigma_change,
            "threshold": thr_95,
            "threshold_80": thr_80,
            "p_value": p,
            "absolute_uncertainty": absolute_unc,
            "start": e["start"],
            "end": e["end"],
            "gap_days": e["gap_days"],
            "n_points": e["n_points"],
            "reason": _reason(label, unit, change, thr_95, te, verdict, e),
        }
    )
    return base


def _reason(label, unit, change, thr, te, verdict, e):
    """给判定配一句能核对的理由 —— 不是一句「效果不显著」。"""
    ratio = abs(change) / thr if thr else 0
    head = (
        f"{e['start']['date']} 的 {e['start']['value']:.2f} → "
        f"{e['end']['date']} 的 {e['end']['value']:.2f}（{label}）"
    )
    if verdict == VERDICT_REAL:
        return (
            f"{head}：变化 {change:+.2f} {unit}，"
            f"是判定阈值 {thr:.2f} 的 {ratio:.1f} 倍 —— 越过 95% 那条线了"
        )
    if verdict == VERDICT_BORDER:
        return (
            f"{head}：变化 {change:+.2f} {unit}，"
            f"超过了典型误差（{te:.2f}），但没到 95% 阈值 {thr:.2f}。"
            f"方向上像是真的，但还差一口气"
        )
    tail = ""
    if ratio < 0.5:
        # 离阈值差得远，说明不是"再多测几次就能看出来"的问题
        tail = " —— 不是数据不够，是这台设备达不到那个分辨率"
    return (
        f"{head}：变化 {change:+.2f} {unit}，"
        f"只有判定阈值 {thr:.2f} 的 {ratio * 100:.0f}%。这个幅度和噪声分不开{tail}"
    )


# ---------------------------------------------------------------- 全指标判定
def verdicts(records, screen_result=None, alpha=DEFAULT_ALPHA, edge_n=EDGE_N):
    """跑完整条流程：筛查 → 估噪声 → 逐指标判定 → 多重比较校正。

    参数
        records        全部测量记录（loaders.load_measurements 或 synth.generate 出来的）
        screen_result  已经跑过的 comparability.screen 结果，省得重复算
        alpha          FDR 水平

    返回
        {
          "metrics":  [每个指标一个判定 dict],
          "ignored":  [该忽略的指标及理由],
          "n_points": int,        # 总测量次数
          "n_clean":  int,        # 可比点数量
          "flagged":  [被降级的点],
          "noise":    {指标: 噪声地板 dict},
          "alpha":    float,
          "te_before": float | None,
          "te_after":  float | None,
        }
    """
    if screen_result is None:
        screen_result = screen(records)

    clean = [records[i] for i in screen_result["clean_idx"]]

    # --- 噪声地板 ---
    noise_map = {}
    te_map = {}
    for m in DIRECTLY_MEASURED:
        pts = series_points(clean, m)
        nf = noise_floor(pts, m)
        noise_map[m] = nf
        if nf.get("te"):
            te_map[m] = nf["te"]

    # --- 逐指标判定 ---
    rows = []
    for m in METRIC_ORDER:
        rows.append(verdict_for(m, clean, te_map, alpha=alpha, edge_n=edge_n))

    # --- 多重比较校正 ---
    # 只对「有可能显著」的指标做校正，也就是参与判定的那些。
    # 数据不足的直接跳过 —— 它们本来就没参与检验，不该占一个检验名额。
    testable = [r for r in rows if r.get("p_value") is not None]
    qvals = _bh([r["p_value"] for r in testable])
    downgraded = []
    for r, q in zip(testable, qvals):
        r["q_value"] = q
        # 校正之后站不住的「真实变化」，降到「临界」并写明原因
        if r["verdict"] == VERDICT_REAL and q >= alpha:
            r["verdict"] = VERDICT_BORDER
            r["reason"] += (
                f"。但多重比较校正后 q = {q:.3f} ≥ {alpha} —— "
                f"同时检验 {len(testable)} 个指标时，这个显著性可能只是撞上的"
            )
            downgraded.append(r["metric"])
        elif q is not None:
            r["reason"] += f"（多重比较校正后 q = {q:.3f}）"

    # --- 该忽略的指标 ---
    ignored = [
        {
            "key": k,
            "label": v[0],
            "why": v[1],
            "present": k in records[0] if records else False,
        }
        for k, v in IGNORE_METRICS.items()
    ]

    return {
        "metrics": rows,
        "ignored": ignored,
        # 原始记录一并带上：报告要画时间序列，评测要拿它对真相，
        # 让调用方自己再传一遍容易传错顺序。
        "records": records,
        "device_id": (records[0].get("device_id") if records else None),
        "n_points": len(records),
        "n_clean": len(clean),
        "flagged": screen_result["flagged"],
        "noise": noise_map,
        "alpha": alpha,
        "te_before": screen_result.get("te_before"),
        "te_after": screen_result.get("te_after"),
        "downgraded": downgraded,
    }


def _plain(text):
    """去掉理由文字里的 Markdown 加粗标记。

    那些 `**` 是给 report.py 排版用的（HTML 里会变成 <b>）。
    但同一批理由也会走到终端 —— 终端不认识 Markdown，
    不处理的话用户会看到 `**确定性函数**` 这种裸标记。

    报告那边有 `_rich` 负责转成 <b>，这里负责把它抹掉，两边对应。
    """
    return text.replace("**", "") if text else text


def format_verdicts(result):
    """把判定结果排成终端报告。"""
    L = []
    L.append("=" * 78)
    L.append("fitxray —— 体成分变化归因")
    L.append("=" * 78)
    L.append(
        f"测量 {result['n_points']} 次，其中 {result['n_clean']} 次可比"
        f"（{len(result['flagged'])} 次被降级）"
    )
    if result.get("te_before") is not None and result.get("te_after") is not None:
        b, a = result["te_before"], result["te_after"]
        L.append(f"筛查前后体重噪声地板：{b:.3f} → {a:.3f} kg")
    L.append("")

    L.append("-" * 78)
    L.append("指标判定")
    L.append("-" * 78)
    icon = {
        VERDICT_REAL: "✅",
        VERDICT_BORDER: "◐ ",
        VERDICT_NOISE: "○ ",
        VERDICT_UNKNOWN: "… ",
    }
    for r in result["metrics"]:
        L.append("")
        L.append(f"{icon.get(r['verdict'], '  ')} {r['label']:<8} {r['verdict']}")
        L.append(f"     {_plain(r['reason'])}")

    if result["ignored"]:
        L.append("")
        L.append("-" * 78)
        L.append("这几项不用看")
        L.append("-" * 78)
        for ig in result["ignored"]:
            L.append(f"  ✕ {ig['label']}")
            L.append(f"      {_plain(ig['why'])}")

    if result["flagged"]:
        L.append("")
        L.append("-" * 78)
        L.append(f"被降级的测量（{len(result['flagged'])} 次，只降级不删除）")
        L.append("-" * 78)
        for f in result["flagged"][:12]:
            L.append(f"  {f['date']}  {_plain('; '.join(f['reasons']))}")
        if len(result["flagged"]) > 12:
            L.append(f"  …… 还有 {len(result['flagged']) - 12} 次")

    if result.get("downgraded"):
        L.append("")
        L.append(
            f"注意：{len(result['downgraded'])} 个指标在多重比较校正后从"
            f"「真实变化」降级 —— 同时看好几个指标时，总有一个会撞上显著性。"
        )

    L.append("")
    L.append("=" * 78)
    return "\n".join(L)
