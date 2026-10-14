"""
噪声地板 —— 从你自己的数据里估出来，不写死常量。

大多数工具会用一张固定的阈值表：「体重变化超过 1 kg 才算数」。
这个做法有两个问题：

    1. 每台设备、每个人的测量习惯不一样，凭什么用同一个阈值？
    2. 噪声**不是常数** —— 相邻两天的噪声，和隔三十天的噪声，不是一回事。

第二条尤其重要。体重序列里除了随机跳动，还混着缓慢的生物漂移
（水分、糖原、肠内容物）。间隔拉长，漂移累积，两个点之间能出现的
差异本来就更大。用一个 σ 去卡所有间隔，会把隔得很远的点判成"显著"。

所以这里估的是一条曲线，不是单个数字。

---- 三个关键设计 ----

**① 用点对，不用相邻点。**
数序列是不规则采样的（有人天天称，有人一周称两次）。按"第 k 个和第 k+1 个"
做差分，lag 在时间上是不均匀的。所以这里取**所有点对**，按它们之间
真实的**天数间隔**分箱。

**② 用中位数，不用标准差。**
一次真实的水分波动（生病、高钠饮食）会在标准差里留下很大的痕迹，
把整个噪声地板抬上去 —— 结果就是工具变得过于保守，真的变化也报不出来。
改用中位数绝对偏差（MAD），一次异常事件影响不了它。

**③ 报 LSC，不报 σ。**
LSC（最小显著变化）= 2.77 × TE，是运动科学里的标准说法
（Hopkins 的典型误差 / ISAK 技术误差）。它直接回答用户真正关心的问题：
「小于多少的变化，和噪声无法区分？」
比报一个标准差可行动得多。
"""

import math

import numpy as np

# 变差函数的档位代表间隔（天）。间隔超过 30 天的点对不参与拟合 ——
# 那时候真实趋势的变化已经盖过噪声，估出来的不是噪声。
#
# 注意是**代表值**不是边界。分箱时取相邻代表值的中点当边界 ——
# 因为 `_day` 带着小时的小数部分，两个"隔 1 天"的点实际算出来是
# 0.96~1.04 天。要是直接拿整数当边界，这批点会被劈到两个箱里，
# 每一箱都只剩一半数据。
DEFAULT_BINS = (1, 2, 3, 5, 7, 10, 14, 21, 30)

# 估噪声需要的最少可比点。少于这个数就不估，用下面的默认值，
# 并在报告里**明确标注**用的是默认值。
MIN_POINTS = 8

# 数据不够时的兜底默认值（日间典型误差 TE，单位同指标）。
# 这些是启发式的，**没有文献出处**，报告里会标出来。
# 它们只在你自己的数据不足时才会被用到。
DEFAULT_TE = {
    "weight_kg": 0.5,
    "body_fat_pct": 1.0,
    "fat_mass_kg": 0.6,
    "lean_mass_kg": 0.6,
    "skeletal_muscle_kg": 0.3,
    "body_water_pct": 1.0,
}

# MAD → 标准差的一致性系数（正态分布下）。
# 中位数绝对偏差天然比标准差小，乘这个系数才换算到同一量纲。
MAD_TO_SD = 1.4826

# LSC = 2.77 × TE 里的 2.77。
# 来历：变化量的 95% 一致性界限 = 1.96 × √2 × TE ≈ 2.77 × TE。
# 那个 √2 是因为「两个测量值相减」，方差要加倍。
LSC_FACTOR = 2.77


# ---------------------------------------------------------------- 点对
def series_points(records, key):
    """把测量记录转成 [(天数, 值), ...]，按时间排序，跳过缺失值。

    天数是从第一条记录算起的浮点天数（同一天的不同时刻也能区分开）。
    返回的列表可以直接喂给下面的所有函数。
    """
    pts = []
    for r in records:
        v = r.get(key)
        if v is None:
            continue
        pts.append((r["_day"], float(v)))
    pts.sort(key=lambda p: p[0])
    return pts


def pair_diffs(points, max_gap=None):
    """取所有点对，返回 [(间隔天数, 差值), ...]。

    用点对而不是相邻点，是因为采样不规则 —— 见模块开头设计①。
    点数 n 时点对数是 n(n-1)/2，几百个点也就是几万对，够用。
    """
    out = []
    n = len(points)
    for i in range(n):
        ti, xi = points[i]
        for j in range(i + 1, n):
            tj, xj = points[j]
            gap = tj - ti
            if gap <= 0:
                continue
            if max_gap is not None and gap > max_gap:
                continue
            out.append((gap, xj - xi))
    return out


# ---------------------------------------------------------------- 稳健估计
def robust_sigma(diffs):
    """从一组差分稳健地估出**单次测量**的标准差。

        x 的两次测量相减 → 差值方差 = 2σ²
        所以 σ = SD(差值) / √2

    把标准差换成中位数绝对偏差就是这个函数的全部内容。注意
    MAD 本身算的是「偏差的中位数」，直接开方没有意义，要先乘
    MAD_TO_SD 换算回标准差的量纲，再除 √2。
    """
    if len(diffs) < 2:
        return None
    mad = float(np.median(np.abs(np.asarray(diffs, dtype=float))))
    return MAD_TO_SD * mad / math.sqrt(2.0)


def lsc(te):
    """由典型误差算最小显著变化。"""
    return LSC_FACTOR * te


# ---------------------------------------------------------------- 变差函数
def _bin_edges(bins):
    """把档位代表值换算成分箱边界：相邻代表值的中点，末尾封在最大间隔。

    bins=(1,2,3,5,...,30) → edges=(0, 1.5, 2.5, 4.0, ..., 25.5, 30.0)
    """
    edges = [0.0]
    for a, b in zip(bins, bins[1:]):
        edges.append((a + b) / 2.0)
    edges.append(float(bins[-1]))
    return edges


def variogram(points, bins=DEFAULT_BINS):
    """半变差函数：每个间隔档位上估出来的噪声标准差。

    返回 [{"gap_lo","gap_hi","sigma","n"}, ...]，间隔从近到远。

    读法：
        曲线**平**       → 噪声是独立的，间隔多远都一样 → 用一个 σ 就行
        曲线**往上走**   → 存在随机游走式的生物漂移 → 间隔越远越不可比
                            判定时必须用对应间隔的 σ，不能用一个数通吃

    实践上最可信的是**最近的那一档**（真实趋势还没累积起来），
    所以 noise_floor 会把整条曲线一起报出来，让你自己看它平不平。
    """
    diffs = pair_diffs(points, max_gap=bins[-1])
    if not diffs:
        return []

    rows = []
    edges = _bin_edges(bins)
    for rep, lo, hi in zip(bins, edges, edges[1:]):
        bucket = [d for gap, d in diffs if lo < gap <= hi]
        if len(bucket) < 3:
            continue
        sigma = robust_sigma(bucket)
        if sigma is None:
            continue
        rows.append(
            {"gap": rep, "gap_lo": lo, "gap_hi": hi, "sigma": sigma, "n": len(bucket)}
        )
    return rows


def sigma_at(curve, gap_days):
    """给定间隔，从变差函数曲线插值出对应的噪声标准差。

    曲线为空（数据不够）时返回 None，由调用方决定兜底。
    间隔落在第一档之内或之外时，就近取端点。
    """
    if not curve:
        return None

    if gap_days <= curve[0]["gap_hi"]:
        return curve[0]["sigma"]
    if gap_days >= curve[-1]["gap_hi"]:
        return curve[-1]["sigma"]

    for lo_row, hi_row in zip(curve, curve[1:]):
        if lo_row["gap_hi"] <= gap_days <= hi_row["gap_hi"]:
            span = hi_row["gap_hi"] - lo_row["gap_hi"]
            if span <= 0:
                return hi_row["sigma"]
            w = (gap_days - lo_row["gap_hi"]) / span
            return lo_row["sigma"] * (1 - w) + hi_row["sigma"] * w

    return curve[-1]["sigma"]


def is_random_walk(curve):
    """曲线是不是明显往上走（存在累积的生物漂移）。

    判据宽松：最远一档的 σ 比最近一档大 50% 以上才算。
    宁可不报，也不要报一个不稳的结论。
    """
    if len(curve) < 3:
        return False
    first = curve[0]["sigma"]
    last = curve[-1]["sigma"]
    if first <= 0:
        return False
    return (last / first) > 1.5


# ---------------------------------------------------------------- 主入口
def noise_floor(points, metric, min_points=MIN_POINTS, bins=DEFAULT_BINS):
    """给一条指标序列估出噪声地板。

    参数
        points  [(天数, 值), ...]（已经过可比性筛选的**干净点**）
        metric  指标名，用于在数据不够时取兜底默认值

    返回
        {
          "metric": str,
          "te": float,          # 典型误差（单次测量的标准差）
          "lsc": float,         # 最小显著变化 = 2.77 × TE
          "n_points": int,
          "n_pairs": int,
          "source": "estimated" | "default",
          "curve": [...],       # 变差函数曲线
          "random_walk": bool,  # 噪声是否随间隔增长
          "note": str,          # 给人看的一句话说明
        }

    source == "default" 时，te 来自 DEFAULT_TE 表，报告里必须标出来 ——
    不能让用户以为这个数字是从他的数据里算的。
    """
    n = len(points)

    if n < min_points:
        te = DEFAULT_TE.get(metric)
        if te is None:
            return {
                "metric": metric,
                "te": None,
                "lsc": None,
                "te_gap": None,
                "n_points": n,
                "n_pairs": 0,
                "source": "default",
                "curve": [],
                "random_walk": False,
                "note": f"只有 {n} 个可比点（需要 {min_points} 个），"
                f"且该指标没有兜底默认值，无法给出噪声地板",
            }
        return {
            "metric": metric,
            "te": te,
            "lsc": lsc(te),
            "te_gap": None,
            "n_points": n,
            "n_pairs": 0,
            "source": "default",
            "curve": [],
            "random_walk": False,
            "note": f"只有 {n} 个可比点（需要 {min_points} 个），"
            f"用的是**默认噪声值**，不是你自己的数据估出来的",
        }

    curve = variogram(points, bins=bins)
    diffs = pair_diffs(points, max_gap=bins[-1])

    # TE 取**最近的一档**，不是全部点对。
    #
    # 这个选择是量出来的，不是拍脑袋。用 30 个随机种子做过对照实验
    # （合成数据，已知注入的真实噪声），比较「估计值 / 实际实现值」：
    #
    #                     平缓减脂        激进减脂（趋势远大于噪声）
    #     最近一档        1.016 ± 0.17    0.996 ± 0.17      无偏
    #     全部点对        1.179 ± 0.08    3.252 ± 0.25      ★ 崩了
    #
    # 全部点对在平缓数据上看着更稳（标准差更小），但那是个陷阱：
    # 它的稳定性来自样本量大，代价是把长间隔上**真实的身体变化**
    # 当成了噪声。趋势一旦超过噪声（激进减脂就是这么回事），
    # 它会把噪声地板高估三倍多 —— 结果就是工具变成一台
    # 「什么都判不显著」的机器，真的减脂也被它说成噪声。
    #
    # 宁可用一个方差大一点、但两种情形下都无偏的估计。
    # 保守方向的偏差可以接受（本项目本来就主张不要过度声称），
    # 系统性高估三倍不可接受。
    #
    # 退化情形：如果你每周才称一次，最短的一档就是 7 天，
    # 那这一档里已经混进一周的真实变化了 —— 这种情况 note 里会写明
    # 「基于约 7 天间隔」，让你知道这个 TE 是偏保守的。
    if curve:
        te = curve[0]["sigma"]
        te_gap = curve[0]["gap"]
    else:
        te = robust_sigma([d for _, d in diffs])
        te_gap = None

    if te is None or te <= 0:
        te = DEFAULT_TE.get(metric)
        return {
            "metric": metric,
            "te": te,
            "lsc": lsc(te) if te else None,
            "te_gap": None,
            "n_points": n,
            "n_pairs": len(diffs),
            "source": "default",
            "curve": curve,
            "random_walk": False,
            "note": "差分全为 0，噪声估不出来（数据可能有大量重复值），改用默认值",
        }

    walk = is_random_walk(curve)
    note = f"从你自己的 {n} 个可比点估出来"
    if te_gap is not None:
        note += f"（基于约 {te_gap:.0f} 天间隔的相邻测量）"
    if walk:
        note += (
            "；噪声随间隔增大，存在累积的生物漂移 —— "
            "隔得越远的两个点，可比性越差"
        )

    return {
        "metric": metric,
        "te": te,
        "lsc": lsc(te),
        "te_gap": te_gap,
        "n_points": n,
        "n_pairs": len(diffs),
        "source": "estimated",
        "curve": curve,
        "random_walk": walk,
        "note": note,
    }
