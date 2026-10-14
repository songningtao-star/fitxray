"""
数据模型、派生量、不确定度传播。

这是整个项目的理论基础。有一件事必须先讲清楚，因为它决定了后面所有的算法：

    **不确定度有两种，它们回答的是两个完全不同的问题。**

    ┌─ 方法不确定度（±3 个百分点这种）────────────────┐
    │  回答：「你那个 24.5% 到底是不是 24.5%？」      │
    │  性质：系统性。来自设备和回归公式，不随测量次数减少。│
    │  用途：**只用于绝对值**。我们永远不声称绝对值准。 │
    └──────────────────────────────────────────┘

    ┌─ 日间典型误差 TE（±零点几这种）──────────────┐
    │  回答：「你减掉的这 0.88 公斤是不是真的？」     │
    │  性质：随测量状态漂移，可以从你自己的数据估出来。│
    │  用途：**判定变化**。                          │
    └──────────────────────────────────────────┘

    为什么「变化」用 TE 而不是方法误差？关键一步推理：

        **稳定的方法偏差，在差值里会自动抵消。**

        同一个人的同一台秤，那个固定的偏移量在
        `脂肪(t2) − 脂肪(t1)` 里被减了两次，直接没了。
        所以拿 ±3 个百分点去卡「变化」，是把同一个东西算了两遍。

    但 TE 传进派生量要走一遍公式 —— 脂肪质量 = 体重 × 体脂率，
    两个输入量的 TE 各自带一份进去，在体脂率那一项上被放大。
    具体算下来，一条 78 kg / 25% 的序列：

        体重    TE ≈ 0.5 kg        →  对脂肪质量几乎没影响
        体脂率  TE ≈ 2.9 个百分点   →  这一项**完全主导**

        合成 TE(脂肪质量) ≈ 2.3 kg
        判定阈值 = 2.77 × 2.3 ≈ 6.3 kg

    也就是说：**BIA 要看出 6 公斤以上的脂肪变化才有把握。**
    那「减了 0.88 公斤」当然在噪声里 —— 而且不是因为数据不够，
    是这台设备本身达不到那个分辨率。

模块内容：
    METHOD_UNCERTAINTY  方法不确定度表（管绝对值，每一格都带依据）
    METRICS             指标注册表（哪些是实测、哪些是派生）
    IGNORE_METRICS      该忽略的指标及理由
    fat_mass / lean_mass            派生量
    fat_mass_uncertainty            绝对值的（方法）不确定度
    derived_te                      变化的（典型误差）传播 ← 判定用这个
"""

import math

# ---------------------------------------------------------------- 不确定度表
# 这里每一个数字都是**系统性**误差的半宽 —— 也就是 90% 以上的读数会落在
# ±这个范围内，而且**多测几次不会让它变小**。
#
# 这一点必须和 noise.py 里估出来的「日间变异」分清楚：
#     方法不确定度  —— 系统性的，来自设备和回归公式    → 本表
#     日间变异      —— 随机的，可以从你自己的数据估出来 → noise.py
#
# 判定「变化是否真实」时，变化必须**同时**超过这两者。
METHOD_UNCERTAINTY = {
    "weight_kg": {
        "half_width": 0.1,
        "basis": "足部秤称重精度，厂商标称级别（一般 ±0.1~0.2 kg）",
        "source": "device",
    },
    "body_fat_pct": {
        "half_width": 3.0,
        "basis": (
            "生物电阻抗（BIA）与 DXA 的 SEE 约 3.5~5 个百分点，"
            "取保守下界 3.0。足部秤只让电流走下半身，误差通常比这个还大"
        ),
        "source": "literature-conservative",
    },
    "skeletal_muscle_kg": {
        "half_width": 1.5,
        "basis": (
            "厂商算法不公开，此处取保守估计。注意：这个值和体脂率来自"
            "**同一次阻抗测量**，所以两者误差相关，不能当独立量处理"
        ),
        "source": "assumed-conservative",
    },
    "body_water_pct": {
        "half_width": 2.5,
        "basis": "同上，厂商算法不公开，取保守估计",
        "source": "assumed-conservative",
    },
}

# ---------------------------------------------------------------- 指标注册表
# kind:
#     "measured" 设备直接读数（带方法不确定度）
#     "derived"  从读数算出来（不确定度要传播）
METRICS = {
    "weight_kg": {
        "label": "体重",
        "unit": "kg",
        "kind": "measured",
        "uncertainty_key": "weight_kg",
        "better": "context",  # 无所谓好坏，看目标
    },
    "body_fat_pct": {
        "label": "体脂率",
        "unit": "%",
        "kind": "measured",
        "uncertainty_key": "body_fat_pct",
        "better": "down",
    },
    "fat_mass_kg": {
        "label": "脂肪质量",
        "unit": "kg",
        "kind": "derived",
        "from": ["weight_kg", "body_fat_pct"],
        "better": "down",
    },
    "lean_mass_kg": {
        "label": "瘦体重",
        "unit": "kg",
        "kind": "derived",
        "from": ["weight_kg", "body_fat_pct"],
        "better": "up",
    },
    "skeletal_muscle_kg": {
        "label": "骨骼肌",
        "unit": "kg",
        "kind": "measured",
        "uncertainty_key": "skeletal_muscle_kg",
        "better": "up",
    },
    "body_water_pct": {
        "label": "体水分率",
        "unit": "%",
        "kind": "measured",
        "uncertainty_key": "body_water_pct",
        "better": "context",
    },
}

# ---------------------------------------------------------------- 该忽略的指标
# 项目的敌人是「数字太多」。体脂秤会给你一堆数字，其中一些**没有任何独立信息** ——
# 它们是别的数字套一个公式算出来的。这里把它们列出来，并说清楚为什么。
#
# 所有现有工具都把这些当卖点展示，没一个告诉你别看。
IGNORE_METRICS = {
    "body_age": (
        "身体年龄",
        "厂商拿体重/体脂率/年龄套一个回归公式算出来的，不是任何独立测量",
    ),
    "health_score": (
        "健康评分",
        "同上。而且这个分数在不同品牌之间不可比，因为公式是各自的商业秘密",
    ),
    "ideal_weight": (
        "理想体重",
        "从身高和一个固定 BMI 目标算出来的常数，跟你的实际数据无关",
    ),
    "bmr_kcal": (
        "基础代谢",
        "足部秤用的是 Katch-McArdle 类公式 BMR = 370 + 21.6 × 瘦体重，"
        "是瘦体重的**确定性函数** —— 瘦体重动了它就动，零独立信息",
    ),
    "visceral_fat_level": (
        "内脏脂肪等级",
        "足部秤测不到内脏，这个等级是从体脂率和年龄推的估算值，误差极大",
    ),
}


# ---------------------------------------------------------------- 派生量
def fat_mass(m):
    """脂肪质量（kg）= 体重 × 体脂率。

    注意这是对**估计值**的恒等式，不是对真值的 —— 见模块开头。
    """
    return m["weight_kg"] * m["body_fat_pct"] / 100.0


def lean_mass(m):
    """瘦体重（kg）= 体重 − 脂肪质量。

    这个是真正的恒等式（定义使然，不含任何额外假设）。
    但它仍然是对估计值的恒等式，误差照样会传进来。
    """
    return m["weight_kg"] - fat_mass(m)


# ---------------------------------------------------------------- 误差传播
def _rel(half_width, value):
    """相对不确定度 = 半宽 / 量值。"""
    if value == 0:
        return float("inf")
    return abs(half_width / value)


def propagate_product(factors):
    """乘除法的误差传播。

        z = a × b      →      (σz/z)² = (σa/a)² + (σb/b)²

    这是相对不确定度的「平方和开方」规则。对乘除链适用；
    加减法要用绝对量的平方和开方（见下面的 propagate_difference）。

    参数
        factors: [(值, 半宽), ...]

    返回
        (结果, 结果的绝对半宽)
    """
    result = 1.0
    rel2 = 0.0
    for value, half_width in factors:
        result *= value
        rel2 += _rel(half_width, value) ** 2
    return result, abs(result) * math.sqrt(rel2)


def propagate_difference(pairs):
    """加减法的误差传播。

        z = a − b      →      σz² = σa² + σb²

    参数
        pairs: [(值, 半宽), ...]（符号体现在值里）

    返回
        (结果, 结果的绝对半宽)
    """
    result = 0.0
    var = 0.0
    for value, half_width in pairs:
        result += value
        var += half_width ** 2
    return result, math.sqrt(var)


# ---------------------------------------------------------------- 派生量的不确定度
def _hw(key):
    return METHOD_UNCERTAINTY[key]["half_width"]


def fat_mass_uncertainty(m):
    """脂肪质量的绝对不确定度（kg）。

    脂肪质量 = 体重 × (体脂率/100)，两因子相乘，用相对不确定度平方和开方。

    代入一组典型数字感受一下量级：
        体重 78.2 kg，方法误差 ±0.1 kg  →  相对 0.13%
        体脂率 25.0%，方法误差 ±3.0 个点 →  相对 12.0%
    体脂率那一项**完全压倒**体重那一项。所以：

        σ(脂肪质量) ≈ 脂肪质量 × 12% ≈ 19.55 × 0.12 ≈ 2.35 kg
    """
    weight = m["weight_kg"]
    pct = m["body_fat_pct"]
    _, half_width = propagate_product(
        [
            (weight, _hw("weight_kg")),
            (pct / 100.0, _hw("body_fat_pct") / 100.0),
        ]
    )
    return half_width


def lean_mass_uncertainty(m):
    """瘦体重 = 体重 − 脂肪质量，用加减法传播。

    体重的误差 0.1 kg 相比脂肪质量的 2.35 kg 可以忽略，
    所以瘦体重的不确定度几乎等于脂肪质量的不确定度。
    """
    _, half_width = propagate_difference(
        [
            (m["weight_kg"], _hw("weight_kg")),
            (-fat_mass(m), fat_mass_uncertainty(m)),
        ]
    )
    return half_width


def value_and_uncertainty(m, metric):
    """取某个指标的 (值, 绝对不确定度)。实测和派生走不同的路。

    未知指标或缺失字段返回 (None, None) —— 由调用方决定怎么办。
    """
    if metric not in METRICS:
        return None, None

    if metric == "fat_mass_kg":
        return fat_mass(m), fat_mass_uncertainty(m)
    if metric == "lean_mass_kg":
        return lean_mass(m), lean_mass_uncertainty(m)

    if m.get(metric) is None:
        return None, None
    spec = METRICS[metric]
    key = spec.get("uncertainty_key")
    return m[metric], (_hw(key) if key else None)


def derived(m):
    """一次性算出所有派生量，返回一个新 dict（不改原记录）。"""
    out = dict(m)
    out["fat_mass_kg"] = fat_mass(m)
    out["lean_mass_kg"] = lean_mass(m)
    return out


# ---------------------------------------------------------------- 典型误差的传播
def derived_te(m, metric, te_map):
    """派生量的**典型误差**（TE）：把输入量的 TE 按公式传过去。

    这个函数和上面的 `fat_mass_uncertainty` 不是一回事，两者回答的是
    **两个不同的问题**：

        fat_mass_uncertainty   「这个**绝对值**准不准？」
                                → 用方法误差（±3 个百分点）

        derived_te             「这个**变化**是不是真的？」
                                → 用日间典型误差（从数据估出来的）

    为什么「变化」用 TE 而不是方法误差 —— 这是本项目最关键的一步推理：

        **稳定的方法偏差，在差值里会自动抵消。**

        同一个人的同一台秤，那个固定的偏移量在
        `脂肪(t2) − 脂肪(t1)` 里被减了两次，直接没了。
        真正会影响「变化」判定的，只有**随测量状态漂移**的那一部分
        （今天水合好、昨天练过腿），而那一部分已经包含在日间 TE 里。

    所以阈值 = 1.96 × √2 × TE = 2.77 × TE，**不需要再叠加方法误差**。
    叠加会把同一个东西算两遍，把工具变成一台「什么都判不显著」的机器。

    方法误差仍然重要 —— 但它管的是「你那个 24.5% 到底是不是 24.5%」，
    管不了「你是不是减了 0.88 公斤」。前者我们永远不声称。
    """
    te_w = te_map.get("weight_kg")
    te_pct = te_map.get("body_fat_pct")

    if metric == "fat_mass_kg":
        # 脂肪质量 = 体重 × 体脂率 / 100
        #   ∂/∂体重   = 体脂率/100
        #   ∂/∂体脂率 = 体重/100
        if te_w is None or te_pct is None:
            return None
        w, pct = m["weight_kg"], m["body_fat_pct"]
        return math.sqrt(
            (pct / 100.0) ** 2 * te_w ** 2 + (w / 100.0) ** 2 * te_pct ** 2
        )

    if metric == "lean_mass_kg":
        # 瘦体重 = 体重 × (1 − 体脂率/100)
        #   ∂/∂体重   = 1 − 体脂率/100
        #   ∂/∂体脂率 = −体重/100
        if te_w is None or te_pct is None:
            return None
        w, pct = m["weight_kg"], m["body_fat_pct"]
        return math.sqrt(
            (1 - pct / 100.0) ** 2 * te_w ** 2 + (w / 100.0) ** 2 * te_pct ** 2
        )

    # 实测指标：TE 就是它自己的
    return te_map.get(metric)
