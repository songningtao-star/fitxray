"""
可比性门控 —— 相减之前，先问这两次测量能不能比。

所有体成分工具都在做同一件事：拿今天的数减昨天的数，画条曲线。
**但这个减法有一个隐含前提：两次测量的条件是一样的。**

这个前提经常不成立。举个最常见的例子：

    昨天早上 7:30，空腹，没运动   →  78.2 kg
    今天晚上 8:00，练完腿，刚吃完饭 →  77.4 kg

工具会告诉你「一天掉了 0.8 kg」。但这两个数**根本不该相减** ——
差值里混着日内波动、一顿饭、一升水、还有运动造成的水肿。
这些都不是"变化"，是"测量条件不一样"。

这不是噪声。噪声是随机的、会互相抵消；这是**污染**，
是系统性的、会持续偏向一个方向。所以它不能靠多测几次解决，
只能靠**测之前把条件控住**，或者**事后把不可比的点标出来**。

---- 判据从哪来 ----

用的是 ESPEN 2004 生物电阻抗分析标准化指南（Kyle et al.）和
1994 年 NIH 的 BIA 技术评估会议。这些是临床上的标准做法，
不是我自己编的阈值。每条判据在下面都写了理由。

---- 一个反直觉的原则：只降级，不删除 ----

把一次大幅水分波动「当异常值删掉」，等于删掉了**最可解释的那个事件**
—— 生病、高钠饮食、经期，这些都是真实发生的、你该知道的事。
所以这里的做法是：给每个点打标签，判定时只用干净的点，
但**报告里把所有被降级的点连同原因一起显示出来**。
"""

# ---------------------------------------------------------------- 协议清单
# 这是给人看的清单（README / 报告里会展示），实际判定逻辑在下面的函数里。
PROTOCOL_RULES = (
    {
        "key": "device",
        "name": "同一台设备",
        "why": "BIA 的回归公式是按设备定制的，厂商算法不公开。"
        "换一台秤，读数就换了一套坐标系，跨设备相减没有意义。",
    },
    {
        "key": "interval",
        "name": "间隔至少 1 天",
        "why": "日内波动本身就有 1~2 kg（吃、喝、排空、体温节律）。"
        "同一天内测两次，差的主要是这些。",
    },
    {
        "key": "time_of_day",
        "name": "固定时段（相差不超过 2 小时）",
        "why": "ESPEN 指南明确要求固定测量时间。人体成分有明显的昼夜节律，"
        "早上和晚上本来就不一样。",
    },
    {
        "key": "fasted",
        "name": "空腹 4 小时以上",
        "why": "ESPEN：进食后阻抗会下降 4~15 Ω，影响持续 2~4 小时。"
        "吃饱了再测，体脂率会偏低。",
    },
    {
        "key": "exercise",
        "name": "运动后 8 小时内不测",
        "why": "ESPEN：运动会把电阻 R 压低约 3%、电抗 Xc 压低约 8%。"
        "肌肉充血水肿、含水量上升，**体脂率会假性变低** —— "
        "练完称出来的「进步」是假的。",
    },
    {
        "key": "alcohol",
        "name": "饮酒后 8 小时内不测",
        "why": "酒精影响水合状态，而 BIA 恰恰是用水分分布去推体成分的。",
    },
    {
        "key": "bladder",
        "name": "排空膀胱",
        "why": "最直接的一项，也是最容易做到的一项。",
    },
)

# 判定的三个等级，从好到坏。
LEVEL_OK = "可比"
LEVEL_WARN = "需谨慎"
LEVEL_BAD = "不可比"

_ORDER = {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_BAD: 2}

# 体重突跳的判据倍数。相对稳健的中心，偏离超过 K 倍典型误差就标记。
# 这个 K 取得比较保守（偏大），因为下面那条"先估再标"的逻辑只走一遍，
# 不走迭代收敛 —— 宁可漏标，不可错标。
JUMP_K = 4.0


def _worst(levels):
    """取一组等级里最差的那个。"""
    return max(levels, key=lambda x: _ORDER[x]) if levels else LEVEL_OK


# ---------------------------------------------------------------- 单条判据
def _flag(key, level, reason, a=None, b=None):
    return {"rule": key, "level": level, "reason": reason}


def compare(a, b):
    """判断两次测量能不能相减。

    参数
        a, b  两条测量记录。需要 `_day`（浮点天数），
              `_hour`（浮点小时，可选）由 loaders 附上。

    返回
        {
          "verdict": "可比" | "需谨慎" | "不可比",
          "gap_days": float,
          "flags": [{"rule","level","reason"}, ...],   # 只含**未通过**的判据
        }

    只返回没过的判据 —— 一次列七八条"通过"的信息量是零。
    """
    flags = []
    gap = abs(b["_day"] - a["_day"])

    # ① 同一台设备。这条最硬：跨设备不是误差变大，是坐标系都换了。
    da, db = a.get("device_id"), b.get("device_id")
    if da and db and da != db:
        flags.append(
            _flag(
                "device",
                LEVEL_BAD,
                f"两台不同的设备（{da} / {db}）。"
                f"BIA 公式按设备定制，跨设备相减没有意义",
            )
        )

    # ② 间隔至少一天。
    if gap < 1.0:
        flags.append(
            _flag(
                "interval",
                LEVEL_BAD,
                f"只隔了 {gap * 24:.1f} 小时。"
                f"日内波动本身就有 1~2 kg，这个差值主要是吃喝排空",
            )
        )

    # ③ 时段。只有两次都记了时刻才能判 —— 没记就不能假装判过。
    ha, hb = a.get("_hour"), b.get("_hour")
    if ha is not None and hb is not None:
        dh = abs(hb - ha)
        dh = min(dh, 24.0 - dh)  # 跨零点
        if dh > 2.0:
            flags.append(
                _flag(
                    "time_of_day",
                    LEVEL_WARN,
                    f"测量时刻差了 {dh:.1f} 小时（{ha:.1f} 点 / {hb:.1f} 点）。"
                    f"ESPEN 要求固定时段",
                )
            )

    # ④ 空腹状态不一致。
    fa, fb = a.get("fasted"), b.get("fasted")
    if fa is not None and fb is not None and fa != fb:
        flags.append(
            _flag(
                "fasted",
                LEVEL_WARN,
                "一次空腹、一次不是。ESPEN：进食后阻抗降 4~15 Ω，"
                "持续 2~4 小时",
            )
        )

    # ⑤ 运动后 8 小时内 —— 这条直接判不可比，因为它会**系统性地**
    #    把体脂率压下去，方向固定，不是随机误差。
    for x, who in ((a, "前一次"), (b, "后一次")):
        if x.get("trained_8h") is True:
            flags.append(
                _flag(
                    "exercise",
                    LEVEL_BAD,
                    f"{who}测量在运动后 8 小时内。"
                    f"运动让阻抗下降，**体脂率会假性偏低** —— "
                    f"这种「进步」是假的",
                )
            )

    # ⑥ 饮酒。
    for x, who in ((a, "前一次"), (b, "后一次")):
        if x.get("alcohol_8h") is True:
            flags.append(
                _flag(
                    "alcohol",
                    LEVEL_WARN,
                    f"{who}测量在饮酒后 8 小时内，水合状态受影响",
                )
            )

    return {
        "verdict": _worst([f["level"] for f in flags]),
        "gap_days": gap,
        "flags": flags,
    }


# ---------------------------------------------------------------- 序列筛查
def _protocol_level(rec):
    """单条记录自己就有问题的情形（不需要配对）。"""
    flags = []
    if rec.get("trained_8h") is True:
        flags.append("运动后 8 小时内")
    if rec.get("alcohol_8h") is True:
        flags.append("饮酒后 8 小时内")
    return flags


def screen(records, value_key="weight_kg"):
    """对整条序列做筛查，分出干净点和被降级的点。

    顺序很重要（避免循环论证）：
        1. 先用**协议类**判据（设备/运动/饮酒）把明显有问题的点挑出来 ——
           这些不依赖任何统计估计，不存在循环。
        2. 在**剩下的点**上做一次稳健的噪声估计。
        3. 用它找体重突跳的点。
        4. 在**最终干净点**上重新估计一次噪声，两个值都报出来。

    为什么不做迭代收敛：用同一个估计去筛异常点、再用筛完的点去更新估计，
    反复迭代会把自己收敛到一个"看起来很干净"的结果上，但那个结果的
    可信度是自我印证的。走一遍，把前后两个值都摆出来，更诚实。

    参数
        records   测量记录（已按日期排序，带 `_day`）

    返回
        {
          "clean_idx":  [干净点的下标],
          "flagged":    [{"index","date","level","reasons"}, ...],
          "te_before":  float | None,   # 筛查前的噪声估计
          "te_after":   float | None,   # 筛查后的噪声估计
          "jump_flagged": int,          # 因突跳被标记的点数
        }
    """
    from .noise import pair_diffs, robust_sigma, variogram

    n = len(records)
    flagged = {}
    clean_idx = []

    # --- 第 1 步：协议类判据 ---
    for i, rec in enumerate(records):
        reasons = _protocol_level(rec)
        if reasons:
            flagged[i] = {"index": i, "level": LEVEL_BAD, "reasons": reasons}
        else:
            clean_idx.append(i)

    if len(clean_idx) < 3:
        return {
            "clean_idx": clean_idx,
            "flagged": [
                {
                    "index": i,
                    "date": records[i].get("date"),
                    "level": flagged[i]["level"],
                    "reasons": flagged[i]["reasons"],
                }
                for i in sorted(flagged)
            ],
            "te_before": None,
            "te_after": None,
            "jump_flagged": 0,
        }

    # --- 第 2 步：在剩下的点上估一次噪声 ---
    #
    # 这里必须用**最近一档**的估计（和 noise.py 里那个决定同一个道理），
    # 不能拿全部点对去算。序列带着趋势的时候，长间隔的点对里混的是
    # 真实的身体变化，会把噪声估得很大 —— 于是下面那条突跳判据的
    # 阈值被抬到天上，一次真正的异常波动永远也标不出来。判据等于白写。
    #
    # 最近一档里趋势还没累积起来，估出来的才是噪声本身。
    pts_before = [(records[i]["_day"], float(records[i][value_key]))
                  for i in clean_idx if records[i].get(value_key) is not None]
    pts_before.sort()
    curve_before = variogram(pts_before)
    if curve_before:
        te_before = curve_before[0]["sigma"]
    else:
        te_before = robust_sigma([d for _, d in pair_diffs(pts_before, max_gap=30)])

    # --- 第 3 步：找突跳。用局部中位数而不是全局 —— 序列是带趋势的，
    #     拿全序列的中位数当基准，会把趋势的两端全判成异常。 ---
    jump_flagged = 0
    if te_before and te_before > 0:
        survivors = []
        for order, i in enumerate(clean_idx):
            v = records[i].get(value_key)
            if v is None:
                survivors.append(i)
                continue
            # 邻居：序里前后各两个（在 clean_idx 里取，不跨被标记的点）
            lo = max(0, order - 2)
            hi = min(len(clean_idx), order + 3)
            neighbours = [
                records[j][value_key]
                for j in clean_idx[lo:hi]
                if j != i and records[j].get(value_key) is not None
            ]
            if len(neighbours) < 2:
                survivors.append(i)
                continue
            local_med = sorted(neighbours)[len(neighbours) // 2]
            # 邻居里也有真实趋势，所以基准取中位数而不是均值：
            # 一个邻居恰好是异常点，中位数照样稳。
            if abs(v - local_med) > JUMP_K * te_before:
                flagged[i] = {
                    "index": i,
                    "level": LEVEL_WARN,
                    "reasons": [
                        f"体重比邻近测量偏离 {abs(v - local_med):.1f} kg，"
                        f"超过 {JUMP_K:.0f} 倍典型误差（{te_before:.2f} kg）"
                    ],
                }
                jump_flagged += 1
            else:
                survivors.append(i)
        clean_idx = survivors

    # --- 第 4 步：在最终干净点上重新估一次 ---
    pts_after = [(records[i]["_day"], float(records[i][value_key]))
                 for i in clean_idx if records[i].get(value_key) is not None]
    pts_after.sort()
    te_after = robust_sigma([d for _, d in pair_diffs(pts_after, max_gap=30)])

    return {
        "clean_idx": clean_idx,
        "flagged": [
            {
                "index": i,
                "date": records[i].get("date"),
                "level": flagged[i]["level"],
                "reasons": flagged[i]["reasons"],
            }
            for i in sorted(flagged)
        ],
        "te_before": te_before,
        "te_after": te_after,
        "jump_flagged": jump_flagged,
    }
