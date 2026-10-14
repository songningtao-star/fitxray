"""
第 2 步：噪声地板 —— 「变化」到底多大才算数？

    python demos/step2_噪声地板.py

第 1 步说了体脂率的误差是方法性的。但那是"绝对值"的问题。
要判断"我这两个月有没有变化"，用的是**另一套**误差：日间典型误差（TE）。

这个脚本从一段只有噪声、没有真实变化的序列里把 TE 估出来，
顺便演示两个会让人算错的坑：

    坑一：用标准差估计 → 一次生病就把地板抬上去，工具变得不敢说话
    坑二：用全部的"点对"估计 → 序列一带趋势就估得离谱地大

不 import fitxray，全部手写，方便核对。
"""

import numpy as np

# 注入的真相：真实值恒定不变，读数上叠了标准差 0.5 kg 的日间波动
TRUE_TE = 0.5
N_DAYS = 120


def robust_sigma(diffs):
    """1.4826 × MAD(diffs) / √2

    为什么是 MAD 而不是标准差：标准差让**每一个点**都有话语权，
    一次 8 kg 的水分波动就能把它拽走一大截。MAD 只看中位数附近，
    少数几个异常点翻不起浪。

    为什么除以 √2：这些是"两个测量值相减"的差分，方差是两个独立
    测量之和，所以要把多出来的一倍除掉，还原成单次测量的标准差。
    """
    d = np.asarray(diffs, dtype=float)
    mad = np.median(np.abs(d - np.median(d)))
    return 1.4826 * mad / np.sqrt(2.0)


def pair_diffs(points):
    """所有点对的 (间隔天数, 读数差)。按**真实间隔**算，不按第几条。"""
    out = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            gap = points[j][0] - points[i][0]
            if gap > 0:
                out.append((gap, points[j][1] - points[i][1]))
    return out


def variogram(points, bins=(1, 2, 3, 5, 7, 10, 14, 21, 30)):
    """变差函数：间隔越远的两个点，差异有多大。分档估计。

    分档的边界取相邻两个代表值的**中点** —— 因为 _day 带着小时小数，
    "隔一天"实际算出来是 0.96~1.04 天，直接拿 1 当边界会把这批点
    劈到两个档里，每档只剩一半数据。
    """
    diffs = pair_diffs(points)
    edges = [0.0] + [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(bins) - 1)]
    edges.append(float("inf"))

    curve = []
    for k, gap in enumerate(bins):
        lo, hi = edges[k], edges[k + 1]
        got = [d for g, d in diffs if lo <= g < hi]
        if len(got) < 3:
            continue
        curve.append({"gap": gap, "n": len(got), "sigma": robust_sigma(got)})
    return curve


def main():
    rng = np.random.default_rng(7)
    true_value = 78.0
    readings = true_value + rng.normal(0.0, TRUE_TE, N_DAYS)
    points = [(float(d), float(readings[d])) for d in range(N_DAYS)]

    print("=" * 70)
    print(f"真值恒定 {true_value} kg，读数上叠了标准差 {TRUE_TE} kg 的日间波动")
    print(f"共 {N_DAYS} 天。真实变化 = 0。")
    print("=" * 70)
    print()

    # --- ① 稳健估计 vs 标准差 --------------------------------------------
    all_diffs = [d for _, d in pair_diffs(points[:30])]  # 只用前 30 天，算得快
    naive = float(np.std(all_diffs, ddof=1)) / np.sqrt(2.0)
    robust = robust_sigma(all_diffs)

    print("① 先看两种估法在一段干净数据上的表现")
    print(f"   标准差估法   TE = {naive:.3f} kg")
    print(f"   稳健估法     TE = {robust:.3f} kg")
    print(f"   注入的真值   TE = {TRUE_TE:.3f} kg")
    print()

    # --- ② 坑一：一次生病 ------------------------------------------------
    dirty = all_diffs + [8.0]  # 某天生病，水分波动 8 kg
    naive_d = float(np.std(dirty, ddof=1)) / np.sqrt(2.0)
    robust_d = robust_sigma(dirty)

    print("② 坑一：序列里混进一次 8 kg 的水分波动（生病 / 高钠 / 经期）")
    print(f"   标准差估法   TE = {naive_d:.3f} kg   ← 被那一根拽走了 "
          f"（+{(naive_d / naive - 1) * 100:.0f}%）")
    print(f"   稳健估法     TE = {robust_d:.3f} kg   ← 几乎没动")
    print()
    print("   地板被抬高会怎样？门槛跟着抬高，于是真的变化也报不出来。")
    print("   一个被一次感冒吓怕了的工具，比不报还糟糕。")
    print()

    # --- ③ 变差函数：噪声不是常数 ----------------------------------------
    print("③ 噪声不是常数 —— 间隔越远，两点之间能出现的差异越大")
    print()

    for label, series, marker in (
        ("纯随机（真值恒定）", readings, "应持平"),
        ("带缓慢漂移（真值在动）", None, "应上升"),
    ):
        if series is None:
            level = 0.0
            walk = []
            for _ in range(N_DAYS):
                level += rng.normal(0.0, 0.12)
                walk.append(true_value + level)
            series = np.array(walk)

        curve = variogram([(float(d), float(series[d])) for d in range(N_DAYS)])
        first, last = curve[0]["sigma"], curve[-1]["sigma"]
        ratio = last / first
        print(f"   {label}（{marker}）")
        for c in curve:
            bar = "█" * max(1, int(c["sigma"] * 40))
            print(f"     间隔 {c['gap']:>3} 天  n={c['n']:>4}  "
                  f"σ={c['sigma']:.3f}  {bar}")
        print(f"     → 首尾比 {ratio:.2f}"
              f"（{'平' if ratio < 1.25 else '在涨'}）")
        print()

    print("   这条曲线决定了判定时该拿哪个 σ：")
    print("   曲线是平的 → 噪声不随间隔累积，用固定门槛就行")
    print("   曲线在涨   → 存在生物漂移，隔得越远门槛要放得越宽")
    print()

    # --- ④ 坑二：别拿全部点对估 ------------------------------------------
    #
    # 这是写这个项目时真踩到的坑，所以专门放一个演示。
    # 门槛 = K × TE，而 TE 用全部点对估的话，长间隔里混的全是
    # 真实的身体变化 —— TE 被估得巨大，门槛高到天上，判据永远不触发。
    print("④ 坑二：估 TE 的时候，别把长间隔的点对也算进去")
    print()
    n_cut = 120
    trend = np.array([true_value - 0.10 * d + rng.normal(0.0, TRUE_TE)
                      for d in range(n_cut)])
    pts = [(float(d), float(trend[d])) for d in range(n_cut)]

    # 这里复现的是 fitxray 里真踩到的那个坑：`screen()` 一开始用
    # pair_diffs(pts, max_gap=30) 估 TE —— 也就是把所有 30 天以内的
    # 点对都算进去。序列一带趋势，这些点对里混的全是真实变化。
    all_pair = robust_sigma([d for _, d in pair_diffs(pts)])
    short_only = variogram(pts)[0]["sigma"]

    print(f"   序列：{n_cut} 天匀速减脂（每天 0.10 kg，累计 "
          f"{0.10 * n_cut:.0f} kg）—— 这是「真实的」身体变化")
    print()
    print(f"   全部点对一起估   TE = {all_pair:.3f} kg  ← 长间隔里混的是真变化")
    print(f"   只取最近一档估   TE = {short_only:.3f} kg  ← 相邻两天，趋势还没累积")
    print(f"   注入的噪声真值   TE = {TRUE_TE:.3f} kg")
    print()
    ratio = all_pair / short_only
    print(f"   差 {ratio:.1f} 倍。门槛是 K × TE，所以门槛也高 {ratio:.1f} 倍 ——")
    print(f"   按 4 倍 TE 算：干净口径的门槛是 {4 * short_only:.2f} kg，")
    print(f"   被污染的口径是 {4 * all_pair:.2f} kg。")
    print()
    print("   一次生病带来的 1.5 kg 水分波动，在前者眼里是异常，")
    print("   在后者眼里什么都算不上 —— 这条判据就永远不触发了。")
    print("   它在代码里、在测试里、在文档里都「存在」，只是从不出声。")
    print()

    # --- ⑤ 门槛 ----------------------------------------------------------
    te = robust
    lsc = 2.77 * te
    print("⑤ 结论：这道地板叫 LSC（最小显著变化）")
    print()
    print(f"   TE  = {te:.3f} kg（从这段数据自己估出来的，不是抄的常量）")
    print(f"   LSC = 1.96 × √2 × TE = 2.77 × TE = {lsc:.3f} kg")
    print()
    print(f"   小于 {lsc:.2f} kg 的变化，和噪声分不开 —— 报出来就是编的。")
    print()
    print("=" * 70)
    print("下一步：demos/step3_偏差为什么可以抵消.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
