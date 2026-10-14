"""
第 3 步：那个 ±2.3 kg 的方法偏差，为什么**不该**算进变化门槛里？

    python demos/step3_偏差为什么可以抵消.py

这是整个项目最容易搞错、也最反直觉的一步。

第 1 步说：BIA 测脂肪质量的偏差有 ±2.3 kg 量级。
直觉的反应是："既然偏差这么大，那变化也得大于 2.3 kg 才算数吧？"

**不对。** 稳定的偏差在"两个时间点相减"的时候会被减掉两次。
所以判断变化用的门槛，比 ±2.3 kg 小得多。

这个脚本用算得出来的数字把这件事说清楚，并且指出它的**适用边界** ——
偏差稳定才抵消，偏差自己会漂就不抵消。
"""

import numpy as np

# 一个人真实的脂肪质量：60 天里从 20.0 kg 稳稳减到 17.5 kg
# （取 2.5 kg 是为了让下面"对错两种门槛"的差别显出来：
#   正确门槛 2.2 kg 抓得住，错误门槛 6.5 kg 抓不住。）
TRUE_START = 20.0
TRUE_END = 17.5
TRUE_CHANGE = TRUE_END - TRUE_START


def fat_mass(weight, pct):
    return weight * pct / 100.0


def main():
    print("=" * 70)
    print(f"真相：脂肪质量从 {TRUE_START} kg 减到 {TRUE_END} kg"
          f"（真实变化 {TRUE_CHANGE:+.1f} kg）")
    print("=" * 70)
    print()

    # --- ① 偏差稳定时，它抵消了 ------------------------------------------
    #
    # 关键代数：设读数和真值之间差一个固定的 b。
    #
    #     读数(t1) = 真值(t1) + b
    #     读数(t2) = 真值(t2) + b
    #     ──────────────────────────────── 相减
    #     变化      = 真值(t2) − 真值(t1) + (b − b)
    #                                  ↑
    #                          这一项恒等于 0
    #
    # b 有多大**完全不重要** —— 2.3 kg 也好，23 kg 也好，
    # 只要是同一个 b，它就一定消失。
    print("① 偏差稳定时，它自己就消失了")
    print()
    print("   设读数和真值之间差一个固定的 b：")
    print("       读数(t1) = 真值(t1) + b")
    print("       读数(t2) = 真值(t2) + b")
    print("       ──────────────────────── 相减")
    print("       变化     = 真值差 + (b − b) = 真值差")
    print()
    print("   b 是 2.3 kg 还是 23 kg，对「变化」都没有影响。")
    print()

    # 数值验证：给所有读数加上一个巨大的固定偏差
    for b in (0.0, 2.3, -2.3, 23.0):
        start_read = TRUE_START + b
        end_read = TRUE_END + b
        print(f"   b = {b:>+6.1f} kg → 读数 {start_read:>6.2f} → {end_read:>6.2f}"
              f"   算出的变化 = {end_read - start_read:>+6.2f} kg")
    print()
    print(f"   真实变化 = {TRUE_CHANGE:+.2f} kg —— 四行全部对上。")
    print()

    # --- ② 所以门槛该用哪个误差 ------------------------------------------
    #
    # 这里要看清两个**不同**的误差在回答**不同**的问题。
    # 把它们混起来，是这一整类工具最常见的错。
    print("② 有两个误差，它们回答的不是同一个问题")
    print()

    te_weight = 0.5          # 体重的日间典型误差（kg）
    te_pct = 1.0             # 体脂率的日间典型误差（百分点）—— 注意不是 3.0
    method_pct = 3.0         # 体脂率的**方法**误差（百分点）—— BIA vs DXA
    w, pct = 78.2, 0.25

    # 日间误差按公式传到脂肪质量上（和 schema.derived_te 同一套算法）
    te_fat_day = np.sqrt((pct * te_weight) ** 2 + (w * te_pct / 100.0) ** 2)
    # 方法误差同样传到脂肪质量上
    method_fat = w * method_pct / 100.0

    print("   ┌ 方法误差   ±3.0 个百分点 → 脂肪质量 ±"
          f"{method_fat:.2f} kg")
    print("   │   回答：「你那个 25% 到底是不是 25%」")
    print("   │   这个项目永远不回答这个问题 —— 答案是不可知的")
    print("   │")
    print("   └ 日间误差   ±1.0 个百分点 → 脂肪质量 ±"
          f"{te_fat_day:.2f} kg")
    print("       回答：「今天和两个月前的差别，是不是噪声」")
    print("       判断变化用的是这个")
    print()

    # --- ③ 把它当成随机噪声，会错到什么程度 ------------------------------
    #
    # 最常见的错法：把方法误差当成"随机噪声"，于是以为它在相减时
    # 也要按方差相加 —— 门槛变成 1.96 × √2 × 2.35 = 6.5 kg。
    #
    # 但它不是随机噪声，它是**稳定偏差**，相减时自己就没了。
    sigma_change = np.sqrt(2.0) * te_fat_day
    thr = 1.96 * sigma_change
    wrong_thr = 1.96 * np.sqrt(2.0) * method_fat

    print("③ 把方法误差当成随机噪声，会错到什么程度")
    print()
    print("   这是最常见的错法：以为它在相减时也要按方差相加 ——")
    print(f"   1.96 × √2 × {method_fat:.2f} = {wrong_thr:.2f} kg")
    print()
    print(f"   正确做法（偏差抵消，只用日间误差）：")
    print(f"   σ(Δ) = √2 × {te_fat_day:.2f} = {sigma_change:.2f} kg")
    print(f"   门槛 = 1.96 × {sigma_change:.2f} = {thr:.2f} kg")
    print()
    print(f"   相差 {wrong_thr / thr:.1f} 倍 —— 门槛被抬高了整整 "
          f"{wrong_thr - thr:.1f} kg。")
    print()
    print(f"   真实变化 {abs(TRUE_CHANGE):.2f} kg 的去向：")
    print(f"     当成随机噪声（门槛 {wrong_thr:.2f} kg）→ "
          f"{'看得出来' if abs(TRUE_CHANGE) > wrong_thr else '看不出来 —— 结论是「你没变化」'}")
    print(f"     正确做法  （门槛 {thr:.2f} kg）→ "
          f"{'看得出来' if abs(TRUE_CHANGE) > thr else '看不出来'}")
    print()
    print("   错法会让工具漏报真实的变化 ——")
    print(f"   一个明明减掉了 {abs(TRUE_CHANGE):.1f} kg 脂肪的人，被告知「你没变」。")
    print()
    print("   这就是把「系统性」误差当成「随机」误差的代价：")
    print("   随机误差多测几次会小，系统性误差不会 ——")
    print("   按随机的算法去处理它，等于凭空给自己加了一倍的方差。")
    print()

    # --- ④ 适用边界 ------------------------------------------------------
    #
    # 上面那套推导有一个前提：b 是**稳定**的。
    # 如果偏差自己会随时间漂，那 b(t2) − b(t1) ≠ 0，它就跑到变化里去了。
    # 这不是理论上的担忧 —— 这正是"饮食结构变了""开始吃肌酸"
    # "换了补水习惯"这些事会让读数整体偏移的机制。
    print("④ 适用边界：这套推导有个前提，必须说清楚")
    print()
    print("   前提是 b 稳定。如果偏差自己会漂，b(t2) − b(t1) ≠ 0，")
    print("   它就漏进「变化」里了，而且你无法把这个漂移和数据分开。")
    print("   两个点，一个未知的真实变化加一个未知的偏差漂移 ——")
    print("   一个方程两个未知数，不可辨识。")
    print()
    rng = np.random.default_rng(3)
    for drift_label, drift in (("偏差稳定", 0.0), ("偏差缓慢漂移", -0.02)):
        readings = []
        true_vals = []
        for d in range(60):
            t = TRUE_START + (TRUE_END - TRUE_START) * d / 59.0
            b = 2.3 + drift * d          # b 随时间漂
            noise = rng.normal(0.0, te_fat_day) if drift else 0.0
            true_vals.append(t)
            readings.append(t + b + noise)

        measured = readings[-1] - readings[0]
        actual = true_vals[-1] - true_vals[0]
        err = measured - actual
        verdict = "可以" if abs(err) < thr else "不可以"
        print(f"   {drift_label:<10} 测出变化 {measured:+.2f} kg，"
              f"真实变化 {actual:+.2f} kg，误差 {err:+.2f} kg")
        print(f"              误差 {abs(err):.2f} kg "
              f"{'<' if abs(err) < thr else '>'} 门槛 {thr:.2f} kg"
              f" → {verdict}当成真实变化")
    print()
    print("   所以：偏差稳定 → 偏移是「同一个数」，抵消掉，不影响变化")
    print("         偏差漂移 → 偏移是「两个不同的数」，差值就是假变化")
    print()
    print("   fitxray 能处理第一种。第二种只能靠「测量条件标准化」来防 ——")
    print("   固定时间、空腹、别在练后测。这也是 comparability.py 存在的理由。")
    print()
    print("=" * 70)
    print("下一步：demos/step4_最后一步判定.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
