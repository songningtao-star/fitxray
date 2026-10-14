"""
第 4 步：把所有东西合起来，然后问一个更难的问题。

    python demos/step4_最后一步判定.py

前三步分别解决了：
    ① 误差分两种 —— 随机的会收敛，方法性的不会
    ② 从你自己的数据里估出噪声地板
    ③ 稳定的偏差在相减时抵消，所以门槛比想象的小

现在合起来做判定。但真正让这个项目站得住的是**后半段**：
喂它一段真值完全不变的序列，它会不会误报？

一个只会说「变了」的工具没有价值，一个只会说「没变」的也没有。
所以最后要同时看两边：该抓的抓到没有，不该报的报了没有。
"""

import math

import numpy as np

# 看得越多，越容易撞上假阳性 —— 这就是多重比较问题。
# α=0.05 下，检验 20 个指标，平均就会撞出 1 个假阳性。
N_METRICS = 6
ALPHA = 0.05


# ---------------------------------------------------------------- 小工具
def robust_sigma(diffs):
    d = np.asarray(diffs, dtype=float)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)


def two_sided_p(z):
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def bh(pvals):
    """Benjamini-Hochberg：把 p 值从小到大排，第 k 小的乘 n/k，
    再从大到小取累计最小值。得到的 q 值回答的是
    「在假发现率 5% 的前提下，这个发现还站得住吗」。"""
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


def judge(true_change, te):
    """拿真值当输入，走一遍判定。返回 (verdict, p, change)。

    这里刻意直接喂真值，不加噪声 —— 想单独看清楚"门槛"这一件事。
    噪声的影响放在后半段的假阳性模拟里单独考察。
    """
    sigma = math.sqrt(2.0) * te
    p = two_sided_p(true_change / sigma) if sigma else 1.0
    if abs(true_change) > 1.96 * sigma:
        return "真实变化", p, true_change
    if abs(true_change) > 1.28 * sigma:
        return "临界", p, true_change
    return "在噪声内", p, true_change


def main():
    print("=" * 70)
    print("前半段：一段真实减脂的序列")
    print("=" * 70)
    print()

    te = 0.6  # 脂肪质量的日间典型误差（kg）
    for label, true_change in (("狠狠减脂", -6.0), ("小幅减脂", -1.2), ("什么都没变", 0.0)):
        v, p, ch = judge(true_change, te)
        print(f"   {label:<8} 真实变化 {ch:>+6.2f} kg  "
              f"（门槛 {1.96 * math.sqrt(2.0) * te:.2f} kg）  →  {v}")
    print()
    print("   门槛之内的变化，工具会明确说「和噪声分不开」，而不是编一个数字。")
    print()

    # --- 后半段：假阳性 --------------------------------------------------
    #
    # 这是真刀真枪的模拟，不是编一个分布出来充数。
    #
    # 每一段序列都**真值恒定不变**：估出来的"变化"完全来自测量噪声，
    # 所以它服从 N(0, σ(Δ))。标准化之后就是标准正态，
    # 于是 p 值天然是均匀分布 —— 这正是零假设下的样子。
    # 每段序列都跑一遍完整的判定，数一数有多少段报了"真实变化"。
    print("=" * 70)
    print("后半段：喂它一堆「真值完全不变」的序列，看它会不会误报")
    print("=" * 70)
    print()

    N_TRIALS = 5000
    rng = np.random.default_rng(11)
    sigma_change = math.sqrt(2.0) * te

    naive = 0
    corrected = 0
    for _ in range(N_TRIALS):
        # 真值变化是 0。估出来的变化全是噪声。
        changes = rng.normal(0.0, sigma_change, N_METRICS)
        pvals = [two_sided_p(c / sigma_change) for c in changes]

        if any(p < ALPHA for p in pvals):
            naive += 1
        if min(bh(pvals)) < ALPHA:
            corrected += 1

    print(f"   跑了 {N_TRIALS} 段真值不变的序列，每段看 {N_METRICS} 个指标")
    print()
    print(f"   不做校正，只要有一个 p < {ALPHA} 就报：")
    print(f"     误报的：{naive} / {N_TRIALS} （{naive / N_TRIALS * 100:.1f}%）")
    print()
    print(f"   做了 BH 校正，看 q < {ALPHA}：")
    print(f"     误报的：{corrected} / {N_TRIALS} "
          f"（{corrected / N_TRIALS * 100:.1f}%）")
    print()
    print(f"   → 从 {naive / N_TRIALS * 100:.0f}% 降到 "
          f"{corrected / N_TRIALS * 100:.1f}%，降到原来的 "
          f"{corrected / max(naive, 1) * 100:.0f}%。")
    print()
    print("   理论上第一行就该是 1 − 0.95^6 = 26.5%，第二行该是 5% 上下。")
    print("   模拟对得上 —— 这就说明校正不是装饰，是真的在控假发现率。")
    print()
    print("   （严谨一点：BH 的保证需要各检验独立或满足 PRDS。真实数据里")
    print("     体重和脂肪质量共享输入，不是完全独立 —— 所以这里是近似，")
    print("     不是严格保证。写下来，免得把话说满。）")
    print()

    # --- 那个自指的讽刺 --------------------------------------------------
    print("=" * 70)
    print("一个不太好意思的地方")
    print("=" * 70)
    print()
    print("   这个项目的立论是「别再给我一堆数字了」。")
    print("   可是判定模块自己每跑一次，就要做好几个假设检验 ——")
    print("   于是它自己又引入了一个多重比较问题。")
    print()
    print(f"   同时看 {N_METRICS} 个指标，每个都用 α={ALPHA}，")
    print(f"   至少撞上一个假阳性的概率是 "
          f"{1 - (1 - ALPHA) ** N_METRICS:.0%}。")
    print()
    print("   这不能装作没看见。显式校正、并把 q 值写进报告，")
    print("   本身就是论点的一部分 ——")
    print("   一个连自己的统计偏差都不检查的工具，")
    print("   没资格教别人怎么看待数字。")
    print()
    print("=" * 70)
    print("四步走完。真实的流程见：python run.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
