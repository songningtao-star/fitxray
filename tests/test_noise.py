"""
noise 的测试：稳健估计、分箱、变差函数、兜底。

这里有两个测试是"防回归"的 —— 它们锁住的是踩过的坑，
不是显然的性质。看到可疑的失败，先去看那条测试的注释。
"""

import math

import numpy as np

from fitxray.noise import (
    DEFAULT_BINS,
    DEFAULT_TE,
    LSC_FACTOR,
    MIN_POINTS,
    _bin_edges,
    is_random_walk,
    lsc,
    noise_floor,
    pair_diffs,
    robust_sigma,
    series_points,
    sigma_at,
    variogram,
)


# ---------------------------------------------------------------- 稳健估计
def test_robust_sigma_recovers_the_true_sd():
    """在一大堆正态差分上，稳健估计应该贴近真实标准差。"""
    rng = np.random.default_rng(42)
    true_sd = 0.7
    # 差分是两个独立测量相减，方差加倍
    diffs = rng.normal(0.0, true_sd * math.sqrt(2.0), 20000)
    est = robust_sigma(diffs)
    assert abs(est - true_sd) < 0.03


def test_robust_sigma_ignores_a_single_wild_outlier():
    """这是用 MAD 而不用标准差的全部理由。

    一次生病引起的水分波动会在标准差里留下很大痕迹，
    把整个噪声地板抬上去 —— 结果工具变得过于保守，真的变化也报不出来。
    """
    rng = np.random.default_rng(7)
    clean = list(rng.normal(0.0, 0.7 * math.sqrt(2.0), 500))
    with_outlier = clean + [8.0]  # 一次 8 kg 的水分波动

    robust = robust_sigma(with_outlier)
    naive = float(np.std(with_outlier, ddof=1))

    assert abs(robust - 0.7) < 0.05        # 稳健估计几乎没动
    assert naive > 0.85                     # 标准差被那一根拽走了


def test_robust_sigma_needs_two_points():
    assert robust_sigma([]) is None
    assert robust_sigma([0.5]) is None


def test_lsc_is_277_times_te():
    """LSC = 1.96 × √2 × TE ≈ 2.77 × TE，那个 √2 来自"两个测量值相减"。"""
    assert abs(lsc(1.0) - 2.77) < 1e-9
    # 2.77 是 1.96 × √2 的四舍五入，差在第 4 位小数上
    assert abs(LSC_FACTOR - 1.96 * math.sqrt(2.0)) < 0.002


# ---------------------------------------------------------------- 点对
def test_pair_diffs_uses_all_pairs_not_just_neighbours():
    """采样不规则，所以按"真实的间隔天数"分箱，而不是第 k 个和第 k+1 个。"""
    pts = [(0.0, 1.0), (1.0, 2.0), (3.0, 4.0)]
    diffs = sorted(pair_diffs(pts))
    assert diffs == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_pair_diffs_respects_max_gap():
    pts = [(0.0, 0.0), (1.0, 1.0), (50.0, 9.0)]
    assert len(pair_diffs(pts, max_gap=5.0)) == 1


def test_pair_diffs_skips_zero_and_negative_gaps():
    assert pair_diffs([(2.0, 1.0), (2.0, 5.0)]) == []


# ---------------------------------------------------------------- 分箱
def test_bin_edges_are_midpoints_between_representatives():
    assert _bin_edges((1, 2, 3)) == [0.0, 1.5, 2.5, 3.0]


def test_daily_gaps_all_land_in_the_first_bin():
    """★ 防回归测试。

    `_day` 带着小时的小数部分，所以"隔一天"的两个点实际算出来是
    0.96~1.04 天。最早 DEFAULT_BINS 被当成箱边界用，这批点会被
    劈到两个箱里，每一箱只剩一半数据 —— 一档 72 对变成了 41 对。

    改成"代表值取中点当边界"之后，全部落回第一档。
    """
    pts = [(float(d), 70.0 + d) for d in range(11)]  # 0..10 天，相邻间隔恰好 1.0
    curve = variogram(pts)

    assert curve[0]["gap"] == 1
    assert curve[0]["n"] == 10  # 10 个相邻点对，一对不少
    assert curve[0]["gap_lo"] == 0.0
    assert curve[0]["gap_hi"] == 1.5


def test_fractional_day_gaps_still_land_in_the_first_bin():
    """带小时小数的"隔一天"必须和整数间隔落在同一档。"""
    pts = [(0.0, 70.0), (0.96, 70.4), (2.04, 69.9), (3.0, 70.6)]
    curve = variogram(pts)
    assert curve[0]["gap"] == 1
    # 0→0.96、0.96→2.04、2.04→3.0 三对都该在第一档
    assert curve[0]["n"] == 3


def test_bins_below_three_pairs_are_dropped():
    """一对两对的档位估不出东西，宁可不报。"""
    pts = [(0.0, 1.0), (1.0, 2.0)]
    assert variogram(pts) == []


# ---------------------------------------------------------------- 变差函数
def test_flat_noise_gives_a_flat_curve():
    """纯随机噪声 → 曲线应该是平的（间隔多远都一样）。"""
    rng = np.random.default_rng(3)
    pts = [(float(d), float(rng.normal(0, 0.5))) for d in range(120)]
    curve = variogram(pts)
    assert not is_random_walk(curve)
    assert len(curve) >= 4


def test_accumulating_drift_gives_a_rising_curve():
    """随机游走 → 间隔越远，两点之间能出现的差异越大。"""
    rng = np.random.default_rng(3)
    level = 0.0
    pts = []
    for d in range(400):
        level += rng.normal(0, 0.25)
        pts.append((float(d), level))
    curve = variogram(pts)
    assert is_random_walk(curve)
    assert curve[-1]["sigma"] > curve[0]["sigma"] * 1.5


def test_sigma_at_clamps_and_interpolates():
    curve = [
        {"gap": 1, "gap_lo": 0.0, "gap_hi": 1.5, "sigma": 0.4, "n": 50},
        {"gap": 5, "gap_lo": 1.5, "gap_hi": 6.0, "sigma": 0.8, "n": 50},
    ]
    assert sigma_at(curve, 1.0) == 0.4          # 第一档之内 → 就近取
    assert sigma_at(curve, 99.0) == 0.8         # 超出末尾 → 取最后一档
    mid = sigma_at(curve, 3.75)                 # 正好在中点 → 线性插值
    assert abs(mid - 0.6) < 1e-9
    assert sigma_at([], 1.0) is None


# ---------------------------------------------------------------- 主入口
def test_noise_floor_estimates_from_enough_points():
    rng = np.random.default_rng(11)
    pts = [(float(d), float(rng.normal(0, 0.6))) for d in range(60)]
    nf = noise_floor(pts, "weight_kg")

    assert nf["source"] == "estimated"
    assert abs(nf["te"] - 0.6) < 0.1
    assert abs(nf["lsc"] - 2.77 * nf["te"]) < 1e-9
    assert nf["n_points"] == 60
    assert nf["curve"]


def test_noise_floor_falls_back_and_says_so():
    """少于 8 个点时用默认值，而且必须在 note 里写明白 ——
    不能让用户以为这个数字是从他自己的数据里算出来的。"""
    pts = [(float(d), 70.0 + d) for d in range(MIN_POINTS - 1)]
    nf = noise_floor(pts, "weight_kg")

    assert nf["source"] == "default"
    assert nf["te"] == DEFAULT_TE["weight_kg"]
    assert "默认" in nf["note"]
    assert str(MIN_POINTS - 1) in nf["note"]


def test_noise_floor_handles_an_unknown_metric_with_no_fallback():
    nf = noise_floor([(0.0, 1.0)], "火星体重")
    assert nf["source"] == "default"
    assert nf["te"] is None
    assert nf["lsc"] is None


def test_noise_floor_does_not_choke_on_a_constant_series():
    """全是同一个值 → 差分全 0，估不出噪声，必须兜底而不是崩。"""
    pts = [(float(d), 70.0) for d in range(30)]
    nf = noise_floor(pts, "weight_kg")
    assert nf["source"] == "default"
    assert nf["te"] is not None


def test_series_points_sorts_and_skips_missing():
    recs = [
        {"_day": 3.0, "weight_kg": 77.0},
        {"_day": 1.0, "weight_kg": 78.0},
        {"_day": 2.0, "weight_kg": None},
    ]
    assert series_points(recs, "weight_kg") == [(1.0, 78.0), (3.0, 77.0)]


def test_default_bins_are_representative_values_not_edges():
    """锁住语义：DEFAULT_BINS 是代表值，箱边界由 _bin_edges 算。
    哪天有人"顺手"把它当成边界用，这个测试会挡住。"""
    assert DEFAULT_BINS[0] == 1
    assert _bin_edges(DEFAULT_BINS)[1] == 1.5
