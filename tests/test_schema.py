"""
schema 的测试：派生量、误差传播、两类不确定度的分工。

核心主张必须能被人工验算，所以这里的第一个测试就是
README 里那个「78.2kg/25% → 77.8kg/24%」的计算 ——
拿计算器按出来的数，代码必须给出同一个。
"""

import math

from fitxray.schema import (
    METRICS,
    derived_te,
    fat_mass,
    fat_mass_uncertainty,
    lean_mass,
    lean_mass_uncertainty,
    propagate_difference,
    propagate_product,
    value_and_uncertainty,
)

# 一条有代表性的记录：78.2 kg / 25.0%
A = {"weight_kg": 78.2, "body_fat_pct": 25.0}
B = {"weight_kg": 77.8, "body_fat_pct": 24.0}

# 一台足部秤的完整读数，用来覆盖全部已注册指标
FULL = {
    "weight_kg": 78.2,
    "body_fat_pct": 25.0,
    "skeletal_muscle_kg": 33.1,
    "body_water_pct": 55.2,
}


def test_fat_mass_is_the_obvious_product():
    assert fat_mass(A) == 78.2 * 25.0 / 100.0


def test_lean_mass_is_an_identity_not_an_estimate():
    """瘦体重 = 体重 − 脂肪质量。这是定义，不是回归，所以必须精确成立。"""
    m = dict(A)
    assert lean_mass(m) + fat_mass(m) == m["weight_kg"]


def test_method_uncertainty_matches_a_hand_calculation():
    """手算一遍：σ(脂肪质量) ≈ 体重 × σ(体脂率)。

    体重那一项的贡献是 0.13%，体脂率那一项是 12%，后者完全压倒前者。
    所以传播出来的结果应该非常接近 78.2 × 0.03 = 2.346。
    """
    half = fat_mass_uncertainty(A)
    assert abs(half - 2.346) < 0.02

    # 严格按相对不确定度平方和开方再算一遍，作为交叉验证
    _, exact = propagate_product([(78.2, 0.1), (0.25, 0.03)])
    assert abs(half - exact) < 1e-9


def test_weight_error_barely_moves_lean_mass():
    """瘦体重的不确定度几乎就等于脂肪质量的不确定度 —— 体重那 0.1 kg 可以忽略。"""
    lean_half = lean_mass_uncertainty(A)
    fat_half = fat_mass_uncertainty(A)
    assert abs(lean_half - fat_half) < 0.01


def test_propagate_difference_adds_variances():
    result, half = propagate_difference([(10.0, 0.3), (-4.0, 0.4)])
    assert result == 6.0
    assert abs(half - math.sqrt(0.09 + 0.16)) < 1e-12


def test_the_readme_case_is_inside_the_noise():
    """「78.2/25% → 77.8/24%」这个例子。

    按百分比一算，减掉了 0.878 kg 脂肪，看着是成果。
    但方法不确定度是 2.3 kg 量级 —— 这个变化完全读不出来。
    这是项目的核心例子，必须能被拿计算器核对。
    """
    change = fat_mass(B) - fat_mass(A)
    half = fat_mass_uncertainty(B)

    assert abs(change - (-0.878)) < 0.001
    assert half > 2.3
    assert abs(change) < half  # 变化量小于不确定度 → 读不出来


def test_derived_te_uses_day_to_day_error_not_method_error():
    """判定"变化"用的是日间 TE，不是文献里的方法误差。

    这是最容易搞错的一步：稳定的方法偏差在
    `脂肪(t2) − 脂肪(t1)` 里会被减掉两次，所以不能叠加进去。
    """
    te_map = {"weight_kg": 0.5, "body_fat_pct": 1.0}
    te = derived_te(A, "fat_mass_kg", te_map)

    expected = math.sqrt((25.0 / 100) ** 2 * 0.5 ** 2 + (78.2 / 100) ** 2 * 1.0 ** 2)
    assert abs(te - expected) < 1e-12

    # 体脂率那一项完全主导：丢掉体重项，结果几乎不变
    only_pct = (78.2 / 100) * 1.0
    assert abs(te - only_pct) < 0.02


def test_derived_te_returns_none_when_inputs_missing():
    assert derived_te(A, "fat_mass_kg", {}) is None
    assert derived_te(A, "lean_mass_kg", {"weight_kg": 0.5}) is None


def test_measured_metrics_pass_te_through():
    te_map = {"weight_kg": 0.5, "body_fat_pct": 1.0}
    assert derived_te(A, "weight_kg", te_map) == 0.5


def test_value_and_uncertainty_covers_every_registered_metric():
    for metric in METRICS:
        value, unc = value_and_uncertainty(FULL, metric)
        assert value is not None, metric
        assert unc is not None and unc > 0, metric


def test_value_and_uncertainty_returns_none_for_unknown():
    assert value_and_uncertainty(A, "body_age") == (None, None)
    assert value_and_uncertainty(A, "不存在的指标") == (None, None)


def test_missing_field_gives_none_not_zero():
    """缺字段必须是 None。当成 0 会算出一个看起来合理的假值，
    比直接报错难查得多。"""
    assert value_and_uncertainty({"weight_kg": 78.2}, "body_fat_pct") == (None, None)
