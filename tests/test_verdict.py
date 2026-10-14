"""
verdict 的测试：判定本身，以及它敢不敢承认自己不行。

两个测试是这个文件存在的理由：

`test_weight_falls_but_fat_mass_stays_inside_the_noise` 是整个项目的
核心场景 —— 体重实实在在掉了 2 kg，但脂肪质量**一动没动**。
工具必须同时说出这两句话，而且理由要能核对。市面上所有体脂秤
在这里都会兴奋地宣布"你减了 2 kg"，而那 2 kg 全是水和瘦体重。

`test_a_pure_noise_series_produces_no_false_positives` 是反面。
一个只会说「变了」的工具没有价值 —— 喂它一段真值完全不变的序列，
它必须一声不吭。
"""

import math

import pytest

from fitxray.synth import generate
from fitxray.verdict import (
    EDGE_N,
    VERDICT_BORDER,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    Z_95,
    _bh,
    _median,
    _two_sided_p,
    verdict_for,
    verdicts,
)

# 一段「真值完全不动」的序列 —— 做对照组用。
# fat_rate 和 lean_gain_rate 都给 0，真实体重和脂肪质量都是常数。
FLAT = ({"days": 120, "fat_rate": 0.0, "note": "不动"},)


def _flat_series(seed=0, **kw):
    """真值完全不变的序列。测量噪声照常存在，只是没有真实变化可找。"""
    opts = dict(
        segments=FLAT,
        seed=seed,
        measure_prob=1.0,
        contamination_rate=0.0,
        lean_gain_rate=0.0,
    )
    opts.update(kw)
    return generate(**opts)


def _by_metric(result):
    return {r["metric"]: r for r in result["metrics"]}


# ---------------------------------------------------------------- 统计小工具
def test_median_handles_both_parities():
    assert _median([3.0, 1.0, 2.0]) == 2.0
    assert _median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert _median([]) is None


def test_two_sided_p_matches_the_textbook_values():
    """1.96σ 对应 5%，这是「真实变化」那条线的出处，必须对得上。"""
    assert abs(_two_sided_p(0.0) - 1.0) < 1e-12
    assert abs(_two_sided_p(1.96) - 0.05) < 0.001
    assert abs(_two_sided_p(1.28) - 0.20) < 0.01
    assert abs(_two_sided_p(-1.96) - 0.05) < 0.001  # 双侧，符号不影响


# ---------------------------------------------------------------- BH 校正
def test_bh_matches_a_hand_calculation():
    """手算：n=3，最小的 p 值 0.001 × 3/1 = 0.003。

    其余两个按「从大到小取累计最小值」被顶到 0.75 / 0.9 ——
    它们的显著性完全靠最小的那个撑着。
    """
    q = _bh([0.001, 0.5, 0.9])
    assert abs(q[0] - 0.003) < 1e-12
    assert abs(q[1] - 0.75) < 1e-12
    assert abs(q[2] - 0.9) < 1e-12


def test_bh_keeps_the_input_order():
    """q 值必须和输入一一对应 —— 顺序错了就会给错的指标降级。"""
    q = _bh([0.9, 0.001, 0.5])
    assert abs(q[1] - 0.003) < 1e-12  # 最小的 p 在中间
    assert q[0] > q[1] and q[0] > q[2]


def test_bh_is_monotone_in_p():
    """p 值排好序之后，q 值也必须排好序 —— 这是 BH 的单调性保证。"""
    pvals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.9]
    q = _bh(pvals)
    assert q == sorted(q)


def test_bh_never_exceeds_one():
    assert all(q <= 1.0 for q in _bh([0.5, 0.6, 0.7, 0.99]))


def test_bh_on_degenerate_inputs():
    assert _bh([]) == []
    assert _bh([1.0, 1.0, 1.0]) == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------- ★ 核心场景
def test_weight_falls_but_fat_mass_stays_inside_the_noise():
    """★ 整个项目的核心场景。

    构造：真脂肪质量恒定不变，真瘦体重缓慢流失（120 天掉 2 kg，
    相当于节食但没配合力量训练的那种"掉肌肉"）。
    于是体重实实在在掉了 2 kg，而脂肪**一动没动**。

    工具必须同时说出两句话：
        · 体重 —— 真实变化（这 2 kg 是真的掉了）
        · 脂肪质量 —— 在噪声内（但掉的不是脂肪）
    街上的体脂秤在这里只会欢呼"你减了 2 kg"。
    """
    records, truth = generate(
        segments=FLAT,
        seed=1,
        measure_prob=1.0,
        contamination_rate=0.0,
        lean_gain_rate=-2.0 / 120.0,  # 瘦体重 120 天掉 2 kg
    )
    result = verdicts(records, edge_n=10)
    m = _by_metric(result)

    # 先确认合成数据真的造出了那个场景（否则测的是别的东西）
    t = truth["rows"]
    assert abs(t[0]["true_fat_mass_kg"] - t[-1]["true_fat_mass_kg"]) < 0.01
    assert t[0]["true_weight_kg"] - t[-1]["true_weight_kg"] > 1.5

    assert m["weight_kg"]["verdict"] == VERDICT_REAL
    assert m["weight_kg"]["change"] < -1.5

    assert m["fat_mass_kg"]["verdict"] == VERDICT_NOISE
    assert abs(m["fat_mass_kg"]["change"]) < m["fat_mass_kg"]["threshold"]


def test_the_fat_mass_threshold_is_about_six_kilograms():
    """脂肪质量的判定门槛应该是 6 kg 量级，不是文献里常被引用的 ±3pp。

    这里是容易搞错的一步：±3pp 是**方法**误差，决定的是「你那个
    25% 到底是不是 25%」。而判定「变化」要用**日间**误差，它从
    √2 × 3pp × 体重 传过来。两者差了一个 √2 和一次传播。
    """
    records, _ = _flat_series(seed=2)
    m = _by_metric(verdicts(records, edge_n=10))

    thr = m["fat_mass_kg"]["threshold"]
    assert 4.0 < thr < 9.0, thr


def test_a_measured_metrics_threshold_is_196_root2_te():
    """实测指标的门槛 = 1.96 × √2 × TE，两个倍数缺一不可。"""
    records, _ = _flat_series(seed=3)
    m = _by_metric(verdicts(records, edge_n=10))

    r = m["weight_kg"]
    assert abs(r["threshold"] - Z_95 * math.sqrt(2.0) * r["te"]) < 1e-9
    assert abs(r["sigma_change"] - math.sqrt(2.0) * r["te"]) < 1e-9


# ---------------------------------------------------------------- ★ 假阳性
def test_a_pure_noise_series_produces_no_false_positives():
    """★ 一个只会说「变了」的工具没有价值。

    真值完全不变，只有测量噪声。工具必须在**每一个**种子上都
    说不出「真实变化」来 —— 不是大多数种子，是全部。
    """
    offenders = []
    for seed in range(20):
        records, _ = _flat_series(seed=seed)
        m = _by_metric(verdicts(records, edge_n=10))
        for r in m.values():
            if r["verdict"] == VERDICT_REAL:
                offenders.append((seed, r["metric"], r["change"], r["q_value"]))

    assert not offenders, f"纯噪声序列上误报了真实变化：{offenders[:5]}"


def test_a_clear_cut_fat_is_actually_detected():
    """反过来也要成立 —— 真的大幅减脂必须被认出来。
    只会说「没变」的工具同样没有价值。"""
    records, _ = generate(
        segments=({"days": 150, "fat_rate": -0.10, "note": "狠减"},),
        seed=4,
        measure_prob=1.0,
        contamination_rate=0.0,
    )
    m = _by_metric(verdicts(records, edge_n=10))
    assert m["fat_mass_kg"]["verdict"] == VERDICT_REAL
    assert m["fat_mass_kg"]["change"] < -10.0


# ---------------------------------------------------------------- 边界
def test_a_single_record_cannot_be_judged():
    """一个点做不了判定 —— 必须说「数据不足」，不能硬给结论。"""
    records, _ = generate(segments=({"days": 1, "fat_rate": 0.0, "note": "x"},),
                          seed=5, measure_prob=1.0, contamination_rate=0.0)
    r = verdict_for("weight_kg", records, {"weight_kg": 0.5})
    assert r["verdict"] == VERDICT_UNKNOWN
    assert "太少" in r["reason"]


def test_a_missing_noise_floor_gives_unknown_not_a_guess():
    """估不出 TE 就不能判定。宁可说不知道，也不能拿个默认值凑。"""
    records, _ = _flat_series(seed=6)
    r = verdict_for("fat_mass_kg", records, {})  # 空的 te_map
    assert r["verdict"] == VERDICT_UNKNOWN
    assert "典型误差" in r["reason"]


def test_an_unknown_metric_raises():
    records, _ = _flat_series(seed=7)
    with pytest.raises(ValueError):
        verdict_for("火星体重", records, {})


def test_every_metric_gets_exactly_one_row():
    """报告是按固定顺序排的，少一个或多一个都会串位。"""
    from fitxray.verdict import METRIC_ORDER

    records, _ = _flat_series(seed=8)
    result = verdicts(records)
    assert [r["metric"] for r in result["metrics"]] == list(METRIC_ORDER)


# ---------------------------------------------------------------- 打包结果
def test_the_result_carries_records_for_the_report():
    """报告要画时间序列，记录必须跟着结果一起回来。

    让调用方自己再传一遍容易传错顺序 —— 这个测试守着那个接口。
    """
    records, _ = _flat_series(seed=9)
    result = verdicts(records)
    assert result["records"] is records
    assert len(result["records"]) == result["n_points"]


def test_ignored_metrics_all_explain_themselves():
    """「该忽略」必须给理由 —— 只给标签等于没给信息。"""
    records, _ = _flat_series(seed=10)
    ignored = verdicts(records)["ignored"]
    assert ignored
    for ig in ignored:
        assert ig["label"] and ig["why"]
        assert len(ig["why"]) > 15


def test_ignored_metrics_do_not_appear_in_the_verdict_table():
    """基础代谢之类的公式产物只出现在「该忽略」里，
    不能混进判定表再占一个检验名额。"""
    from fitxray.schema import IGNORE_METRICS

    records, _ = _flat_series(seed=11)
    result = verdicts(records)
    judged = {r["metric"] for r in result["metrics"]}
    assert not (judged & set(IGNORE_METRICS))


def test_the_downgrade_note_is_written_into_the_reason():
    """被 FDR 降级的指标，理由里必须留下痕迹 —— 不能悄悄改判决。"""
    records, _ = _flat_series(seed=12)
    result = verdicts(records)
    for metric in result["downgraded"]:
        row = _by_metric(result)[metric]
        assert row["verdict"] == VERDICT_BORDER
        assert "多重比较" in row["reason"]


def test_only_directly_measured_metrics_get_a_noise_floor():
    """派生量的 TE 是传过去的，不该自己去估 —— 那不是独立信息。"""
    from fitxray.verdict import DIRECTLY_MEASURED

    records, _ = _flat_series(seed=13)
    result = verdicts(records)
    assert set(result["noise"]) == set(DIRECTLY_MEASURED)


def test_a_clean_series_has_nothing_flagged():
    records, _ = _flat_series(seed=14)
    result = verdicts(records)
    assert result["flagged"] == []
    assert result["n_clean"] == result["n_points"]


def test_contaminated_points_are_downgraded_not_dropped():
    """★ 只降级，不删除 —— 端到端再确认一次。

    被标记的点必须仍然留在 records 里。删掉就等于删掉了
    最该被解释的那次波动（生病、高钠、经期）。
    """
    records, truth = generate(seed=15, measure_prob=1.0, contamination_rate=0.15)
    result = verdicts(records)

    assert len(result["records"]) == len(records)
    assert result["n_points"] == len(records)
    assert result["n_clean"] < result["n_points"]
    # 注入口是真相给的，标记是工具判的，两边都非空
    assert len(truth["contaminated_indices"]) > 0
    assert len(result["flagged"]) > 0


def test_edge_n_one_still_works():
    """首尾各取一个点是最脆的设置，但不该崩。"""
    records, _ = _flat_series(seed=16)
    result = verdicts(records, edge_n=1)
    assert all(r["verdict"] for r in result["metrics"])
    assert EDGE_N == 3  # 默认值没被改坏


# ---------------------------------------------------------------- 排版
def test_no_markdown_leaks_into_the_terminal_report():
    """★ 理由文字里的 `**` 是给 HTML 报告排版用的。

    同一批理由也会走到终端，而终端不认识 Markdown —— 不处理的话
    用户会看到 `**确定性函数**` 这种裸标记。report.py 那边有 `_rich`
    负责转成 <b>，这边必须有 `_plain` 负责抹掉。
    """
    from fitxray.verdict import format_verdicts

    records, _ = _flat_series(seed=17)
    text = format_verdicts(verdicts(records))
    assert "**" not in text


def test_plain_strips_bold_but_keeps_the_words():
    from fitxray.verdict import _plain

    assert _plain("是瘦体重的**确定性函数**") == "是瘦体重的确定性函数"
    assert _plain("没有标记") == "没有标记"
    assert _plain("") == ""
    assert _plain(None) is None
