"""
comparability 的测试：相减之前先门控。

两个测试值得单独说一句：

`test_screening_flags_but_keeps_everything` 锁住的是"只降级，不删除"。
这条原则很容易在重构时被无意破坏 —— "顺手把异常点滤掉"看起来很自然，
但滤掉就等于删掉了最该被解释的那次波动（生病、高钠、经期）。

`test_a_real_jump_gets_flagged` 锁住的是一个真踩过的坑：突跳判据的
尺度如果拿"全部点对"去估，序列一有趋势，阈值就被抬到天上去，
这条判据实际上永远不会触发。必须用最近一档估。
"""

import datetime as _dt

import numpy as np

from fitxray.comparability import (
    JUMP_K,
    LEVEL_BAD,
    LEVEL_OK,
    LEVEL_WARN,
    PROTOCOL_RULES,
    compare,
    screen,
)
from fitxray.loaders import attach_day

START = _dt.date(2026, 1, 1)


def _rec(day, hour=7.5, weight=78.0, **kw):
    """造一条测量记录。默认是"早上空腹、没运动"的标准条件。

    day 是从 START 起的天数偏移，这样造 60 天的序列也不会撞上月份边界。
    """
    d = START + _dt.timedelta(days=day)
    r = {
        "date": d.isoformat(),
        "_date": d,
        "_hour": hour,
        "weight_kg": weight,
        "body_fat_pct": 25.0,
        "fasted": True,
        "trained_8h": False,
        "alcohol_8h": False,
        "note": "",
    }
    r.update(kw)
    return r


def _pair(a, b):
    attach_day([a, b])
    return compare(a, b)


# ---------------------------------------------------------------- 单条判据
def test_a_clean_pair_is_comparable():
    r = _pair(_rec(0), _rec(7, weight=77.5))
    assert r["verdict"] == LEVEL_OK
    assert r["flags"] == []  # 通过的判据不返回 —— 列一堆"通过"没有信息量
    assert abs(r["gap_days"] - 7.0) < 1e-9


def test_different_devices_are_not_comparable():
    """跨设备不是"误差变大"，是坐标系都换了，所以直接判不可比。"""
    r = _pair(_rec(0, device_id="mi-scale-2"), _rec(7, device_id="huawei-3"))
    assert r["verdict"] == LEVEL_BAD
    assert [f["rule"] for f in r["flags"]] == ["device"]


def test_missing_device_id_is_not_treated_as_a_mismatch():
    """没标设备 ≠ 设备不同。宁可放过，不可误判。"""
    assert _pair(_rec(0), _rec(7))["verdict"] == LEVEL_OK


def test_same_day_measurements_are_not_comparable():
    """日内波动本身就有 1~2 kg，同一天测两次差的主要是吃喝排空。"""
    r = _pair(_rec(0, hour=7.0), _rec(0, hour=21.0))
    assert "interval" in [f["rule"] for f in r["flags"]]
    assert r["verdict"] == LEVEL_BAD


def test_a_big_time_of_day_gap_only_warns():
    r = _pair(_rec(0, hour=7.0), _rec(4, hour=21.0))
    assert r["verdict"] == LEVEL_WARN
    assert [f["rule"] for f in r["flags"]] == ["time_of_day"]


def test_time_of_day_is_skipped_when_the_hour_is_missing():
    """没记时刻就不能假装判过了 —— 缺信息时宁愿不判。"""
    assert _pair(_rec(0, hour=None), _rec(7, hour=None))["verdict"] == LEVEL_OK


def test_time_of_day_wraps_across_midnight():
    """23:00 和 01:00 只差 2 小时，不该被判成差了 22 小时。"""
    r = _pair(_rec(0, hour=23.0), _rec(7, hour=1.0))
    assert "time_of_day" not in [f["rule"] for f in r["flags"]]


def test_mixed_fasting_state_only_warns():
    r = _pair(_rec(0), _rec(7, fasted=False))
    assert r["verdict"] == LEVEL_WARN
    assert [f["rule"] for f in r["flags"]] == ["fasted"]


def test_fasting_is_skipped_when_unknown():
    assert _pair(_rec(0, fasted=None), _rec(7))["verdict"] == LEVEL_OK


def test_post_exercise_is_not_comparable():
    """这一条直接判不可比，因为它会**系统性地**把体脂率压下去 ——
    练完称出来的「进步」是假的。方向固定，不是随机误差。"""
    r = _pair(_rec(0), _rec(7, trained_8h=True))
    assert r["verdict"] == LEVEL_BAD
    assert [f["rule"] for f in r["flags"]] == ["exercise"]


def test_alcohol_only_warns():
    assert _pair(_rec(0), _rec(7, alcohol_8h=True))["verdict"] == LEVEL_WARN


def test_the_worst_flag_decides():
    """一条不可比就整体不可比，不会被"需谨慎"冲淡。"""
    r = _pair(
        _rec(0, hour=6.0),
        _rec(7, hour=20.0, alcohol_8h=True, trained_8h=True),
    )
    assert r["verdict"] == LEVEL_BAD
    assert len(r["flags"]) >= 3


def test_false_is_not_falsey():
    """`trained_8h = False` 是明确地说"没练"，不该触发判据。
    这条同时守着 loaders 那边的解析 —— 缺失要变 None，不能变 False。"""
    r = _pair(_rec(0), _rec(7))
    assert all(f["rule"] != "exercise" for f in r["flags"])


def test_protocol_rules_all_have_a_stated_reason():
    """清单是给人看的，每条都得写清楚依据 —— 不能只有结论。"""
    for r in PROTOCOL_RULES:
        assert r["name"] and r["why"]
        assert len(r["why"]) > 15


# ---------------------------------------------------------------- 序列筛查
def _series(n=30, trained_at=(), wobble=True, base=78.0, step=0.0, seed=5):
    """造一条序列。wobble 给一点随机的小抖动 —— 完全没抖动的序列
    估不出噪声，很多判据会自动失效，那样测的就不是判据本身了。

    抖动必须用**随机**的，不能拿 `(d % 3)` 这种周期函数凑。
    周期 3 的抖动只有三个不同取值，差分里超过一半彼此相同，
    MAD 直接算成 0 —— 噪声估成 0，突跳判据就静默失效了，
    测出来的"通过"是假的。种子固定，所以结果可复现。
    """
    rng = np.random.default_rng(seed)
    recs = []
    for d in range(n):
        w = base + step * d
        if wobble:
            w += float(rng.normal(0.0, 0.15))
        recs.append(
            _rec(d, weight=w,
                 trained_8h=(d in trained_at),
                 note="晚上练后测" if d in trained_at else "")
        )
    attach_day(recs)
    return recs


def test_screening_flags_but_keeps_everything():
    """★ 只降级，不删除。

    被标记的点仍然在记录里（报告要显示它、图要画它），
    只是不出现在 clean_idx 里。一个都不能少。
    """
    recs = _series(30, trained_at={5, 12})
    r = screen(recs)

    assert len(r["clean_idx"]) + len(r["flagged"]) == len(recs)
    assert {f["index"] for f in r["flagged"]} == {5, 12}
    assert 5 not in r["clean_idx"] and 12 not in r["clean_idx"]
    assert all(0 <= i < len(recs) for i in r["clean_idx"])


def test_screening_reports_the_protocol_reason():
    flagged = {f["index"]: f for f in screen(_series(20, trained_at={7}))["flagged"]}
    assert flagged[7]["level"] == LEVEL_BAD
    assert any("运动后" in why for why in flagged[7]["reasons"])


def test_screening_estimates_noise_before_and_after():
    """两个值都报出来 —— 只报一个等于把"筛掉了什么"藏起来了。"""
    r = screen(_series(40, trained_at={10, 25}))
    assert r["te_before"] is not None and r["te_after"] is not None
    assert r["te_before"] > 0 and r["te_after"] > 0


def test_screening_is_deterministic():
    """走一遍就停，不做收敛迭代 —— 同样的输入必须给同样的输出。"""
    a = screen(_series(40))
    b = screen(_series(40))
    assert a["clean_idx"] == b["clean_idx"]
    assert a["jump_flagged"] == b["jump_flagged"]


def test_a_steady_trend_flags_nothing():
    """序列是带趋势的，判据必须用**局部**中位数当基准。

    拿全序列的中位数当基准，趋势的两端会被整片判成异常：
    50 天累计掉 4 kg，序列开头离中位数就有 2 kg，
    远超 4 倍典型误差，两头会被整片误标。

    局部中位数下，一个点只比邻居偏半个趋势步长（0.08 kg），
    离阈值还远。所以这里一个点都不该标。
    """
    r = screen(_series(50, base=88.0, step=-0.08))
    assert r["jump_flagged"] == 0
    assert len(r["clean_idx"]) == 50


def test_a_real_jump_gets_flagged():
    """★ 防回归测试 —— 这条是有真实数字支撑的。

    场景：60 天的减脂期（每天 0.10 kg），第 30 天突然高钠 / 生病，
    体重一次跳了 1.5 kg。这种波动**必须**被标出来，它就是
    "只降级不删除"那条原则存在的理由。

    坑在这里：突跳判据的尺度如果拿**全部点对**去估，长间隔的点对里
    混的全是这 6 kg 的真实减脂，噪声被估成 1.38 kg，
    阈值抬到 5.5 kg —— 一个 1.5 kg 的波动连门槛都摸不到，
    这条判据永远不触发。实测：旧口径下注入 2 kg 也照样漏。

    用最近一档估才对（和 noise.py 里那个决定同一个道理）：
    相邻两天之间趋势还没累积起来，估出来才是噪声本身（0.18 kg）。
    """
    recs = _series(61, base=80.0, step=-0.10)
    recs[30]["weight_kg"] += 1.5
    attach_day(recs)

    r = screen(recs)
    assert 30 in {f["index"] for f in r["flagged"]}
    assert r["jump_flagged"] == 1


def test_jump_threshold_is_reported_in_the_reason():
    recs = _series(61, base=80.0, step=-0.10)
    recs[30]["weight_kg"] += 1.5
    attach_day(recs)
    flagged = {f["index"]: f for f in screen(recs)["flagged"]}[30]
    assert str(int(JUMP_K)) in flagged["reasons"][0]


def test_screening_on_a_tiny_series_does_not_crash():
    recs = _series(2)
    r = screen(recs)
    assert r["te_before"] is None and r["te_after"] is None
    assert r["clean_idx"] == [0, 1]
    assert r["flagged"] == []


def test_screening_handles_a_series_with_no_wobble():
    """全是同一个值 → 差分全 0，估不出噪声。不能崩，判据自动失效即可。"""
    r = screen(_series(30, wobble=False))
    assert len(r["clean_idx"]) == 30
    assert r["jump_flagged"] == 0
