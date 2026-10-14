"""
loaders 的测试：把外部的脏数据变成内部干净记录的这一层。

最要紧的是**布尔值的三态**：真 / 假 / 不知道。用 CSV 记数据的时候，
空单元格和 "false" 长得一样近，一不小心就把"没记录"读成"确定没运动"——
而这两者在可比性判定里是完全不同的待遇（一个跳过判据，一个触发判据）。
所以这里整整有一组测试专门守这件事。
"""

import datetime as _dt

import pytest

from fitxray.loaders import (
    _parse_date,
    _parse_hour,
    _to_bool,
    attach_day,
    load_measurements,
    load_profile,
    missing_report,
)

HEADER = "date,time,weight_kg,body_fat_pct,skeletal_muscle_kg,body_water_pct,fasted,trained_8h,alcohol_8h,note"


def _csv(tmp_path, *rows, header=HEADER):
    p = tmp_path / "m.csv"
    p.write_text("\n".join([header] + list(rows)), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------- 三态布尔
def test_missing_boolean_is_none_not_false():
    """★ 这个文件最重要的一个测试。

    空单元格必须读成 None（不知道），不能读成 False（确定没运动）。
    读成 False 的话，一条没记运动信息的记录会被当成"确认没运动"，
    判据就白过了 —— 而且错得悄无声息。
    """
    assert _to_bool(None) is None
    assert _to_bool("") is None
    assert _to_bool("   ") is None
    assert _to_bool("随便写的") is None  # 认不出来就说不知道，不猜


def test_boolean_true_forms():
    for s in ("true", "True", "TRUE", "t", "1", "yes", "y", "是", "有"):
        assert _to_bool(s) is True, s


def test_boolean_false_forms():
    for s in ("false", "False", "f", "0", "no", "n", "否", "无"):
        assert _to_bool(s) is False, s


def test_real_booleans_pass_through():
    """JSONL 里读到的是真 bool，不能再转一道。"""
    assert _to_bool(True) is True
    assert _to_bool(False) is False


def test_boolean_zero_is_false_not_missing():
    """数字 0 和空字符串是两回事 —— str(0).strip() 不是空的。
    这条守着上面那条 `s == ""` 的判断别写成 `if not s`。"""
    assert _to_bool(0) is False
    assert _to_bool(1) is True


# ---------------------------------------------------------------- 日期与时刻
def test_four_date_formats_are_all_accepted():
    want = _dt.date(2026, 1, 15)
    for s in ("2026-01-15", "2026/01/15", "2026.01.15", "20260115"):
        assert _parse_date(s, "第 2 条") == want, s


def test_a_bad_date_says_which_row_and_what_is_supported():
    with pytest.raises(ValueError) as e:
        _parse_date("15/01/2026", "第 7 条")
    msg = str(e.value)
    assert "第 7 条" in msg and "2026-01-15" in msg


def test_hour_parsing():
    assert _parse_hour("07:30") == 7.5
    assert _parse_hour("7:30") == 7.5
    assert _parse_hour("7") == 7.0
    assert _parse_hour("19:00") == 19.0
    assert _parse_hour(None) is None
    assert _parse_hour("") is None


# ---------------------------------------------------------------- _day
def test_attach_day_uses_real_intervals_not_row_numbers():
    """★ 采样不规则是常态（有人天天称，有人一周两次）。

    按"第几条记录"算间隔，噪声估计和可比性判定全都会错。
    所以这里附的是**真实天数**，而且带小时的小数部分。
    """
    recs = [
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": 7.5},
        {"date": "2026-01-02", "_date": _dt.date(2026, 1, 2), "_hour": 7.5},
        {"date": "2026-01-09", "_date": _dt.date(2026, 1, 9), "_hour": 7.5},
    ]
    attach_day(recs)
    assert recs[0]["_day"] == 0.0 + 7.5 / 24
    assert recs[1]["_day"] == 1.0 + 7.5 / 24
    assert recs[2]["_day"] == 8.0 + 7.5 / 24
    # 第 1 条和第 2 条相邻，第 3 条隔了 7 天 —— 间隔不是常数
    assert recs[2]["_day"] - recs[1]["_day"] == 7.0


def test_attach_day_sorts_in_place():
    recs = [
        {"date": "2026-01-09", "_date": _dt.date(2026, 1, 9), "_hour": 7.5},
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": 7.5},
    ]
    attach_day(recs)
    assert recs[0]["date"] == "2026-01-01"
    assert recs[1]["date"] == "2026-01-09"


def test_same_day_records_sort_by_hour():
    recs = [
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": 20.0},
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": 7.0},
    ]
    attach_day(recs)
    assert [r["_hour"] for r in recs] == [7.0, 20.0]


def test_a_missing_hour_sorts_as_midnight():
    recs = [
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": 20.0},
        {"date": "2026-01-01", "_date": _dt.date(2026, 1, 1), "_hour": None},
    ]
    attach_day(recs)
    assert recs[0]["_hour"] is None


def test_attach_day_on_an_empty_list():
    assert attach_day([]) == []


# ---------------------------------------------------------------- 读文件
def test_load_measurements_reads_csv_and_sorts(tmp_path):
    path = _csv(
        tmp_path,
        "2026-01-03,07:30,77.9,24.8,,,true,false,false,",
        "2026-01-01,07:30,78.2,25.0,,,true,false,false,",
    )
    recs = load_measurements(path)
    assert [r["date"] for r in recs] == ["2026-01-01", "2026-01-03"]
    assert recs[0]["weight_kg"] == 78.2
    assert recs[0]["_day"] == 7.5 / 24


def test_an_excel_exported_bom_does_not_break_the_header(tmp_path):
    """Excel 导出的 CSV 头里有个 BOM，不处理的话第一列名会是
    `\\ufeffdate` —— 于是每一行都"缺 date"，报错还看不出为什么。"""
    p = tmp_path / "bom.csv"
    p.write_text(f"{HEADER}\n2026-01-01,07:30,78.2,25.0,,,true,false,false,\n",
                 encoding="utf-8-sig")
    assert load_measurements(str(p))[0]["weight_kg"] == 78.2


def test_blank_optionals_become_none(tmp_path):
    path = _csv(tmp_path, "2026-01-01,07:30,78.2,25.0,,,true,false,false,")
    r = load_measurements(path)[0]
    assert r["skeletal_muscle_kg"] is None
    assert r["body_water_pct"] is None
    assert r["note"] is None


def test_missing_weight_is_an_error(tmp_path):
    path = _csv(tmp_path, "2026-01-01,07:30,,25.0,,,true,false,false,")
    with pytest.raises(ValueError) as e:
        load_measurements(path)
    assert "weight_kg" in str(e.value)


def test_missing_body_fat_says_why_it_matters(tmp_path):
    """体脂率缺了不是"少一个字段"，是核心指标没了。
    报错必须说清楚这一点，不然用户会以为是格式问题。"""
    path = _csv(tmp_path, "2026-01-01,07:30,78.2,,,,true,false,false,")
    with pytest.raises(ValueError) as e:
        load_measurements(path)
    assert "脂肪质量" in str(e.value)


def test_a_missing_date_column_is_an_error(tmp_path):
    p = tmp_path / "nodate.csv"
    p.write_text("weight_kg,body_fat_pct\n78.2,25.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_measurements(str(p))


def test_duplicate_dates_are_kept_not_merged(tmp_path):
    """同一天测两次是合法的（早晚各一次），合并掉就没法判可比性了。"""
    path = _csv(
        tmp_path,
        "2026-01-01,07:00,78.2,25.0,,,true,false,false,",
        "2026-01-01,20:00,79.0,25.6,,,false,false,false,",
    )
    recs = load_measurements(path)
    assert len(recs) == 2
    assert recs[0]["_day"] != recs[1]["_day"]


def test_jsonl_input_and_comments(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        "# 这是注释，会被跳过\n"
        '{"date": "2026-01-01", "time": "07:30", "weight_kg": 78.2, "body_fat_pct": 25.0}\n'
        "\n"
        '{"date": "2026-01-02", "weight_kg": 78.0, "body_fat_pct": 24.9}\n',
        encoding="utf-8",
    )
    recs = load_measurements(str(p))
    assert len(recs) == 2
    assert recs[0]["fasted"] is None  # 没写就是不知道，不是 False
    assert recs[1]["_hour"] is None


def test_a_broken_jsonl_line_names_the_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"date": "2026-01-01", "weight_kg": 78.2}\n{不是 JSON}\n',
                 encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_measurements(str(p))
    assert "第 2 行" in str(e.value)  # 行号必须报出来，不然大文件没法查


def test_an_empty_file_is_an_error(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text(HEADER + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_measurements(str(p))


def test_a_missing_file_says_which_path(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        load_measurements(str(tmp_path / "没有这个文件.csv"))
    assert "没有这个文件" in str(e.value)


def test_profile_device_id_fills_in_for_rows_that_omit_it(tmp_path):
    path = _csv(tmp_path, "2026-01-01,07:30,78.2,25.0,,,true,false,false,")
    r = load_measurements(path, device_id="mi-scale-2")[0]
    assert r["device_id"] == "mi-scale-2"


# ---------------------------------------------------------------- profile
def test_load_profile_from_json(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text('{"sex": "male", "device_id": "mi-scale-2"}', encoding="utf-8")
    assert load_profile(str(p))["device_id"] == "mi-scale-2"


def test_load_profile_rejects_a_non_dict(tmp_path):
    p = tmp_path / "profile.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        load_profile(str(p))
    assert "字典" in str(e.value)


def test_the_shipped_profile_has_a_device_id(tmp_path):
    """仓库里那份 profile 必须带 device_id —— 没有它就判不了跨设备。"""
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prof = load_profile(os.path.join(root, "data", "profile.yaml"))
    assert prof["device_id"]
    # 隐私：档案里不该出现姓名之类的定位信息
    assert not ({"name", "email", "phone"} & set(prof))


# ---------------------------------------------------------------- 缺失统计
def test_missing_report_counts_only_what_is_missing(tmp_path):
    path = _csv(
        tmp_path,
        "2026-01-01,07:30,78.2,25.0,33.1,55.2,true,false,false,",
        "2026-01-02,,78.0,24.9,,,true,false,false,",
    )
    rep = missing_report(load_measurements(path))
    assert rep["n"] == 2
    assert rep["missing"]["time"] == 1
    assert rep["missing"]["skeletal_muscle_kg"] == 1  # 只有第二条缺
    assert "weight_kg" not in rep["missing"]  # 全都有，就不列出来


def test_missing_report_lists_derived_metrics(tmp_path):
    path = _csv(tmp_path, "2026-01-01,07:30,78.2,25.0,33.1,55.2,true,false,false,")
    rep = missing_report(load_measurements(path))
    assert "fat_mass_kg" in rep["derived"]
