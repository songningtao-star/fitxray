"""
读测量记录和体征档案。

输入格式有两条路，都不需要你改代码：

    CSV    从秤的 App 导出，或者自己用 Excel 记 —— 最省事的做法
    JSONL  一行一条 JSON，字段更自由

读进来之后会给每条记录附上两个内部字段：

    _day   从**最早那条记录**算起的浮点天数（含小时的小数部分）
    _hour  当天的小时数（7.5 表示 7:30），没记时刻就是 None

这两个字段是整个项目能工作的前提 —— 噪声估计、可比性判定、
变化量计算全都要用**真实的时间间隔**，而不是"第几条记录"。
采样不规则的时候（有人天天称，有人一周两次），按条数算间隔是错的。
"""

import csv
import datetime as _dt
import json
import os

from .schema import METRICS

# 真值解析表。留空 / 缺失一律当 None（未知），**不能当成 False** ——
# 「没记录有没有运动」和「确定没运动」是两回事，判定时区别对待。
_TRUE = {"true", "t", "1", "yes", "y", "是", "有"}
_FALSE = {"false", "f", "0", "no", "n", "否", "无"}

_NUMERIC_FIELDS = (
    "weight_kg",
    "body_fat_pct",
    "skeletal_muscle_kg",
    "body_water_pct",
)
_BOOL_FIELDS = ("fasted", "trained_8h", "alcohol_8h")


def _to_bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s == "":
        return None
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def _to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"这个值不是数字：「{v}」")


def _parse_date(s, where):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"{where}：日期格式认不出来「{s}」，支持 2026-01-15 / 2026/01/15")


def _parse_hour(s):
    """把 "07:30" / "7:30" / "7" 解析成浮点小时。"""
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    if ":" in s:
        hh, _, mm = s.partition(":")
        try:
            return int(hh) + int(mm or 0) / 60.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------- profile
def load_profile(path):
    """读体征档案（YAML 或 JSON）。

    需要 sex / age / height_cm / device_id 里的哪些，由下游决定 ——
    这里只负责读进来，不做校验强迫。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到体征档案：{path}")

    with open(path, encoding="utf-8") as f:
        text = f.read()

    if path.endswith(".json"):
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "读 YAML 需要 PyYAML：pip install PyYAML。"
                "或者把档案存成 JSON（改成 .json 后缀即可）"
            )
        data = yaml.safe_load(text)

    if not isinstance(data, dict):
        raise ValueError(f"体征档案应该是一个字典，读到的却是 {type(data).__name__}")
    return data


# ---------------------------------------------------------------- measurements
def _rows_from_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _rows_from_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{e}")
    return rows


def _normalize(row, where, device_id=None):
    """把一行原始数据整理成标准记录。"""
    if "date" not in row or str(row.get("date", "")).strip() == "":
        raise ValueError(f"{where}：缺 date 字段")

    rec = {
        "date": _parse_date(row["date"], where).isoformat(),
        "_date": _parse_date(row["date"], where),
        "_hour": _parse_hour(row.get("time")),
        "time": (str(row.get("time")).strip() or None) if row.get("time") else None,
        "note": (str(row.get("note", "")).strip() or None),
        "device_id": row.get("device_id") or device_id,
    }
    for fld in _NUMERIC_FIELDS:
        rec[fld] = _to_float(row.get(fld))
    for fld in _BOOL_FIELDS:
        rec[fld] = _to_bool(row.get(fld))

    if rec["weight_kg"] is None:
        raise ValueError(f"{where}：缺 weight_kg —— 这个是必须有的")
    if rec["body_fat_pct"] is None:
        raise ValueError(
            f"{where}：缺 body_fat_pct。"
            f"没有体脂率就算不出脂肪质量，而脂肪质量是这个项目的核心指标"
        )
    return rec


def attach_day(records):
    """排序并给每条记录附上 `_day`（从最早那条算起的浮点天数）。

    单独抽出来是因为 synth 生成的记录也要走这一步 —— 派生逻辑只该有一个出处，
    否则两边的口径迟早会不一致。
    """
    if not records:
        return records
    records.sort(
        key=lambda r: (r["_date"], r["_hour"] if r["_hour"] is not None else 0.0)
    )
    first = records[0]["_date"]
    for r in records:
        offset = (r["_date"] - first).days
        r["_day"] = offset + (r["_hour"] / 24.0 if r["_hour"] is not None else 0.0)
    return records


def load_measurements(path, device_id=None):
    """读测量记录，返回按时间排好序的标准记录列表。

    自动识别 CSV / JSONL（按扩展名）。读完之后：
        · 按日期排序
        · 附上 `_day`（从最早那条算起的浮点天数）
        · 缺字段的补 None，不会静默当成 0

    重复日期**不合并** —— 同一天测两次是合法的，可比性判定会处理它。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到测量记录：{path}")

    if path.lower().endswith(".jsonl") or path.lower().endswith(".ndjson"):
        rows = _rows_from_jsonl(path)
    else:
        rows = _rows_from_csv(path)

    if not rows:
        raise ValueError(f"{path} 里一条记录都没有")

    records = [
        _normalize(r, f"{os.path.basename(path)} 第 {i} 条", device_id=device_id)
        for i, r in enumerate(rows, 2)  # 2 = 表头占了一行
    ]
    records.sort(key=lambda r: (r["_date"], r["_hour"] if r["_hour"] is not None else 0.0))

    attach_day(records)
    return records


def missing_report(records):
    """统计每个字段的缺失情况。

    不是每个字段都必须有（比如时刻、运动标记），但用户应该知道
    自己的数据缺了什么 —— 缺得多了，有些判据就自动失效了。
    """
    n = len(records)
    out = {}
    for fld in _NUMERIC_FIELDS + _BOOL_FIELDS + ("time",):
        missing = sum(1 for r in records if r.get(fld) is None)
        if missing:
            out[fld] = missing
    ignored = [k for k in METRICS if k not in _NUMERIC_FIELDS]
    return {"n": n, "missing": out, "derived": ignored}
