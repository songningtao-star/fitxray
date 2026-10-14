"""
把判定结果渲染成一份「自包含」的 HTML 报告。

自包含 = 一个 .html 文件，双击就能看，不联网、不装东西、不带任何前端框架。
别人 clone 下来跑一遍，就能直接看到一张能截图放进 README 的图。

---- 设计规则（顺序很重要：先决定图形，再决定颜色）----

1. **主视觉是「变化量 vs 它的门槛」，不是「体重曲线」。**
   每个指标一条横杆，从中线往外长；外面套一条灰带，就是它的判定阈值。
   一眼能看出来的事：**好几根杆子根本没伸出灰带**。
   这比任何数字都直接 —— 曲线图会告诉你"在下降"，
   这张图会告诉你"那个下降还埋在噪声里"。

2. **状态色是保留色，而且带图标和文字。**
   绿/黄/灰不是"三种系列色"，是三种状态。每种都配图标和一句中文，
   色盲、黑白打印、强制高对比模式下都还能读。

3. **数值不能只藏在悬浮提示里。**
   每根杆子右边直接写变化量，判定徽章直接写结论。
   变差函数那四张小图底下还有个「看数字」，把每一档都列出来。

4. **深色模式是另外挑的一套颜色，不是把浅色反相。**
   两套都跑过色觉校验（详见下面的注释），都显式声明。

5. **只用一条轴。** 变化量就是变化量，不叠百分比，不做双轴。

排版上的一个决定沿用 rag-xray：页面底色刻意比卡片底色深一档
（#f1f0ed vs #fcfcfb），卡片才浮得起来。

---- 关于配色的一点说明 ----

状态色最初想用「绿/黄/红」这个直觉组合，跑校验直接挂了：
绿和黄在红色盲下几乎重合（ΔE 3.8），而这两个颜色恰恰会在同一张表里挨着出现。
改成偏青的绿之后才拉开。**这件事本身就是项目立场的一个小小注脚** ——
看起来最合理的方案，量一下才知道不行。
"""

import html
from datetime import datetime

from .comparability import LEVEL_BAD, PROTOCOL_RULES
from .schema import IGNORE_METRICS, METHOD_UNCERTAINTY
from .verdict import (
    VERDICT_BORDER,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
)

# 判定 → (样式名, 图标, 一句话)
# 图标和文字是必须的：不能让颜色单独承担表意。
VERDICT_STYLE = {
    VERDICT_REAL: ("good", "✓", "越过了 95% 那条线"),
    VERDICT_BORDER: ("warn", "◐", "超过了典型误差，但没到 95%"),
    VERDICT_NOISE: ("mute", "○", "和噪声分不开"),
    VERDICT_UNKNOWN: ("mute", "…", "数据不够，不做判定"),
}

# 时间序列那张图里画哪两个指标 —— 其它四个都是这两个的函数
SERIES_METRICS = ("weight_kg", "body_fat_pct")

FONT_SANS = (
    'system-ui, -apple-system, "Segoe UI", "PingFang SC", "Hiragino Sans GB", '
    '"Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", sans-serif'
)
FONT_MONO = (
    'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)


def _esc(text):
    return html.escape(str(text if text is not None else ""))


def _rich(text):
    """转义，再把 `**重点**` 变成真正的粗体。

    代码注释里习惯用 Markdown 的星号标重点，那些字符串会一路流到这里。
    直接转义的话，页面上就会出现一对光秃秃的星号 —— 得不偿失。
    先转义再替换，顺序不能反，否则就等于开了个注入口子。
    """
    parts = _esc(text).split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def _signed(v, digits=2, unit=""):
    if v is None:
        return "—"
    return f"{v:+.{digits}f}{unit}"


def _plain(v, digits=2, unit=""):
    if v is None:
        return "—"
    return f"{v:.{digits}f}{unit}"


def _abs(v, digits=2, unit=""):
    if v is None:
        return "—"
    return f"{abs(v):.{digits}f}{unit}"


# ================================================================ 样式
def _css():
    return """
:root {
  color-scheme: light;
  /* 表面与平面：刻意拉开一档，卡片才浮得起来 */
  --surface-1:      #fcfcfb;
  --surface-2:      #f5f4f1;
  --page:           #f1f0ed;
  --text-primary:   #0b0b0b;
  --text-secondary: #52514e;
  --muted:          #898781;
  --grid:           #e6e5df;
  --axis:           #c3c2b7;
  --border:         rgba(11, 11, 11, 0.09);
  --series-1:       #2a78d6;
  --chip-bg:        rgba(11, 11, 11, 0.045);
  --shadow:         0 1px 2px rgba(11, 11, 11, 0.04), 0 2px 8px rgba(11, 11, 11, 0.03);

  /* 状态色。四色一组跑过色觉校验（浅色底 #fcfcfb）：
     最差相邻对 ΔE 9.5（protan）/ 16.1（正常视觉），全项通过。
     注意「好」用的是偏青的绿 —— 纯绿和琥珀在红色盲下会撞在一起。 */
  --good:           #0d8f6f;
  --warning:        #c9820a;
  --critical:       #d03b3b;

  /* 噪声带：和主序列同色系，但要弱到不抢戏 */
  --band:           rgba(42, 120, 214, 0.15);
  --band-edge:      rgba(42, 120, 214, 0.28);
  --flag-mark:      #898781;
  /* 「在噪声内」那根横杆的颜色。深色底上不能沿用轴线的灰 ——
     轴线要退到背景里，横杆得看得见，两个需求正好相反。 */
  --bar-mute:       #a9a89f;
}

@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --surface-2:      #232322;
    --page:           #0d0d0d;
    --text-primary:   #f0efec;
    --text-secondary: #c3c2b7;
    --muted:          #898781;
    --grid:           #2c2c2a;
    --axis:           #383835;
    --border:         rgba(255, 255, 255, 0.10);
    --series-1:       #3987e5;
    --chip-bg:        rgba(255, 255, 255, 0.07);
    --shadow:         none;
    /* 深色模式是**另外挑的一套**，不是浅色那套反相。
       深底上要重新拉开明度：校验通过（最差相邻对 ΔE 9.7 / 15.7）。 */
    --good:           #26a883;
    --warning:        #9c6f18;
    --critical:       #e06b6b;
    --band:           rgba(57, 135, 229, 0.20);
    --band-edge:      rgba(57, 135, 229, 0.34);
    --flag-mark:      #6f6f6a;
    --bar-mute:       #8f8e87;
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-1:      #1a1a19;
  --surface-2:      #232322;
  --page:           #0d0d0d;
  --text-primary:   #f0efec;
  --text-secondary: #c3c2b7;
  --muted:          #898781;
  --grid:           #2c2c2a;
  --axis:           #383835;
  --border:         rgba(255, 255, 255, 0.10);
  --series-1:       #3987e5;
  --chip-bg:        rgba(255, 255, 255, 0.07);
  --shadow:         none;
  --good:           #26a883;
  --warning:        #9c6f18;
  --critical:       #e06b6b;
  --band:           rgba(57, 135, 229, 0.20);
  --band-edge:      rgba(57, 135, 229, 0.34);
  --flag-mark:      #6f6f6a;
  --bar-mute:       #8f8e87;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: 56px 28px 80px;
  background: var(--page);
  color: var(--text-primary);
  font-family: __FONT_SANS__;
  font-size: 14px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

::selection { background: var(--chip-bg); }

.wrap { max-width: 1060px; margin: 0 auto; }

/* ================= 页头 ================= */
.masthead { position: relative; margin-bottom: 30px; padding-bottom: 22px; }
.masthead .wordmark {
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--series-1);
  margin-bottom: 10px;
}
.masthead h1 {
  margin: 0 0 8px;
  font-size: 30px;
  font-weight: 680;
  letter-spacing: -0.022em;
  line-height: 1.25;
}
.masthead .lede {
  margin: 0;
  max-width: 64ch;
  font-size: 14.5px;
  color: var(--text-secondary);
}
.masthead::after {
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: 0;
  height: 1px;
  background: linear-gradient(90deg, var(--axis), transparent 72%);
}

.chips { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 18px; }
.chip {
  font-size: 12px;
  padding: 5px 11px;
  border-radius: 999px;
  background: var(--surface-1);
  border: 1px solid var(--border);
  color: var(--muted);
  white-space: nowrap;
}
.chip b { color: var(--text-primary); font-weight: 620; margin-left: 3px; }

.theme-toggle {
  position: absolute;
  top: 0; right: 0;
  width: 34px; height: 34px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--surface-1);
  color: var(--text-secondary);
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
  transition: color .15s, border-color .15s, transform .15s;
}
.theme-toggle:hover { color: var(--text-primary); border-color: var(--axis); transform: rotate(18deg); }

/* ================= 卡片 ================= */
.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  box-shadow: var(--shadow);
  padding: 24px 26px 26px;
  margin-bottom: 16px;
}
.card > h2 {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 0 0 5px;
  font-size: 15.5px;
  font-weight: 640;
  letter-spacing: -0.005em;
}
.card > h2 .idx {
  font-size: 11.5px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
  letter-spacing: 0.06em;
}
.card > .sub {
  margin: 0 0 20px;
  font-size: 13px;
  color: var(--muted);
  max-width: 76ch;
}
.card > .sub b { color: var(--text-secondary); font-weight: 600; }

/* ================= hero ================= */
.hero-row {
  display: grid;
  grid-template-columns: minmax(250px, 0.9fr) 1.35fr;
  gap: 30px;
  align-items: center;
}
@media (max-width: 760px) { .hero-row { grid-template-columns: 1fr; gap: 22px; } }

.hero .value {
  font-size: 66px;
  font-weight: 690;
  line-height: 0.95;
  letter-spacing: -0.04em;
  font-variant-numeric: tabular-nums;
}
.hero .value small { font-size: 26px; font-weight: 620; margin-left: 3px; letter-spacing: -0.02em; }
.hero .label { font-size: 13px; color: var(--muted); margin-top: 10px; }
.hero .label b { color: var(--text-secondary); font-weight: 620; }

/* 单一比例对上限 → 用 meter，不用「只有一根柱子的柱状图」 */
.meter {
  margin-top: 14px;
  height: 9px;
  border-radius: 999px;
  background: var(--chip-bg);
  overflow: hidden;
}
.meter .fill { height: 100%; border-radius: 999px; background: var(--axis); }
.meter .fill.good { background: var(--good); }
.meter .fill.warn { background: var(--warning); }

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(104px, 1fr));
  gap: 2px;                       /* 2px 表面色缝隙，不画边框 */
  background: var(--grid);
  border-radius: 10px;
  overflow: hidden;
}
.tile { background: var(--surface-1); padding: 16px 18px; }
.tile .t-label { font-size: 12px; color: var(--muted); }
.tile .t-value { font-size: 23px; font-weight: 640; margin-top: 3px; letter-spacing: -0.01em; }
.tile .t-value small { font-size: 13px; font-weight: 500; color: var(--muted); margin-left: 2px; }

/* ================= 主视觉：变化量 vs 阈值 ================= */
.cmp-scroll { overflow-x: auto; margin: 0 -2px; padding: 0 2px; }
.cmp { min-width: 680px; }

.cmp-row {
  display: grid;
  grid-template-columns: 82px minmax(190px, 1fr) 56px 96px 112px;
  align-items: center;
  gap: 14px;
  padding: 7px 0;
}
.cmp-row.head {
  padding-bottom: 4px;
  font-size: 11.5px;
  color: var(--muted);
  border-bottom: 1px solid var(--axis);
  margin-bottom: 6px;
}
.cmp-name { font-size: 13px; color: var(--text-secondary); white-space: nowrap; }

/* 轨道：中线在 50%，灰带左右对称，横杆从中线往外长 */
.cmp-track { position: relative; height: 26px; }
.cmp-band {
  position: absolute;
  top: 4px; bottom: 4px;
  background: var(--band);
  border-left: 1px solid var(--band-edge);
  border-right: 1px solid var(--band-edge);
  border-radius: 3px;
}
.cmp-axisline {
  position: absolute;
  left: 50%; top: 0; bottom: 0;
  width: 1px;
  background: var(--axis);
}
/* 数据端 4px 圆角，起点贴中线 */
.cmp-bar {
  position: absolute;
  top: 7px; height: 12px;
  min-width: 3px;
}
.cmp-bar.pos { left: 50%; border-radius: 0 4px 4px 0; }
.cmp-bar.neg { right: 50%; border-radius: 4px 0 0 4px; }
.cmp-bar.good { background: var(--good); }
.cmp-bar.warn { background: var(--warning); }
.cmp-bar.mute { background: var(--bar-mute); }

.cmp-ratio {
  font-size: 13px;
  font-weight: 620;
  text-align: right;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.cmp-val { font-size: 13px; text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--text-secondary); }
.cmp-flag { font-size: 12.5px; white-space: nowrap; }
.cmp-flag.good { color: var(--good); }
.cmp-flag.warn { color: var(--warning); }
.cmp-flag.mute { color: var(--muted); }

.cmp-ruler {
  display: grid;
  grid-template-columns: 82px minmax(190px, 1fr) 56px 96px 112px;
  gap: 14px;
  margin-top: 2px;
  font-size: 11px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
/* 标尺上的刻度要落在轨道的具体百分比位置上，所以用绝对定位而不是均分 */
.cmp-ruler .ticks { position: relative; height: 14px; }
.cmp-ruler .ticks i { position: absolute; top: 0; font-style: normal; }

/* ================= 小型多图（变差函数） ================= */
/* 固定 2 列（4 个指标正好 2×2）。用 auto-fit 的话 1060px 宽会排成 3 列，
   第 4 个格子孤零零地掉到第二行，右下角还会露出网格底色。 */
.multiples {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 2px;                       /* 2px 表面色缝隙 */
  background: var(--grid);
  border-radius: 10px;
  overflow: hidden;
}
@media (max-width: 700px) {
  .multiples { grid-template-columns: 1fr; }
  .cell.filler { display: none; }
}
.cell { background: var(--surface-1); padding: 15px 17px 13px; }
.cell .c-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; }
.cell .c-name { font-size: 13px; font-weight: 600; }
.cell .c-cap { font-size: 12px; color: var(--muted); margin-top: 4px; font-variant-numeric: tabular-nums; }
.cell svg { display: block; width: 100%; height: auto; margin: 8px 0 2px; }

/* ================= 时间序列 ================= */
.series-wrap { display: grid; gap: 22px; }
.series-block .s-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 4px;
}
.series-block .s-name { font-size: 13.5px; font-weight: 620; }
.series-block .s-note { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.series-block svg { display: block; width: 100%; height: auto; }

/* ================= 图例 ================= */
.legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 18px;
  margin-top: 18px;
  font-size: 12px;
  color: var(--muted);
}
.legend .k { display: inline-flex; align-items: center; gap: 7px; }
.legend .sw { width: 15px; height: 11px; border-radius: 3px; display: inline-block; }
.legend .sw.line { height: 2px; border-radius: 0; }
.legend .sw.ring {
  width: 11px; height: 11px; border-radius: 999px;
  background: transparent; border: 2px solid var(--flag-mark);
}
.legend .sw.dot { width: 11px; height: 11px; border-radius: 999px; background: var(--series-1); }
.legend .sw.band {
  width: 22px; height: 11px;
  background: var(--band);
  border-left: 1px solid var(--band-edge);
  border-right: 1px solid var(--band-edge);
}

/* ================= 表格 ================= */
table.grid { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.grid th {
  text-align: left;
  font-size: 11.5px;
  font-weight: 550;
  color: var(--muted);
  padding: 0 14px 9px 0;
  border-bottom: 1px solid var(--axis);
  white-space: nowrap;
}
table.grid td {
  padding: 12px 14px 12px 0;
  border-bottom: 1px solid var(--grid);
  vertical-align: top;
}
table.grid tbody tr:last-child td { border-bottom: none; }
table.grid tbody tr { transition: background .1s; }
table.grid tbody tr:hover td { background: var(--chip-bg); }
table.grid .num { font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--text-secondary); }
table.grid .mono { font-family: __FONT_MONO__; font-size: 12.5px; color: var(--text-secondary); }
table.grid td.q { font-weight: 550; }

/* 变差函数那张数字表：很密，单独一套紧凑样式 */
table.dense { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
table.dense th {
  text-align: right; font-weight: 500; color: var(--muted);
  padding: 0 0 6px; border-bottom: 1px solid var(--grid);
}
table.dense th:first-child { text-align: left; }
table.dense td { text-align: right; padding: 5px 0; border-bottom: 1px solid var(--grid); color: var(--text-secondary); }
table.dense td:first-child { text-align: left; color: var(--text-primary); }
table.dense tbody tr:last-child td { border-bottom: none; }

/* ================= 徽章：图标 + 文字，不靠颜色单独表意 ================= */
.badge { display: inline-flex; align-items: center; gap: 5px; font-size: 12.5px; font-weight: 550; white-space: nowrap; }
.badge.good { color: var(--good); }
.badge.warn { color: var(--warning); }
.badge.mute { color: var(--muted); }
.badge.cut {
  color: var(--text-secondary);
  background: var(--chip-bg);
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 12px;
}

/* ================= 该忽略的清单 ================= */
.ignores { display: grid; gap: 10px; }
.ignore {
  display: grid;
  grid-template-columns: 128px 1fr;
  gap: 6px 16px;
  border: 1px solid var(--border);
  border-radius: 11px;
  padding: 15px 18px;
  transition: border-color .15s;
}
.ignore:hover { border-color: var(--axis); }
.ignore .n { font-size: 14px; font-weight: 620; }
.ignore .n .tag {
  display: block;
  font-size: 11px;
  font-weight: 500;
  color: var(--muted);
  margin-top: 2px;
}
.ignore .w { font-size: 13.5px; color: var(--text-secondary); }
@media (max-width: 620px) { .ignore { grid-template-columns: 1fr; } }

/* ================= 理由单元 ================= */
.why-list { display: grid; gap: 14px; margin: 0; }
.why-item { display: grid; grid-template-columns: 20px 1fr; gap: 12px; }
.why-item .m { color: var(--muted); font-size: 13px; }
.why-item .t { font-size: 13.5px; color: var(--text-secondary); }
.why-item .t b { color: var(--text-primary); font-weight: 620; }

/* ================= 折叠 ================= */
details {
  margin-top: 18px;
  border-top: 1px solid var(--border);
  padding-top: 14px;
}
details > summary {
  cursor: pointer;
  font-size: 12.5px;
  color: var(--muted);
  list-style: none;
  user-select: none;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸ "; color: var(--axis); }
details[open] > summary::before { content: "▾ "; }
details > summary:hover { color: var(--text-secondary); }
details .body { padding-top: 14px; }

.empty { color: var(--muted); font-size: 13.5px; padding: 4px 0; }

.foot {
  margin-top: 26px;
  padding-top: 18px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--muted);
  text-align: center;
}
.foot a { color: var(--muted); }

@media (max-width: 700px) {
  body { padding: 32px 16px 56px; }
  .masthead h1 { font-size: 23px; }
  .hero .value { font-size: 50px; }
  .card { padding: 20px 18px 22px; }
}

/* 打印 / 导出 PDF 时不要阴影和按钮，让内容贴着纸面 */
@media print {
  body { background: #fff; padding: 0; }
  .card { box-shadow: none; border-color: #ddd; break-inside: avoid; }
  .theme-toggle { display: none; }
  details { display: block; }
  details > summary { display: none; }
  details .body { padding-top: 8px; }
}
""".replace("__FONT_SANS__", FONT_SANS).replace("__FONT_MONO__", FONT_MONO)


# ================================================================ hero
def _pick_primary(rows):
    """挑一个当 hero 的指标：优先脂肪质量 —— 那是用户真正想减的东西。

    体重一定会动（吃顿饭就动），所以拿体重当头条没信息量。
    「脂肪质量」才是"我这段时间到底减掉了什么"的答案。
    """
    by = {r["metric"]: r for r in rows}
    for key in ("fat_mass_kg", "body_fat_pct", "weight_kg"):
        r = by.get(key)
        if r and r.get("change") is not None and r.get("threshold"):
            return r
    for r in rows:
        if r.get("change") is not None and r.get("threshold"):
            return r
    return None


def _render_hero(rows, result):
    primary = _pick_primary(rows)
    n_real = sum(1 for r in rows if r["verdict"] == VERDICT_REAL)
    n_testable = sum(1 for r in rows if r.get("threshold") is not None)

    tiles = [
        ("测量次数", result["n_points"], ""),
        ("可比点", result["n_clean"], ""),
        ("被降级", len(result["flagged"]), ""),
        ("判定为真", f"{n_real}<small>/ {n_testable}</small>", ""),
    ]
    tiles_html = "".join(
        f"<div class='tile'><div class='t-label'>{_esc(a)}</div>"
        f"<div class='t-value'>{b}{c}</div></div>"
        for a, b, c in tiles
    )

    if primary is None:
        return f"""
  <div class="hero-row">
    <div class="hero">
      <div class="value">—</div>
      <div class="label">可比点太少，做不出任何判定</div>
    </div>
    <div class="tiles">{tiles_html}</div>
  </div>"""

    ratio = abs(primary["change"]) / primary["threshold"] if primary["threshold"] else 0.0
    style, icon, _ = VERDICT_STYLE.get(primary["verdict"], ("mute", "·", ""))
    width = min(1.0, ratio) * 100.0
    fill_cls = "good" if ratio >= 1 else ("warn" if ratio >= 0.65 else "")

    unit = primary["unit"]
    label = primary["label"]

    return f"""
  <div class="hero-row">
    <div class="hero">
      <div class="value">{ratio * 100:.0f}<small>%</small></div>
      <div class="meter" role="img"
           aria-label="{_esc(label)}的变化量走到了判定阈值的 {ratio * 100:.0f}%">
        <div class="fill {fill_cls}" style="width:{width:.1f}%"></div>
      </div>
      <div class="label">
        <b>{_esc(label)}</b> 的变化，只走到了判定阈值的 {ratio * 100:.0f}%<br>
        {_abs(primary['change'], 2)} {unit} vs 门槛 {_plain(primary['threshold'], 2)} {unit}
      </div>
    </div>
    <div class="tiles">{tiles_html}</div>
  </div>"""


# ================================================================ 主视觉
def _render_comparison(rows):
    """每个指标一条横杆，外面套一条固定宽度的灰带。

    ---- 为什么横轴是「倍数」而不是 kg ----

    一开始想的是所有指标共用一条绝对值轴。跑出来立刻发现问题：
    体脂率的阈值有 ±1.0 个百分点，比体重的 ±0.5 kg 大一倍，
    于是它那一条的灰带撑满了整个轨道，把别的指标全压成了小点。

    而且那个做法在原理上就是错的：**kg 和「个百分点」本来就不可比**，
    硬放在一条轴上比较，比的是单位，不是结论。

    真正需要比较的是每个指标**相对它自己的噪声**走了多远。
    所以横轴改成倍数：灰带边缘恒等于 1×，横杆长度就是
    「变化量 ÷ 判定阈值」。这样六行才真的能互相比较 ——
    而且一眼能看出哪些杆子根本没伸出灰带。
    """
    body = []
    n_inside = 0

    for r in rows:
        style, icon, _ = VERDICT_STYLE.get(r["verdict"], ("mute", "·", ""))
        if r.get("change") is None or not r.get("threshold"):
            body.append(
                f"<div class='cmp-row'>"
                f"<span class='cmp-name'>{_esc(r['label'])}</span>"
                f"<span class='cmp-track'><span class='cmp-band' "
                f"style='left:25%;width:50%'></span>"
                f"<span class='cmp-axisline'></span></span>"
                f"<span class='cmp-ratio'>—</span>"
                f"<span class='cmp-val'>—</span>"
                f"<span class='cmp-flag {style}'>{icon} {_esc(r['verdict'])}</span>"
                f"</div>"
            )
            continue

        change, thr = r["change"], r["threshold"]
        ratio = abs(change) / thr
        # 灰带恒占轨道中间一半（25%~75%），边缘就是 1×；
        # 横杆按倍数往外长，2× 封顶（超出的话数字栏里仍然是真的）
        length = min(ratio, 2.0) / 2.0 * 50.0
        side = "pos" if change >= 0 else "neg"
        if r["verdict"] == VERDICT_NOISE:
            n_inside += 1

        body.append(
            f"<div class='cmp-row'>"
            f"<span class='cmp-name'>{_esc(r['label'])}</span>"
            f"<span class='cmp-track'>"
            f"<span class='cmp-band' style='left:25%;width:50%'></span>"
            f"<span class='cmp-axisline'></span>"
            f"<span class='cmp-bar {side} {style}' style='width:{max(length, 0.4):.2f}%'>"
            f"<title>{ratio:.2f} 倍判定阈值</title></span>"
            f"</span>"
            f"<span class='cmp-ratio'>{ratio:.2f}×</span>"
            f"<span class='cmp-val'>{_signed(change, 2, ' ' + r['unit'])}</span>"
            f"<span class='cmp-flag {style}'>{icon} {_esc(r['verdict'])}</span>"
            f"</div>"
        )

    head = (
        "<div class='cmp-row head'>"
        "<span>指标</span>"
        "<span>变化量 ÷ 判定阈值（灰带边缘 = 1×）</span>"
        "<span style='text-align:right'>倍数</span>"
        "<span style='text-align:right'>实际变化</span>"
        "<span>判定</span>"
        "</div>"
    )

    # 标尺：0 在中线，1× 落在灰带两条边上，2× 在轨道两端
    ticks = ""
    for pos, label, anchor in (
        (0.0, "2×", "start"),
        (25.0, "1×", "mid"),
        (50.0, "0", "mid"),
        (75.0, "1×", "mid"),
        (100.0, "2×", "end"),
    ):
        shift = {"start": "0", "mid": "-50%", "end": "-100%"}[anchor]
        ticks += (
            f"<i style='left:{pos}%;transform:translateX({shift})'>{label}</i>"
        )
    ruler = f"<div class='cmp-ruler'><span></span><span class='ticks'>{ticks}</span></div>"

    legend = (
        "<div class='legend'>"
        "<span class='k'><i class='sw band'></i>灰带 = 判定阈值（±1.96×√2×TE），边缘 = 1×</span>"
        "<span class='k'><i class='sw' style='background:var(--good)'></i>越过阈值</span>"
        "<span class='k'><i class='sw' style='background:var(--warning)'></i>只到临界</span>"
        "<span class='k'><i class='sw' style='background:var(--bar-mute)'></i>在噪声内</span>"
        "</div>"
    )

    note = ""
    if n_inside:
        note = (
            f"<p class='sub' style='margin-top:16px;margin-bottom:0'>"
            f"<b>{n_inside} 个指标的横杆完全缩在灰带里。</b>"
            f"它们不是「没变化」，是「这个变化和噪声分不开」—— "
            f"多测几次也不会让它们变清楚，因为噪声来自测量方法本身。</p>"
        )

    return (
        f"<div class='cmp-scroll'><div class='cmp'>{head}{''.join(body)}{ruler}</div></div>"
        f"{legend}{note}"
    )


# ================================================================ 判定表
def _render_verdicts(rows):
    body = []
    for r in rows:
        style, icon, tip = VERDICT_STYLE.get(r["verdict"], ("mute", "·", ""))
        thr = _plain(r.get("threshold"), 2, " " + r["unit"]) if r.get("threshold") else "—"
        p = r.get("p_value")
        q = r.get("q_value")
        extra = ""
        if p is not None:
            extra = f"p = {p:.3f}"
            if q is not None and q != p:
                extra += f"　q = {q:.3f}"
        body.append(
            f"<tr>"
            f"<td class='q'>{_esc(r['label'])}</td>"
            f"<td class='num'>{_signed(r.get('change'), 2)}</td>"
            f"<td class='num'>{thr}</td>"
            f"<td class='num'>{extra or '—'}</td>"
            f"<td><span class='badge {style}'>{icon} {_esc(r['verdict'])}</span></td>"
            f"<td>{_rich(r['reason'])}</td>"
            f"</tr>"
        )
    return (
        "<table class='grid'><thead><tr>"
        "<th style='width:9%'>指标</th>"
        "<th style='width:10%'>变化</th>"
        "<th style='width:12%'>判定阈值</th>"
        "<th style='width:12%'>显著性</th>"
        "<th style='width:12%'>判定</th>"
        "<th>理由（可以自己拿计算器核对）</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


# ================================================================ 噪声地板
def _variogram_svg(curve):
    """变差函数的小图：横轴是间隔天数，纵轴是 σ。

    点本身就带 <title>，鼠标停上去能看到具体数字；
    但首尾两个值**直接写在图上**，不用悬停也能读。

    ---- 为什么四张图各画各的纵轴 ----

    一开始让四张图共用一条纵轴，理由是「小图共用刻度才可比」。
    截图一看不行：体重的 σ 在 0.4 kg 上下，体脂率的 σ 有 2.5 个百分点，
    共用一条轴的话三条曲线全被压在底部一条线上，什么都看不出来。

    更根本的是，**共用轴在这里本来就是错的**：一条轴的单位是 kg，
    另一条是「个百分点」，两个量根本不可比，何况本来也不可比 ——
    它们量的不是同一种东西。所以各画各的，纵轴刻度自己标出来。

    每张图里画一条「最近一档 σ」的参考线。这张图要回答的问题是
    「噪声会不会随间隔增长」，参照点就是最近那一档，不是 0 ——
    曲线贴在参考线上还是爬到它上面，才是要读的东西。
    """
    if len(curve) < 2:
        return "<div class='empty' style='padding:12px 0'>数据点不够，画不出曲线</div>"

    w, h = 300, 96
    pad_l, pad_r, pad_t, pad_b = 38, 16, 16, 24
    xs = [c["gap"] for c in curve]
    ys = [c["sigma"] for c in curve]
    x0, x1 = min(xs), max(xs)
    y0, y1 = 0.0, max(ys) * 1.18
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= 0:
        y1 = 1.0

    def px(g):
        return pad_l + (g - x0) / (x1 - x0) * (w - pad_l - pad_r)

    def py(s):
        return h - pad_b - (s - y0) / (y1 - y0) * (h - pad_t - pad_b)

    parts = []
    # 基线与一条中线刻度：发丝级实线，不用虚线
    for val in (y1, y1 / 2.0):
        y = py(val)
        parts.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{w - pad_r}' y2='{y:.1f}' "
            f"stroke='var(--grid)' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pad_l - 6}' y='{y + 4:.1f}' text-anchor='end' "
            f"font-size='10' fill='var(--muted)'>{val:.2f}</text>"
        )
    parts.append(
        f"<line x1='{pad_l}' y1='{h - pad_b}' x2='{w - pad_r}' y2='{h - pad_b}' "
        f"stroke='var(--axis)' stroke-width='1'/>"
    )

    # 参考线：最近一档的 σ。曲线在它上面还是下面，就是"平不平"的答案
    ref = curve[0]["sigma"]
    parts.append(
        f"<line x1='{pad_l}' y1='{py(ref):.1f}' x2='{w - pad_r}' y2='{py(ref):.1f}' "
        f"stroke='var(--axis)' stroke-width='1'/>"
    )

    pts = " ".join(f"{px(c['gap']):.1f},{py(c['sigma']):.1f}" for c in curve)
    parts.append(
        f"<polyline points='{pts}' fill='none' stroke='var(--series-1)' "
        f"stroke-width='1.7' stroke-linejoin='round' stroke-linecap='round'/>"
    )
    for c in curve:
        parts.append(
            f"<circle cx='{px(c['gap']):.1f}' cy='{py(c['sigma']):.1f}' r='4' "
            f"fill='var(--series-1)' stroke='var(--surface-1)' stroke-width='2'>"
            f"<title>{c['gap']:.0f} 天：σ = {c['sigma']:.3f}（{c['n']} 对）</title>"
            f"</circle>"
        )

    # 首尾直接标值
    first, last = curve[0], curve[-1]
    parts.append(
        f"<text x='{px(first['gap']):.1f}' y='{py(first['sigma']) - 10:.1f}' "
        f"text-anchor='start' font-size='10.5' font-weight='600' "
        f"fill='var(--text-secondary)'>{first['sigma']:.2f}</text>"
    )
    parts.append(
        f"<text x='{px(last['gap']):.1f}' y='{py(last['sigma']) - 10:.1f}' "
        f"text-anchor='end' font-size='10.5' font-weight='600' "
        f"fill='var(--text-secondary)'>{last['sigma']:.2f}</text>"
    )

    # 横轴：只标首尾，中间的间隔自己在 <title> 里看
    parts.append(
        f"<text x='{pad_l}' y='{h - 6}' font-size='10' "
        f"fill='var(--muted)'>{x0:.0f} 天</text>"
    )
    parts.append(
        f"<text x='{w - pad_r}' y='{h - 6}' text-anchor='end' font-size='10' "
        f"fill='var(--muted)'>{x1:.0f} 天</text>"
    )

    return (
        f"<svg viewBox='0 0 {w} {h}' role='img' "
        f"aria-label='变差函数：噪声随间隔天数的变化'>"
        + "".join(parts)
        + "</svg>"
    )


def _drift_sentence(curve):
    """用最近一档和最远一档的比值，说清楚这条曲线平不平。"""
    if len(curve) < 2 or curve[0]["sigma"] <= 0:
        return ""
    ratio = curve[-1]["sigma"] / curve[0]["sigma"]
    far = curve[-1]["gap"]
    if ratio < 1.25:
        return f"噪声基本不随间隔变化（{far:.0f} 天处只是 {ratio:.0%}）—— 用一个 σ 就够"
    if ratio < 2.0:
        return f"噪声随间隔缓慢上升（{far:.0f} 天处到了 {ratio:.0%}）—— 长间隔的点要放宽一点"
    return f"噪声随间隔明显累积（{far:.0f} 天处是 {ratio:.0%}）—— 隔得越远越不可比"


def _render_noise(noise_map, rows):
    cells = []
    dense_rows = []
    labels = {r["metric"]: r["label"] for r in rows}

    for metric, nf in noise_map.items():
        if not nf:
            continue
        name = labels.get(metric, metric)
        src = nf.get("source")
        tag = "从你的数据估出来" if src == "estimated" else "默认值（数据不够）"
        walk = nf.get("random_walk")
        curve = nf.get("curve") or []
        unit = next((r["unit"] for r in rows if r["metric"] == metric), "")
        cap = (
            f"TE {_plain(nf.get('te'), 2)} {unit}　"
            f"LSC {_plain(nf.get('lsc'), 2)} {unit}　{nf.get('n_points', 0)} 点"
        )
        drift = _drift_sentence(curve)
        # 漂移那句话本身就是"要不要当心"的结论，不必再单挂一行警告
        drift_html = (
            f"<div class='c-cap' style='margin-top:2px;color:var(--warning)'>"
            f"{_esc(drift)}</div>"
            if walk and drift
            else f"<div class='c-cap' style='margin-top:2px'>{_esc(drift)}</div>"
        )
        cells.append(
            f"<div class='cell'>"
            f"<div class='c-head'><span class='c-name'>{_esc(name)}</span>"
            f"<span class='badge cut'>{_esc(tag)}</span></div>"
            f"{_variogram_svg(curve)}"
            f"<div class='c-cap'>{cap}</div>{drift_html}"
            f"<div class='c-cap' style='margin-top:2px'>{_rich(nf.get('note', ''))}</div>"
            f"</div>"
        )
        for c in curve:
            dense_rows.append(
                f"<tr><td>{_esc(name)}</td><td>{c['gap']:.0f}</td>"
                f"<td>{c['sigma']:.3f}</td><td>{c['n']}</td></tr>"
            )

    dense = ""
    if dense_rows:
        dense = (
            "<details><summary>看数字（每一档的 σ）</summary><div class='body'>"
            "<table class='dense'><thead><tr>"
            "<th>指标</th><th>间隔（天）</th><th>σ</th><th>点对数</th>"
            "</tr></thead><tbody>" + "".join(dense_rows) + "</tbody></table>"
            "</div></details>"
        )

    legend = (
        "<div class='legend'>"
        "<span class='k'><i class='sw dot'></i>每个点是该间隔档位上估出的 σ</span>"
        "<span class='k'><i class='sw line' style='background:var(--axis)'></i>"
        "参考线 = 最近一档的 σ</span>"
        "<span class='k'>曲线贴着参考线 → 噪声与间隔无关，一天和一周一样可比</span>"
        "<span class='k'>曲线爬到参考线以上 → 存在累积的生物漂移，隔得越远越不可比</span>"
        "</div>"
    )

    # 2 列布局下，奇数个格子会在右下角露出一块网格底色 —— 补一个占位的。
    # 窄屏变成 1 列时它会多出一段空白，所以那个断点里直接隐藏。
    if len(cells) % 2:
        cells.append("<div class='cell filler'></div>")

    return f"<div class='multiples'>{''.join(cells)}</div>{legend}{dense}"


# ================================================================ 时间序列
def _moving_median(points, window=5):
    """按时间排序的点上做居中移动中位数。

    用中位数不用均值：一次生病引起的水分波动会把均值整段拽走，
    中位数不受影响。
    """
    n = len(points)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals = sorted(p[1] for p in points[lo:hi])
        m = len(vals)
        out.append(vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2.0)
    return out


def _series_svg(records, metric, row, te, flagged_idx):
    """一条指标序列：点 + ±TE 噪声带 + 首尾中位数。"""
    pts = [(r["_day"], float(r[metric])) for r in records if r.get(metric) is not None]
    if len(pts) < 4:
        return "<div class='empty' style='padding:10px 0'>可比点太少，画不出来</div>"

    pts.sort(key=lambda p: p[0])
    dated = sorted(
        (r for r in records if r.get(metric) is not None), key=lambda r: r["_day"]
    )
    first_date = dated[0].get("date", "") if dated else ""
    last_date = dated[-1].get("date", "") if dated else ""
    flagged_pts = [
        (r["_day"], float(r[metric]))
        for i, r in enumerate(records)
        if r.get(metric) is not None and i in flagged_idx
    ]
    clean = [(r["_day"], float(r[metric])) for i, r in enumerate(records)
             if r.get(metric) is not None and i not in flagged_idx]
    clean.sort(key=lambda p: p[0])
    if len(clean) < 2:
        return "<div class='empty' style='padding:10px 0'>可比点太少，画不出来</div>"

    med = _moving_median(clean, window=5)
    te_v = abs(te or 0.0)
    band_days = [p[0] for p in clean]
    band_hi = [m + te_v for m in med]
    band_lo = [m - te_v for m in med]

    days = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    allv = vals + band_hi + band_lo
    x0, x1 = min(days), max(days)
    y0, y1 = min(allv), max(allv)
    pad = (y1 - y0) * 0.12 or 1.0
    y0, y1 = y0 - pad, y1 + pad
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0

    w, h = 760, 210
    pl, pr, pt, pb = 52, 14, 14, 26

    def px(d):
        return pl + (d - x0) / (x1 - x0) * (w - pl - pr)

    def py(v):
        return h - pb - (v - y0) / (y1 - y0) * (h - pt - pb)

    parts = []

    # 横向刻度：三条，值直接标出来
    steps = [y0 + (y1 - y0) * f for f in (0.0, 0.5, 1.0)]
    for s in steps:
        y = py(s)
        parts.append(
            f"<line x1='{pl}' y1='{y:.1f}' x2='{w - pr}' y2='{y:.1f}' "
            f"stroke='var(--grid)' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{pl - 8}' y='{y + 4:.1f}' text-anchor='end' font-size='10.5' "
            f"fill='var(--muted)'>{s:.1f}</text>"
        )

    # 噪声带：上界正向描一遍、下界反向描回来，闭合成一条带
    up = " ".join(f"{px(d):.1f},{py(v):.1f}" for d, v in zip(band_days, band_hi))
    dn = " ".join(
        f"{px(d):.1f},{py(v):.1f}"
        for d, v in zip(reversed(band_days), reversed(band_lo))
    )
    parts.append(f"<polygon points='{up} {dn}' fill='var(--band)' stroke='none'/>")

    # 首尾中位数：两段短横线，值直接标上去 —— 这是判定的两个输入
    for key, anchor in (("start", "start"), ("end", "end")):
        e = row.get(key)
        if not e:
            continue
        y = py(e["value"])
        xa, xb = px(e["day"]) - 26, px(e["day"]) + 26
        xa = max(pl, xa)
        xb = min(w - pr, xb)
        parts.append(
            f"<line x1='{xa:.1f}' y1='{y:.1f}' x2='{xb:.1f}' y2='{y:.1f}' "
            f"stroke='var(--series-1)' stroke-width='2' stroke-linecap='round'/>"
        )
        parts.append(
            f"<text x='{xb if anchor == 'start' else xa:.1f}' "
            f"y='{y - 7:.1f}' text-anchor='{anchor}' font-size='11' font-weight='600' "
            f"fill='var(--text-secondary)'>{e['value']:.1f}</text>"
        )

    # 被降级的点：空心圈，画在底下但一样看得见 —— 不删除，只是不参与判定
    for d, v in flagged_pts:
        parts.append(
            f"<circle cx='{px(d):.1f}' cy='{py(v):.1f}' r='3.6' fill='var(--surface-1)' "
            f"stroke='var(--flag-mark)' stroke-width='1.8'>"
            f"<title>被降级：{v:.1f}</title></circle>"
        )

    # 可比点
    for d, v in clean:
        parts.append(
            f"<circle cx='{px(d):.1f}' cy='{py(v):.1f}' r='4' fill='var(--series-1)' "
            f"stroke='var(--surface-1)' stroke-width='2'>"
            f"<title>{v:.1f}</title></circle>"
        )

    # 横轴首尾日期
    parts.append(
        f"<text x='{pl}' y='{h - 7}' font-size='10.5' fill='var(--muted)'>"
        f"{_esc(first_date)}</text>"
    )
    parts.append(
        f"<text x='{w - pr}' y='{h - 7}' text-anchor='end' font-size='10.5' "
        f"fill='var(--muted)'>{_esc(last_date)}</text>"
    )

    return (
        f"<svg viewBox='0 0 {w} {h}' role='img' "
        f"aria-label='{_esc(row['label'])} 随时间的变化，带 ±TE 噪声带'>"
        + "".join(parts)
        + "</svg>"
    )


def _render_series(records, rows, noise_map, flagged_idx):
    blocks = []
    for metric in SERIES_METRICS:
        row = next((r for r in rows if r["metric"] == metric), None)
        if row is None or row.get("change") is None:
            continue
        nf = noise_map.get(metric) or {}
        te = nf.get("te")
        style, icon, _ = VERDICT_STYLE.get(row["verdict"], ("mute", "·", ""))
        note = (
            f"TE {_plain(te, 2)} {row['unit']}　"
            f"带内的一切和噪声分不开"
        )
        blocks.append(
            f"<div class='series-block'>"
            f"<div class='s-head'>"
            f"<span class='s-name'>{_esc(row['label'])}"
            f"　<span class='badge {style}'>{icon} {_esc(row['verdict'])}</span></span>"
            f"<span class='s-note'>{_esc(note)}</span></div>"
            f"{_series_svg(records, metric, row, te, flagged_idx)}"
            f"</div>"
        )

    legend = (
        "<div class='legend'>"
        "<span class='k'><i class='sw dot'></i>可比点</span>"
        "<span class='k'><i class='sw ring'></i>被降级（保留显示，只是不参与判定）</span>"
        "<span class='k'><i class='sw line' style='background:var(--series-1)'></i>首尾中位数（判定的输入）</span>"
        "<span class='k'><i class='sw band'></i>±TE 噪声带</span>"
        "</div>"
    )
    return f"<div class='series-wrap'>{''.join(blocks)}</div>{legend}"


# ================================================================ 降级点
def _render_flagged(flagged):
    if not flagged:
        return "<p class='empty'>这次没有任何测量被降级。</p>"

    rows = []
    for f in flagged[:40]:
        lvl = "不可比" if f["level"] == LEVEL_BAD else "需谨慎"
        cls = "good" if f["level"] == LEVEL_BAD else "warn"
        rows.append(
            f"<tr>"
            f"<td class='num'>{_esc(f.get('date', ''))}</td>"
            f"<td><span class='badge {cls}'>{_esc(lvl)}</span></td>"
            f"<td>{_rich('；'.join(f['reasons']))}</td>"
            f"</tr>"
        )
    more = ""
    if len(flagged) > 40:
        more = f"<p class='empty'>…… 还有 {len(flagged) - 40} 次</p>"

    rule_rows = "".join(
        f"<tr><td class='q'>{_esc(r['name'])}</td><td>{_rich(r['why'])}</td></tr>"
        for r in PROTOCOL_RULES
    )

    return (
        "<table class='grid'><thead><tr>"
        "<th style='width:14%'>日期</th>"
        "<th style='width:12%'>等级</th>"
        "<th>为什么不该拿来比</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>" + more
        + "<details><summary>这些判据是从哪来的（ESPEN 2004 标准化指南 / NIH 1994）</summary>"
        "<div class='body'><table class='grid'><thead><tr>"
        "<th style='width:26%'>判据</th><th>依据</th>"
        "</tr></thead><tbody>" + rule_rows + "</tbody></table>"
        f"<p class='empty' style='margin-top:14px'>"
        f"原则是<b>只降级，不删除</b>：把一次大幅水分波动当异常值删掉，"
        f"等于删掉了最该被解释的那个事件（生病、高钠、经期）。"
        f"所以这里只把它标出来，判定时不采用，但报告里一直显示着。</p>"
        "</div></details>"
    )


# ================================================================ 该忽略的
def _render_ignores(ignored):
    if not ignored:
        return "<p class='empty'>没有可列出的项。</p>"

    out = []
    for ig in ignored:
        tag = "你的数据里有这一项" if ig.get("present") else "你的数据里没有这一项"
        out.append(
            f"<div class='ignore'>"
            f"<div class='n'>{_esc(ig['label'])}<span class='tag'>{_esc(tag)}</span></div>"
            f"<div class='w'>{_rich(ig['why'])}</div>"
            f"</div>"
        )
    return f"<div class='ignores'>{''.join(out)}</div>"


# ================================================================ 方法说明
def _render_method(rows, result):
    hw_rows = "".join(
        f"<tr><td class='q'>{_esc(k)}</td>"
        f"<td class='num'>±{v['half_width']}</td>"
        f"<td>{_rich(v['basis'])}</td>"
        f"<td class='mono'>{_esc(v['source'])}</td></tr>"
        for k, v in METHOD_UNCERTAINTY.items()
    )

    items = [
        ("我不声称我的数字准。",
         "本工具只对<b>变化</b>做推理，不声称任何单次读数准确。"
         "报告里从不出现「你的体脂率是 X%」这种句子。"),
        ("阈值用的是日间典型误差，不是方法误差。",
         "稳定的方法偏差在 <code>值(t2) − 值(t1)</code> 里会被减掉两次，自动抵消。"
         "拿 ±3 个百分点去卡变化，等于把同一个东西算了两遍。"),
        ("门控只降级，不删除。",
         "被标记的测量仍然显示在图上和表里。删掉它们，等于删掉了最该被解释的那次波动。"),
        ("可比点少于 8 个就不估噪声。",
         "改用一张默认值表，并在卡片上标明「默认值（数据不够）」，"
         "不让人误以为那个数字是从自己数据里算出来的。"),
        ("跨设备的数据直接拒绝。",
         "足部秤的回归公式按设备定制，换一台秤就换了一套坐标系。"),
        ("不做诊断，不给补剂或饮食建议。",
         "这是刻意的边界。自动化地告诉你「该补什么维生素」没有可靠依据，做出来是错的。"),
    ]
    why = "".join(
        f"<div class='why-item'><span class='m'>▸</span>"
        f"<span class='t'><b>{_esc(a)}</b> {b}</span></div>"
        for a, b in items
    )

    alpha = result.get("alpha")
    corr = ""
    if alpha is not None:
        corr = (
            f"<p class='empty' style='margin-top:16px'>"
            f"多重比较校正（Benjamini-Hochberg，α = {alpha}）："
            f"同时检验 {sum(1 for r in rows if r.get('p_value') is not None)} 个指标，"
            f"不校正的话，每 20 次判定就会撞上一次假阳性。"
            f"本项目的立论是「别再给我一堆数字」，"
            f"而判定模块自己就制造了一堆假设检验 —— "
            f"所以这一步必须显式做出来，不能藏起来。</p>"
        )

    return (
        f"<div class='why-list'>{why}</div>{corr}"
        "<details><summary>方法不确定度表（只用于绝对值，不用于判定变化）</summary>"
        "<div class='body'><table class='grid'><thead><tr>"
        "<th style='width:20%'>指标</th><th style='width:10%'>半宽</th>"
        "<th>依据</th><th style='width:16%'>来源</th>"
        "</tr></thead><tbody>" + hw_rows + "</tbody></table>"
        "<p class='empty' style='margin-top:14px'>"
        "这些是<b>系统性</b>误差：多测几次不会让它变小。"
        "所以它们管的是「你那个 24.5% 到底是不是 24.5%」，"
        "管不了「你是不是减了 0.9 公斤」—— 后者由上面那套日间误差负责。</p>"
        "</div></details>"
    )


# ================================================================ 组装
def _card(idx, heading, sub, body):
    num = f"<span class='idx'>{idx:02d}</span>" if idx else ""
    return (
        f"<div class='card'><h2>{num}{_esc(heading)}</h2>"
        f"<p class='sub'>{sub}</p>{body}</div>"
    )


def build_html(result, title="fitxray 体成分变化归因报告"):
    """生成完整 HTML 文本。result 是 verdict.verdicts() 的返回值。"""
    rows = result["metrics"]
    records = result.get("records") or []
    noise_map = result.get("noise") or {}
    flagged = result.get("flagged") or []
    flagged_idx = {f["index"] for f in flagged}

    chips = "".join(
        f"<span class='chip'>{k}<b>{v}</b></span>"
        for k, v in [
            ("设备", _esc(result.get("device_id") or "未标注")),
            ("测量", f"{result['n_points']} 次"),
            ("可比点", f"{result['n_clean']} 个"),
            ("降级", f"{len(flagged)} 次"),
            ("FDR α", result.get("alpha")),
        ]
    )

    cards = []
    n = 0

    n += 1
    cards.append(
        _card(
            n,
            "这次到底变了没有",
            "最大的那个数字是「变化量 ÷ 判定阈值」。"
            "<b>低于 100% 就意味着：这个变化和噪声分不开。</b>",
            _render_hero(rows, result),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "每个指标的变化，够不够穿过噪声",
            "横杆是从起点到终点的变化量，灰带是这个指标自己的判定阈值。"
            "<b>横杆没伸出灰带，就说明这个变化还读不出来。</b>",
            _render_comparison(rows),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "逐指标判定",
            "每一条都配了一句可以自己拿计算器核对的理由，"
            "不是一句「差异不显著」。",
            _render_verdicts(rows),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "噪声地板：从你自己的数据估出来",
            "没有写死的阈值表。每条指标都从<b>你自己的测量序列</b>里估一个典型误差，"
            "再换算成最小显著变化（LSC）。数据不够时才退回落回默认值，并明确标注。",
            _render_noise(noise_map, rows),
        )
    )

    if records:
        n += 1
        cards.append(
            _card(
                n,
                "原始序列与噪声带",
                "点落在灰带里的一切起伏，都还读不出生物学意义。"
                "首尾那两条短横线就是判定的两个输入。",
                _render_series(records, rows, noise_map, flagged_idx),
            )
        )

    n += 1
    cards.append(
        _card(
            n,
            f"被降级的测量（{len(flagged)} 次）",
            "这些测量没有参与判定，但也没有被删掉 —— "
            "它们往往正是最该被解释的那几次。",
            _render_flagged(flagged),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "这几项不用看",
            "体脂秤会给你二十个数字，其中一部分是别的数字套公式算出来的，"
            "<b>没有任何独立信息</b>。别的工具把它们当卖点展示，这里直接告诉你别看。",
            _render_ignores(result.get("ignored") or []),
        )
    )

    n += 1
    cards.append(
        _card(
            n,
            "方法、边界与已知限制",
            "一份诚实的报告要先说清楚自己不能做什么。",
            _render_method(rows, result),
        )
    )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_css()}</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <button class="theme-toggle" id="theme-toggle" type="button"
            aria-label="切换深色 / 浅色模式" title="切换深色 / 浅色模式">◐</button>
    <div class="wordmark">fitxray</div>
    <h1>{_esc(title)}</h1>
    <p class="lede">体重掉了几公斤，其中有多少是真的？这个工具不告诉你数字准不准，
       它追问的是另一个问题：<b>这个变化，够不够大到一个体脂秤能看出来。</b>
       先估噪声，再做门控，最后才谈结论。</p>
    <div class="chips">{chips}</div>
  </header>

{''.join(cards)}

  <p class="foot">由 fitxray 生成 · {stamp} · 不构成任何医疗建议</p>
</div>

<script>
(function () {{
  var root = document.documentElement;
  try {{
    var saved = localStorage.getItem('fitxray-theme');
    if (saved) root.setAttribute('data-theme', saved);
  }} catch (e) {{}}

  document.getElementById('theme-toggle').addEventListener('click', function () {{
    var cur = root.getAttribute('data-theme');
    if (!cur) {{
      cur = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }}
    var next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try {{ localStorage.setItem('fitxray-theme', next); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""


def write_html(path, result, title="fitxray 体成分变化归因报告"):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(result, title))
    return path
