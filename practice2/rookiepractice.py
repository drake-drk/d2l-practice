# -*- coding: utf-8 -*-
"""
practice2 / rookiepractice.py
第一课的练习与答案：预测、基线，以及"你在优化什么"。

运行：
    cd ~/DeepLearning/practice2
    python3 rookiepractice.py

三个练习：
    练习 1  同一个"猜一个常数"，三种猜法，比较 RMSLE 和 RMSE
    练习 2  删掉最长的一趟行程，看两个指标各自的反应
    练习 3  把"按小时取平均"做成模型，换一种组内统计量
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- 0. 读数据 ----------------
# 数据可能和脚本同级，也可能在上一级，两个位置都找一下
HERE = Path(__file__).resolve().parent
CSV = next((p for p in (HERE / "train.csv", HERE.parent / "train.csv") if p.exists()), None)
if CSV is None:
    raise SystemExit("找不到 train.csv，请把脚本放在数据旁边再运行。")

df = pd.read_csv(CSV, usecols=["pickup_datetime", "trip_duration"], parse_dates=["pickup_datetime"])
y = df["trip_duration"].to_numpy(dtype=float)      # 真值，单位：秒
log_y = np.log1p(y)                                 # 取对数后的真值


def rmsle(pred):
    """RMSLE：先把真值和预测都取 log1p，再算均方根误差。"""
    return float(np.sqrt(np.mean((log_y - np.log1p(pred)) ** 2)))


def rmse(pred):
    """RMSE：直接在"秒"上算均方根误差，会被极端值放大。"""
    return float(np.sqrt(np.mean((y - pred) ** 2)))


print("=" * 64)
print(f"样本量 {len(y):,} 条，最长一趟 {y.max() / 3600:.1f} 小时")
print("=" * 64)

# ---------------- 练习 1 ----------------
# 题目：三种"猜一个常数"的方式，分别算 RMSLE 和 RMSE；
#       如果改用 RMSE 当评分标准，哪个常数会赢？为什么？
print("\n[练习 1] 同一个常数，三种猜法")
print(f"{'方式':<10}{'预测值/s':>12}{'RMSLE':>10}{'RMSE/s':>12}")

candidates = {
    "mean": float(y.mean()),
    "median": float(np.median(y)),
    "log-mean": float(np.expm1(log_y.mean())),
}
for name, pred in candidates.items():
    print(f"{name:<10}{pred:>12.2f}{rmsle(pred):>10.4f}{rmse(pred):>12.2f}")

print("""
答案：
  1) RMSLE 下 log-mean 最好（0.7958），mean 最差（0.8908）。
     因为 RMSLE 是在 log 空间量误差，log 空间的最优常数是"对数的平均"，
     换算回来就是 642.54 秒 —— 不是算术平均 959.49 秒。
  2) 换成 RMSE，赢家变成 mean（5237.43 < 5245.87 < 5247.01）。
     RMSE 在"秒"上量误差，平方误差的最优常数正是算术平均。
  3) 同一份数据、同一批预测，只因为评分标准不同，"最优答案"就换人了。
     这就是第一课那句话：评分标准决定最优答案。
""")

# ---------------- 练习 2 ----------------
# 题目：删掉最长的那趟 979 小时行程，RMSLE 和 RMSE 各下降多少？为什么差这么多？
print("\n[练习 2] 删掉最长的一趟行程")
keep = np.ones(len(y), dtype=bool)
keep[int(np.argmax(y))] = False
y2, log_y2 = y[keep], log_y[keep]

print(f"{'方式':<10}{'RMSLE 前':>10}{'RMSLE 后':>10}{'变化':>9}"
      f"{'RMSE 前':>11}{'RMSE 后':>11}{'变化':>10}")
for name, pred in candidates.items():
    after = float(np.expm1(np.log1p(y2).mean())) if name == "log-mean" else (
        float(y2.mean()) if name == "mean" else float(np.median(y2)))
    r1 = float(np.sqrt(np.mean((log_y - np.log1p(pred)) ** 2)))
    r2 = float(np.sqrt(np.mean((log_y2 - np.log1p(after)) ** 2)))
    m1 = float(np.sqrt(np.mean((y - pred) ** 2)))
    m2 = float(np.sqrt(np.mean((y2 - after) ** 2)))
    print(f"{name:<10}{r1:>10.4f}{r2:>10.4f}{r2 - r1:>+9.4f}"
          f"{m1:>11.1f}{m2:>11.1f}{m2 - m1:>+10.1f}")

print("""
答案：
  RMSE 一口气掉了约 888 秒（-17%），RMSLE 几乎不动（变化 <0.002）。
  原因：一条 979 小时的记录在城市数据里几乎不可能，但它离均值有几十万秒，
  平方之后就成了 RMSE 里的巨无霸；而 log 把"差了 100 倍"压成一个小数字，
  所以 RMSLE 对这条记录几乎无感。
  这正是竞赛选 RMSLE 的理由：不让排行榜被几条疯记录决定。
""")

# ---------------- 练习 3（进阶） ----------------
# 题目：按"小时"分组、每组猜该组的平均时长，这个模型能打败常数基线吗？
#       如果用对数平均代替算术平均，结果会怎样？
print("\n[练习 3] 按小时分组：两种组内统计量")
train = df.sample(frac=0.7, random_state=42)
test = df.drop(train.index)

per_hour_mean = train.groupby(train["pickup_datetime"].dt.hour)["trip_duration"].mean()
per_hour_logmean = train.groupby(train["pickup_datetime"].dt.hour)["trip_duration"].apply(
    lambda s: float(np.expm1(np.log1p(s).mean()))
)

test_hour = test["pickup_datetime"].dt.hour
test_log = np.log1p(test["trip_duration"].to_numpy(dtype=float))
constant = float(np.expm1(np.log1p(train["trip_duration"]).mean()))


def score(pred):
    return float(np.sqrt(np.mean((test_log - np.log1p(pred.to_numpy(dtype=float))) ** 2)))


print(f"常数基线（log 均值）            RMSLE = {score(pd.Series(np.full(len(test), constant))):.4f}")
print(f"按小时 + 组内算术平均           RMSLE = {score(test_hour.map(per_hour_mean)):.4f}")
print(f"按小时 + 组内对数平均           RMSLE = {score(test_hour.map(per_hour_logmean)):.4f}")

print("""
答案：
  组内算术平均：0.8859 —— 比常数基线（0.7979）还差，模型"翻车"了。
  组内对数平均：0.7936 —— 反过来超过了基线。
  同一份数据、同一个"按小时"的模型，只换了组内统计量，
  就从"不如基线"变成"超过基线"。
  结论：你用来训练的统计量，必须和评估用的指标对齐；
        复杂模型不保证更好，没有基线你永远不知道自己站在哪。
""")

print("=" * 64)
print("三句话带走：")
print("  1. 先做基线，再谈模型。")
print("  2. 评分标准决定最优答案（log 空间 → log 均值）。")
print("  3. 优化的量必须和评估的量一致。")
print("=" * 64)