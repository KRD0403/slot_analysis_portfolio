import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# =========================
# ■ 日本語対応
# =========================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Hiragino Sans',
    'Yu Gothic',
    'Meirio',
    'IPAexGothic',
    'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# =========================
# ■ データ読み込み
# =========================
df = pd.read_csv("../data/data.csv")

# カラム名統一
df.rename(columns={
    "推定設定": "setting",
    "1Gあたり利益": "profit_per_game",
    "稼働率": "utilization"
}, inplace=True)

# イベント名
df["event_name"] = df["event_frag"].map({
    0: "通常",
    1: "特定日",
    2: "ファン感"
})

# =========================
# ■ 設定別分析
# =========================
setting_grouped = df.groupby("setting")[["utilization", "profit_per_game"]].mean().sort_index()
x = range(len(setting_grouped.index))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(x, setting_grouped["utilization"])
axes[0].set_title("設定別 平均稼働率")
axes[0].set_xticks(x)
axes[0].set_xticklabels(setting_grouped.index)

for i, v in enumerate(setting_grouped["utilization"]):
    axes[0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

axes[1].bar(x, setting_grouped["profit_per_game"])
axes[1].set_title("設定別 平均1G利益")
axes[1].set_xticks(x)
axes[1].set_xticklabels(setting_grouped.index)

for i, v in enumerate(setting_grouped["profit_per_game"]):
    axes[1].text(i, v, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("../images/bar_chart_setting.png")
plt.show()

# =========================
# ■ 店舗別分析
# =========================
store_grouped = df.groupby("store")[["utilization", "profit_per_game"]].mean()
x = range(len(store_grouped.index))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(x, store_grouped["utilization"])
axes[0].set_title("店舗別 平均稼働率")
axes[0].set_xticks(x)
axes[0].set_xticklabels(store_grouped.index)

for i, v in enumerate(store_grouped["utilization"]):
    axes[0].text(i, v, f"{v:.2f}", ha='center', va='bottom')

axes[1].bar(x, store_grouped["profit_per_game"])
axes[1].set_title("店舗別 平均1G利益")
axes[1].set_xticks(x)
axes[1].set_xticklabels(store_grouped.index)

for i, v in enumerate(store_grouped["profit_per_game"]):
    axes[1].text(i, v, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("../images/bar_chart_store.png")
plt.show()

# =========================
# ■ イベント①：設定配分
# =========================
dist = df.groupby(["store", "event_name", "setting"]).size().reset_index(name="count")
dist["ratio"] = dist.groupby(["store", "event_name"])["count"].transform(lambda x: x / x.sum())

pivot_dist = dist.pivot_table(
    index=["store", "event_name"],
    columns="setting",
    values="ratio",
    fill_value=0
)

pivot_dist.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("店舗×イベント別 設定配分")
plt.xlabel("店舗・イベント")
plt.ylabel("割合")
plt.legend(title="設定", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig("../images/setting_distribution.png")
plt.show()

# =========================
# ■ イベント②：店舗ごとの稼働比較（通常 vs イベント）
# =========================

event_util = df.groupby(["store", "event_name"])["utilization"].mean().unstack()

x = range(len(event_util.index))
width = 0.25

plt.figure(figsize=(10,6))

plt.bar([i - width for i in x], event_util["通常"], width=width)
plt.bar(x, event_util["特定日"], width=width)
plt.bar([i + width for i in x], event_util["ファン感"], width=width)

plt.xticks(x, event_util.index)
plt.title("店舗別 イベントごとの稼働率比較")
plt.xlabel("店舗")
plt.ylabel("稼働率")

plt.legend(["通常", "特定日", "ファン感"])

plt.tight_layout()
plt.savefig("../images/event_utilization_compare.png")
plt.show()