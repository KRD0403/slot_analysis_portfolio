import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import itertools
import os
import numpy as np

# =========================
# ■ パス設定
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "..", "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# =========================
# ■ 日本語対応
# =========================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'IPAexGothic', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# ■ データ読み込み
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    df.columns = df.columns.str.replace(" ", "").str.strip()

    df.rename(columns={
        "推定設定": "setting",
        "稼働率": "utilization",
        "利益(円)": "profit",
        "1Gあたり利益": "profit_per_game"
    }, inplace=True)

    # 異常値除外
    df = df[df["データ異常"] == 0]

    # profit_per_game保証
    if "profit_per_game" not in df.columns:
        df["profit_per_game"] = df["profit"] / df["spins"]

    df["daily_profit"] = df["profit"]

    df["event_name"] = df["event_frag"].map({
        0: "通常",
        1: "特定日",
        2: "ファン感"
    })

    print("データ件数:", len(df))
    return df


# =========================
# ■ 設定別分析
# =========================
def analyze_setting(df):
    grouped = df.groupby("setting")[["utilization", "profit_per_game", "daily_profit"]].mean().sort_index()

    print("\n■ 設定別平均")
    print(grouped)

    return grouped


def plot_setting(grouped):
    x = range(len(grouped.index))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cols = ["utilization", "profit_per_game", "daily_profit"]
    titles = ["設定別 平均稼働率", "設定別 平均1G利益", "設定別 平均1日利益"]
    ylabels = ["稼働率", "1Gあたり利益（円）", "1日利益（円）"]

    for ax, col, title, ylabel in zip(axes, cols, titles, ylabels):
        bars = ax.bar(x, grouped[col])
        ax.set_title(title)
        ax.set_xlabel("推定設定")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        for bar, val in zip(bars, grouped[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_1.png"))
    plt.show()


# =========================
# ■ 店舗別分析
# =========================
def plot_store(df):
    grouped = df.groupby("store")[["utilization", "profit_per_game"]].mean()

    x = range(len(grouped.index))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cols = ["utilization", "profit_per_game"]
    titles = ["店舗別 平均稼働率", "店舗別 平均1G利益"]
    ylabels = ["稼働率", "1Gあたり利益（円）"]

    for ax, col, title, ylabel in zip(axes, cols, titles, ylabels):
        bars = ax.bar(x, grouped[col])
        ax.set_title(title)
        ax.set_xlabel("店舗")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)
        for bar, val in zip(bars, grouped[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_2.png"))
    plt.show()


# =========================
# ■ イベント分析
# =========================
def plot_event_distribution(df):
    dist = df.groupby(["store", "event_name", "setting"]).size().reset_index(name="count")
    dist["ratio"] = dist.groupby(["store", "event_name"])["count"].transform(lambda x: x / x.sum())

    pivot = dist.pivot_table(
        index=["store", "event_name"],
        columns="setting",
        values="ratio",
        fill_value=0
    )

    ax = pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("店舗×イベント別 設定配分")
    ax.set_xlabel("店舗 / イベント")
    ax.set_ylabel("設定比率")
    ax.legend(title="推定設定", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_3.png"))
    plt.show()


def plot_event_utilization(df):
    grouped = df.groupby(["store", "event_name"])["utilization"].mean().unstack()

    x = range(len(grouped.index))
    width = 0.25

    event_labels = ["通常", "特定日", "ファン感"]
    colors = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (label, color) in enumerate(zip(event_labels, colors)):
        if label in grouped.columns:
            offset = (i - 1) * width
            bars = ax.bar([xi + offset for xi in x], grouped[label], width,
                          label=label, color=color)
            for bar, val in zip(bars, grouped[label]):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index)
    ax.set_title("店舗別 イベント別稼働率")
    ax.set_xlabel("店舗")
    ax.set_ylabel("稼働率")
    ax.legend(title="イベント種別")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_4.png"))
    plt.show()


# =========================
# ■ 相関分析（Figure_5）
# =========================
def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(8, 6))

    settings = sorted(df["setting"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(settings)))

    for setting, color in zip(settings, colors):
        subset = df[df["setting"] == setting]
        ax.scatter(subset["utilization"], subset["profit_per_game"],
                   label=f"設定{setting}", color=color, alpha=0.6, s=40)

    # 相関係数
    corr = df["utilization"].corr(df["profit_per_game"])

    # 回帰直線
    z = np.polyfit(df["utilization"], df["profit_per_game"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["utilization"].min(), df["utilization"].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1.5, label=f"回帰直線 (r={corr:.2f})")

    ax.set_title("稼働率 vs 1Gあたり利益（相関分析）")
    ax.set_xlabel("稼働率")
    ax.set_ylabel("1Gあたり利益（円）")
    ax.legend(title="推定設定", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_5.png"))
    plt.show()

    print(f"\n■ 稼働率 vs 1G利益 相関係数: {corr:.3f}")


# =========================
# ■ 分散分析 + 設定信頼度（Figure_6）
# =========================
def plot_variance_and_confidence(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    settings = sorted(df["setting"].unique())
    x = range(len(settings))

    # 稼働率の標準偏差
    std_util = df.groupby("setting")["utilization"].std()
    bars = axes[0].bar(x, std_util)
    axes[0].set_title("設定別 稼働率の標準偏差")
    axes[0].set_xlabel("推定設定")
    axes[0].set_ylabel("標準偏差")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(settings)
    for bar, val in zip(bars, std_util):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    # 1G利益の標準偏差
    std_ppg = df.groupby("setting")["profit_per_game"].std()
    bars = axes[1].bar(x, std_ppg)
    axes[1].set_title("設定別 1Gあたり利益の標準偏差")
    axes[1].set_xlabel("推定設定")
    axes[1].set_ylabel("標準偏差（円）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(settings)
    for bar, val in zip(bars, std_ppg):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}", ha='center', va='bottom', fontsize=9)

    # 設定別平均信頼度
    mean_conf = df.groupby("setting")["設定信頼度"].mean()
    bars = axes[2].bar(x, mean_conf, color="darkorange")
    axes[2].set_title("設定別 平均推定信頼度")
    axes[2].set_xlabel("推定設定")
    axes[2].set_ylabel("平均信頼度")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(settings)
    for bar, val in zip(bars, mean_conf):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "Figure_6.png"))
    plt.show()

    print("\n■ 設定別平均信頼度")
    print(mean_conf)


# =========================
# ■ 最適配分（全探索）
# =========================
def optimize_with_rate(df):
    df["payout_rate"] = df["払出枚数(OUT)"] / df["投入枚数(IN)"]

    df["payout_rate_adj"] = df["payout_rate"]
    df.loc[df["setting"] == 5, "payout_rate_adj"] *= 1.1
    df.loc[df["setting"] == 6, "payout_rate_adj"] *= 1.2

    stats = df.groupby("setting")[["payout_rate_adj", "utilization"]].mean()

    print("\n■ 設定別（補正後割数）")
    print(stats)

    targets = {
        "利益最大": 0.35,
        "稼働重視": 0.40,
        "バランス": 0.45,
        "イベント": 0.50
    }

    def calc_rate(alloc):
        return sum(stats.loc[s, "payout_rate_adj"] * c for s, c in alloc.items()) / 10

    def calc_util(alloc):
        return sum(stats.loc[s, "utilization"] * c for s, c in alloc.items()) / 10

    def format_alloc(alloc):
        return " / ".join([f"設定{s}：{c}台" for s, c in alloc.items() if c > 0])

    for name, target in targets.items():

        best_alloc = None
        best_score = -999

        for alloc in itertools.product(range(11), repeat=6):
            if sum(alloc) != 10:
                continue

            alloc_dict = {i+1: alloc[i] for i in range(6)}

            # 実務的制約
            # 設定6は最大3台まで
            if alloc_dict[6] > 3:
                continue
            # 設定の種類を最低3種類以上使用
            if sum(1 for c in alloc_dict.values() if c > 0) < 3:
                continue
            # 低設定（1〜3）は最低2台以上
            if sum(alloc_dict[s] for s in [1, 2, 3]) < 2:
                continue

            rate = calc_rate(alloc_dict)
            util = calc_util(alloc_dict)

            diff = abs(rate - target)
            score = -diff + util * 0.3

            if "イベント" in name:
                high_bonus = alloc_dict[5] * 0.1 + alloc_dict[6] * 0.2
                score += high_bonus

            if score > best_score:
                best_score = score
                best_alloc = alloc_dict

        print(f"\n{name}")
        print("設定配分:", format_alloc(best_alloc))
        print(f"割数: {calc_rate(best_alloc):.3f}")
        print(f"稼働率: {calc_util(best_alloc):.3f}")


def optimize(df):
    stats = df.groupby("setting")[["utilization", "profit", "profit_per_game"]].mean()

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    stats["util_norm"] = normalize(stats["utilization"])
    stats["profit_norm"] = normalize(stats["profit"])
    stats["ppg_norm"] = normalize(stats["profit_per_game"])

    stats["balance"] = stats["profit_norm"] * 0.5 + stats["util_norm"] * 0.5
    stats["customer"] = 1 - stats["ppg_norm"]

    best_profit = best_util = best_balance = best_event = None
    best_profit_score = best_util_score = best_balance_score = best_event_score = -1

    for alloc in itertools.product(range(11), repeat=6):
        if sum(alloc) != 10:
            continue

        alloc_dict = {i+1: alloc[i] for i in range(6)}

        # 実務的制約
        # 設定6は最大3台まで
        if alloc_dict[6] > 3:
            continue
        # 設定の種類を最低3種類以上使用
        if sum(1 for c in alloc_dict.values() if c > 0) < 3:
            continue
        # 低設定（1〜3）は最低2台以上
        if sum(alloc_dict[s] for s in [1, 2, 3]) < 2:
            continue

        util = sum(stats.loc[s, "utilization"] * c for s, c in alloc_dict.items()) / 10
        profit = sum(stats.loc[s, "profit"] * c for s, c in alloc_dict.items())
        balance = sum(stats.loc[s, "balance"] * c for s, c in alloc_dict.items())
        event = sum((stats.loc[s, "util_norm"] + stats.loc[s, "customer"]) * c for s, c in alloc_dict.items())

        if profit > best_profit_score:
            best_profit_score = profit
            best_profit = alloc_dict

        if util > best_util_score:
            best_util_score = util
            best_util = alloc_dict

        if balance > best_balance_score:
            best_balance_score = balance
            best_balance = alloc_dict

        if event > best_event_score:
            best_event_score = event
            best_event = alloc_dict

    def show(name, alloc):
        util = sum(stats.loc[s, "utilization"] * c for s, c in alloc.items()) / 10
        profit = sum(stats.loc[s, "profit"] * c for s, c in alloc.items())
        text = " / ".join([f"設定{s}：{c}台" for s, c in alloc.items() if c > 0])
        print(f"\n{name}")
        print("設定配分:", text)
        print(f"稼働率: {util:.3f}")
        print(f"利益: {int(profit):,}円")

    print("\n■ 最適配分（全探索）")
    show("利益最大", best_profit)
    show("稼働最大", best_util)
    show("バランス最適", best_balance)
    show("イベント最適", best_event)


# =========================
# ■ 実行
# =========================
if __name__ == "__main__":
    print("実行開始")

    df = load_data()

    setting_data = analyze_setting(df)
    plot_setting(setting_data)

    plot_store(df)
    plot_event_distribution(df)
    plot_event_utilization(df)

    plot_correlation(df)
    plot_variance_and_confidence(df)

    optimize(df)
    optimize_with_rate(df)

    print("実行完了")