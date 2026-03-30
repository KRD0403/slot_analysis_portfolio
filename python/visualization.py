import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import itertools

# =========================
# ■ 日本語対応
# =========================
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = [
    'Hiragino Sans', 'Yu Gothic', 'Meirio', 'IPAexGothic', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# ■ データ読み込み
# =========================
def load_data():
    df = pd.read_csv("../data/data.csv")

    df.columns = df.columns.str.replace(" ", "").str.strip()

    df.rename(columns={
        "推定設定": "setting",
        "稼働率": "utilization",
        "利益(円)": "profit"
    }, inplace=True)

    # 異常値除外
    df = df[df["データ異常"] == 0]

    # profit_per_game保証
    if "profit_per_game" not in df.columns:
        if "1Gあたり利益" in df.columns:
            df["profit_per_game"] = df["1Gあたり利益"]
        else:
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

    axes[0].bar(x, grouped["utilization"])
    axes[0].set_title("設定別 平均稼働率")

    axes[1].bar(x, grouped["profit_per_game"])
    axes[1].set_title("設定別 平均1G利益")

    axes[2].bar(x, grouped["daily_profit"])
    axes[2].set_title("設定別 平均1日利益")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)

    plt.tight_layout()
    plt.savefig("../images/Figure_1.png")
    plt.show()


# =========================
# ■ 店舗別分析
# =========================
def plot_store(df):
    grouped = df.groupby("store")[["utilization", "profit_per_game"]].mean()

    x = range(len(grouped.index))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, grouped["utilization"])
    axes[0].set_title("店舗別 平均稼働率")

    axes[1].bar(x, grouped["profit_per_game"])
    axes[1].set_title("店舗別 平均1G利益")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(grouped.index)

    plt.tight_layout()
    plt.savefig("../images/Figure_2.png")
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

    pivot.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("店舗×イベント別 設定配分")
    plt.tight_layout()
    plt.savefig("../images/Figure_3.png")
    plt.show()


def plot_event_utilization(df):
    grouped = df.groupby(["store", "event_name"])["utilization"].mean().unstack()

    x = range(len(grouped.index))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar([i - width for i in x], grouped["通常"], width)
    plt.bar(x, grouped["特定日"], width)
    plt.bar([i + width for i in x], grouped["ファン感"], width)

    plt.xticks(x, grouped.index)
    plt.title("店舗別 イベント別稼働率")

    plt.tight_layout()
    plt.savefig("../images/Figure_4.png")
    plt.show()


# =========================
# ■ 最適配分（全探索）
# =========================
def optimize_with_rate(df):

    import itertools

    df["payout_rate"] = df["払出枚数(OUT)"] / df["投入枚数(IN)"]

    # ★疑似還元
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

            # ★制約
            if alloc_dict[6] > 3:
                continue

            rate = calc_rate(alloc_dict)
            util = calc_util(alloc_dict)

            diff = abs(rate - target)

            score = -diff + util * 0.3

            # ★イベント専用ロジック追加
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

    # 正規化
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())

    stats["util_norm"] = normalize(stats["utilization"])
    stats["profit_norm"] = normalize(stats["profit"])
    stats["ppg_norm"] = normalize(stats["profit_per_game"])

    stats["balance"] = stats["profit_norm"] * 0.5 + stats["util_norm"] * 0.5
    stats["customer"] = 1 - stats["ppg_norm"]

    best_profit = None
    best_util = None
    best_balance = None
    best_event = None

    best_profit_score = -1
    best_util_score = -1
    best_balance_score = -1
    best_event_score = -1

    # 全探索（6^10じゃなく、10台の配分）
    for alloc in itertools.product(range(11), repeat=6):
        if sum(alloc) != 10:
            continue

        alloc_dict = {i+1: alloc[i] for i in range(6)}

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

    optimize(df)
    optimize_with_rate(df)

    print("実行完了")