# ============================================================
# visualization.py
#
# 서울시 장애인콜택시 / UD택시
# 수요예측 및 동적 재배치 정책 시각화
#
# 사용 데이터
# 1. demand_forecast_hourly_v2_2025.csv
# 2. demand_forecast_test_v2_2025.csv
# 3. allocation_summary_v5_2025.csv
# 4. allocation_by_hour_v5_2025.csv
# 5. allocation_by_day_v5_2025.csv
# 6. allocation_routes_v5_2025.csv
# 7. allocation_time_limit_v5_2025.csv
# 8. allocation_efficiency_v5_2025.csv
#
# 출력
# results/figures/*.png
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 파일 경로
# ------------------------------------------------------------

HOURLY_FORECAST_FILE = (
    RESULT_DIR /
    "demand_forecast_hourly_v2_2025.csv"
)

FORECAST_TEST_FILE = (
    RESULT_DIR /
    "demand_forecast_test_v2_2025.csv"
)

SUMMARY_FILE = (
    RESULT_DIR /
    "allocation_summary_v5_2025.csv"
)

BY_HOUR_FILE = (
    RESULT_DIR /
    "allocation_by_hour_v5_2025.csv"
)

BY_DAY_FILE = (
    RESULT_DIR /
    "allocation_by_day_v5_2025.csv"
)

ROUTES_FILE = (
    RESULT_DIR /
    "allocation_routes_v5_2025.csv"
)

TIME_LIMIT_FILE = (
    RESULT_DIR /
    "allocation_time_limit_v5_2025.csv"
)

EFFICIENCY_FILE = (
    RESULT_DIR /
    "allocation_efficiency_v5_2025.csv"
)


# ============================================================
# 1. 한글 폰트 설정
# ============================================================

def setup_korean_font():

    candidates = [
        "AppleGothic",
        "Arial Unicode MS",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR"
    ]

    available_fonts = {
        f.name
        for f in fm.fontManager.ttflist
    }

    selected = None

    for font in candidates:
        if font in available_fonts:
            selected = font
            break

    if selected is not None:
        plt.rcParams["font.family"] = selected

    plt.rcParams["axes.unicode_minus"] = False

    print(
        "사용 한글 폰트:",
        selected if selected else "기본 폰트"
    )


setup_korean_font()


# ============================================================
# 2. 그래프 공통 설정
# ============================================================

plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 11

plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

plt.rcParams["legend.fontsize"] = 10


def comma_formatter(x, pos):
    return f"{x:,.0f}"


def save_figure(filename):

    path = FIGURE_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"저장 완료: {path}")


def annotate_bars(
    ax,
    decimals=1,
    suffix=""
):

    for container in ax.containers:

        labels = []

        for value in container.datavalues:

            if np.isnan(value):
                labels.append("")
            else:
                labels.append(
                    f"{value:.{decimals}f}{suffix}"
                )

        ax.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9
        )


# ============================================================
# 3. 데이터 로드
# ============================================================

print(
    "\n========== 시각화 데이터 로드 =========="
)

hourly_forecast = pd.read_csv(
    HOURLY_FORECAST_FILE,
    encoding="utf-8-sig"
)

forecast_test = pd.read_csv(
    FORECAST_TEST_FILE,
    encoding="utf-8-sig"
)

summary = pd.read_csv(
    SUMMARY_FILE,
    encoding="utf-8-sig"
)

by_hour = pd.read_csv(
    BY_HOUR_FILE,
    encoding="utf-8-sig"
)

by_day = pd.read_csv(
    BY_DAY_FILE,
    encoding="utf-8-sig"
)

routes = pd.read_csv(
    ROUTES_FILE,
    encoding="utf-8-sig"
)

time_limit = pd.read_csv(
    TIME_LIMIT_FILE,
    encoding="utf-8-sig"
)

efficiency = pd.read_csv(
    EFFICIENCY_FILE,
    encoding="utf-8-sig"
)

forecast_test["날짜"] = pd.to_datetime(
    forecast_test["날짜"]
)

print(
    f"수요예측 테스트 데이터: "
    f"{len(forecast_test):,}개"
)

print(
    f"정책 비교 데이터: "
    f"{len(summary):,}개"
)

print(
    f"재배치 경로 데이터: "
    f"{len(routes):,}개"
)


# ============================================================
# 4. 기준 정책 추출 함수
# ============================================================

def get_baseline_row(policy_name):

    temp = summary[
        summary["정책"] == policy_name
    ]

    if len(temp) == 0:
        return None

    return temp.iloc[0]


current_row = get_baseline_row("현재배치")
fixed_row = get_baseline_row("고정배치")


# ============================================================
# 5. LightGBM / Oracle 데이터
# ============================================================

lgb_summary = summary[
    summary["정책"] == "LightGBM선제"
].copy()

oracle_summary = summary[
    summary["정책"] == "Oracle선제"
].copy()

lgb_summary = lgb_summary.sort_values(
    "시간제한_분"
)

oracle_summary = oracle_summary.sort_values(
    "시간제한_분"
)


# ============================================================
# 6. 권장 시간제약 계산
# ============================================================

print(
    "\n========== 권장 시간제약 계산 =========="
)

eff = efficiency.copy()

eff = eff[
    eff["정책"] == "LightGBM선제"
].copy()

eff = eff.sort_values(
    "시간제한_분"
)

# ------------------------------------------------------------
# 단순히 커버리지가 가장 높은 시간보다
# 비용 대비 개선효율도 함께 확인
# ------------------------------------------------------------

best_efficiency_idx = (
    eff["개선율_per_1000공차분"]
    .idxmax()
)

best_efficiency_row = eff.loc[
    best_efficiency_idx
]

BEST_EFFICIENCY_TIME = int(
    best_efficiency_row["시간제한_분"]
)

BEST_EFFICIENCY_COVERAGE = float(
    best_efficiency_row[
        "수요가중커버리지_%"
    ]
)

# 최대 커버리지 시간
best_coverage_idx = (
    eff["수요가중커버리지_%"]
    .idxmax()
)

best_coverage_row = eff.loc[
    best_coverage_idx
]

BEST_COVERAGE_TIME = int(
    best_coverage_row["시간제한_분"]
)

BEST_COVERAGE = float(
    best_coverage_row[
        "수요가중커버리지_%"
    ]
)

print(
    f"비용효율 최고 시간제약: "
    f"{BEST_EFFICIENCY_TIME}분"
)

print(
    f"해당 커버리지: "
    f"{BEST_EFFICIENCY_COVERAGE:.2f}%"
)

print(
    f"최대 커버리지 시간제약: "
    f"{BEST_COVERAGE_TIME}분"
)

print(
    f"최대 커버리지: "
    f"{BEST_COVERAGE:.2f}%"
)


# ============================================================
# GRAPH 01
# 시간대별 실제 수요
# ============================================================

print(
    "\n========== GRAPH 01 =========="
)

df = hourly_forecast.sort_values(
    "시간대"
)

fig, ax = plt.subplots(
    figsize=(13, 7)
)

bars = ax.bar(
    df["시간대"],
    df["실제수요"]
)

ax.set_title(
    "시간대별 장애인콜택시 실제 수요"
)

ax.set_xlabel(
    "시간대"
)

ax.set_ylabel(
    "총 수요 건수"
)

ax.set_xticks(
    range(24)
)

ax.yaxis.set_major_formatter(
    FuncFormatter(comma_formatter)
)

ax.grid(
    axis="y",
    alpha=0.25
)

# 최대 수요 시간 표시
max_row = df.loc[
    df["실제수요"].idxmax()
]

max_hour = int(
    max_row["시간대"]
)

max_demand = float(
    max_row["실제수요"]
)

ax.annotate(
    f"최대 수요\n{max_hour}시\n"
    f"{max_demand:,.0f}건",
    xy=(
        max_hour,
        max_demand
    ),
    xytext=(
        max_hour + 2,
        max_demand * 0.90
    ),
    arrowprops=dict(
        arrowstyle="->"
    ),
    fontsize=10
)

save_figure(
    "01_hourly_demand.png"
)


# ============================================================
# GRAPH 02
# 실제 vs LightGBM 예측 수요
# ============================================================

print(
    "\n========== GRAPH 02 =========="
)

fig, ax = plt.subplots(
    figsize=(13, 7)
)

ax.plot(
    df["시간대"],
    df["실제수요"],
    marker="o",
    linewidth=2,
    label="실제 수요"
)

ax.plot(
    df["시간대"],
    df["예측수요"],
    marker="s",
    linewidth=2,
    label="LightGBM 예측 수요"
)

ax.set_title(
    "시간대별 실제 수요와 LightGBM 예측 수요"
)

ax.set_xlabel(
    "시간대"
)

ax.set_ylabel(
    "총 수요 건수"
)

ax.set_xticks(
    range(24)
)

ax.yaxis.set_major_formatter(
    FuncFormatter(comma_formatter)
)

ax.legend()

ax.grid(
    alpha=0.25
)

save_figure(
    "02_actual_vs_prediction.png"
)


# ============================================================
# GRAPH 03
# 정책 비교
# ============================================================

print(
    "\n========== GRAPH 03 =========="
)

policy_rows = []

if fixed_row is not None:

    policy_rows.append({
        "정책": "고정배치",
        "커버리지": fixed_row[
            "수요가중커버리지_%"
        ]
    })

if current_row is not None:

    policy_rows.append({
        "정책": "현재배치",
        "커버리지": current_row[
            "수요가중커버리지_%"
        ]
    })

# LightGBM 최대 커버리지
lgb_best = lgb_summary.loc[
    lgb_summary[
        "수요가중커버리지_%"
    ].idxmax()
]

policy_rows.append({
    "정책":
        f"LightGBM\n"
        f"({int(lgb_best['시간제한_분'])}분)",
    "커버리지":
        lgb_best[
            "수요가중커버리지_%"
        ]
})

if len(oracle_summary) > 0:

    oracle_best = oracle_summary.loc[
        oracle_summary[
            "수요가중커버리지_%"
        ].idxmax()
    ]

    policy_rows.append({
        "정책":
            f"Oracle\n"
            f"({int(oracle_best['시간제한_분'])}분)",
        "커버리지":
            oracle_best[
                "수요가중커버리지_%"
            ]
    })


policy_df = pd.DataFrame(
    policy_rows
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.bar(
    policy_df["정책"],
    policy_df["커버리지"]
)

ax.set_title(
    "UD 12대 배치 정책별 수요가중 커버리지"
)

ax.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax.set_ylim(
    0,
    min(
        100,
        policy_df[
            "커버리지"
        ].max() + 12
    )
)

ax.grid(
    axis="y",
    alpha=0.25
)

annotate_bars(
    ax,
    decimals=2,
    suffix="%"
)

save_figure(
    "03_policy_comparison.png"
)


# ============================================================
# GRAPH 04
# 재배치 시간제약별 성능
# ============================================================

print(
    "\n========== GRAPH 04 =========="
)

tl = time_limit.sort_values(
    "시간제한_분"
)

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.plot(
    tl["시간제한_분"],
    tl["수요가중커버리지_%"],
    marker="o",
    linewidth=2.5,
    markersize=8
)

for _, row in tl.iterrows():

    ax.annotate(
        f"{row['수요가중커버리지_%']:.2f}%",
        (
            row["시간제한_분"],
            row["수요가중커버리지_%"]
        ),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

if current_row is not None:

    current_coverage = float(
        current_row[
            "수요가중커버리지_%"
        ]
    )

    ax.axhline(
        current_coverage,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"현재배치 "
            f"{current_coverage:.2f}%"
        )
    )

ax.set_title(
    "재배치 허용시간에 따른 LightGBM 정책 성능"
)

ax.set_xlabel(
    "최대 재배치 허용시간 (분)"
)

ax.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax.set_xticks(
    tl["시간제한_분"]
)

ax.grid(
    alpha=0.25
)

if current_row is not None:
    ax.legend()

save_figure(
    "04_time_limit_performance.png"
)


# ============================================================
# GRAPH 05
# 비용-성능 Trade-off
# ============================================================

print(
    "\n========== GRAPH 05 =========="
)

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.plot(
    eff["총공차차량시간_분"],
    eff["수요가중커버리지_%"],
    marker="o",
    linewidth=2.5,
    markersize=9
)

for _, row in eff.iterrows():

    ax.annotate(
        f"{int(row['시간제한_분'])}분",
        (
            row["총공차차량시간_분"],
            row["수요가중커버리지_%"]
        ),
        textcoords="offset points",
        xytext=(8, 8)
    )

ax.set_title(
    "재배치 비용과 수요 커버리지의 Trade-off"
)

ax.set_xlabel(
    "총 공차 차량시간 (분)"
)

ax.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax.xaxis.set_major_formatter(
    FuncFormatter(comma_formatter)
)

ax.grid(
    alpha=0.25
)

save_figure(
    "05_cost_performance_tradeoff.png"
)


# ============================================================
# GRAPH 06
# 시간대별 LightGBM 커버리지
#
# 최대 커버리지 정책을 기준으로 표시
# ============================================================

print(
    "\n========== GRAPH 06 =========="
)

hour_policy = by_hour[
    (
        by_hour["정책"] ==
        "LightGBM선제"
    )
    &
    (
        by_hour["시간제한_분"] ==
        BEST_COVERAGE_TIME
    )
].copy()

hour_policy = hour_policy.sort_values(
    "시간대"
)

fig, ax = plt.subplots(
    figsize=(13, 7)
)

ax.bar(
    hour_policy["시간대"],
    hour_policy[
        "수요가중커버리지_%"
    ]
)

ax.set_title(
    f"시간대별 LightGBM 수요가중 커버리지 "
    f"({BEST_COVERAGE_TIME}분 재배치)"
)

ax.set_xlabel(
    "시간대"
)

ax.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax.set_xticks(
    range(24)
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    axis="y",
    alpha=0.25
)

save_figure(
    "06_hourly_coverage.png"
)


# ============================================================
# GRAPH 07
# 요일별 LightGBM 커버리지
# ============================================================

print(
    "\n========== GRAPH 07 =========="
)

day_order = [
    "월",
    "화",
    "수",
    "목",
    "금",
    "토",
    "일"
]

day_policy = by_day[
    (
        by_day["정책"] ==
        "LightGBM선제"
    )
    &
    (
        by_day["시간제한_분"] ==
        BEST_COVERAGE_TIME
    )
].copy()

day_policy["요일"] = pd.Categorical(
    day_policy["요일"],
    categories=day_order,
    ordered=True
)

day_policy = day_policy.sort_values(
    "요일"
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.bar(
    day_policy["요일"].astype(str),
    day_policy[
        "수요가중커버리지_%"
    ]
)

ax.set_title(
    f"요일별 LightGBM 수요가중 커버리지 "
    f"({BEST_COVERAGE_TIME}분 재배치)"
)

ax.set_xlabel(
    "요일"
)

ax.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax.set_ylim(
    0,
    100
)

ax.grid(
    axis="y",
    alpha=0.25
)

annotate_bars(
    ax,
    decimals=1,
    suffix="%"
)

save_figure(
    "07_weekday_coverage.png"
)


# ============================================================
# GRAPH 08
# 주요 재배치 경로 TOP 10
# ============================================================

print(
    "\n========== GRAPH 08 =========="
)

route_policy = routes[
    (
        routes["정책"] ==
        "LightGBM선제"
    )
    &
    (
        routes["시간제한_분"] ==
        BEST_COVERAGE_TIME
    )
].copy()

route_policy["경로"] = (
    route_policy["재배치출발구"]
    + " → "
    + route_policy["재배치도착구"]
)

top_routes = (
    route_policy
    .sort_values(
        "총재배치UD",
        ascending=False
    )
    .head(10)
    .sort_values(
        "총재배치UD",
        ascending=True
    )
)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

bars = ax.barh(
    top_routes["경로"],
    top_routes["총재배치UD"]
)

ax.set_title(
    f"LightGBM 주요 선제 재배치 경로 TOP 10 "
    f"({BEST_COVERAGE_TIME}분)"
)

ax.set_xlabel(
    "총 재배치량 (대·회)"
)

ax.set_ylabel(
    "재배치 경로"
)

ax.grid(
    axis="x",
    alpha=0.25
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width,
        bar.get_y()
        + bar.get_height() / 2,
        f" {width:.1f}",
        va="center",
        fontsize=9
    )

save_figure(
    "08_rebalancing_routes_top10.png"
)


# ============================================================
# GRAPH 09
# 차량 없는 수요 비율
# ============================================================

print(
    "\n========== GRAPH 09 =========="
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.bar(
    tl["시간제한_분"].astype(str),
    tl["차량없는수요비율_%"]
)

ax.set_title(
    "재배치 시간제약별 차량 없는 수요 비율"
)

ax.set_xlabel(
    "최대 재배치 허용시간 (분)"
)

ax.set_ylabel(
    "차량 없는 수요 비율 (%)"
)

ax.grid(
    axis="y",
    alpha=0.25
)

annotate_bars(
    ax,
    decimals=3,
    suffix="%"
)

save_figure(
    "09_zero_vehicle_demand.png"
)


# ============================================================
# GRAPH 10
# 정책 종합 Dashboard
# ============================================================

print(
    "\n========== GRAPH 10 =========="
)

fig = plt.figure(
    figsize=(16, 11)
)

# ------------------------------------------------------------
# A. 시간대별 수요
# ------------------------------------------------------------

ax1 = fig.add_subplot(
    2,
    2,
    1
)

ax1.plot(
    df["시간대"],
    df["실제수요"],
    marker="o",
    label="실제"
)

ax1.plot(
    df["시간대"],
    df["예측수요"],
    marker="s",
    label="예측"
)

ax1.set_title(
    "① 시간대별 실제·예측 수요"
)

ax1.set_xlabel(
    "시간대"
)

ax1.set_ylabel(
    "수요 건수"
)

ax1.set_xticks(
    range(0, 24, 2)
)

ax1.yaxis.set_major_formatter(
    FuncFormatter(comma_formatter)
)

ax1.grid(
    alpha=0.2
)

ax1.legend()


# ------------------------------------------------------------
# B. 시간제약별 커버리지
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    2,
    2,
    2
)

ax2.plot(
    tl["시간제한_분"],
    tl["수요가중커버리지_%"],
    marker="o",
    linewidth=2
)

ax2.set_title(
    "② 재배치 허용시간별 커버리지"
)

ax2.set_xlabel(
    "재배치 허용시간 (분)"
)

ax2.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax2.set_xticks(
    tl["시간제한_분"]
)

ax2.grid(
    alpha=0.2
)


# ------------------------------------------------------------
# C. 정책 비교
# ------------------------------------------------------------

ax3 = fig.add_subplot(
    2,
    2,
    3
)

bars = ax3.bar(
    policy_df["정책"],
    policy_df["커버리지"]
)

ax3.set_title(
    "③ UD 배치 정책 비교"
)

ax3.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax3.set_ylim(
    0,
    100
)

ax3.grid(
    axis="y",
    alpha=0.2
)

for bar, value in zip(
    bars,
    policy_df["커버리지"]
):

    ax3.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=9
    )


# ------------------------------------------------------------
# D. 비용-효과
# ------------------------------------------------------------

ax4 = fig.add_subplot(
    2,
    2,
    4
)

ax4.plot(
    eff["총공차차량시간_분"],
    eff["수요가중커버리지_%"],
    marker="o",
    linewidth=2
)

for _, row in eff.iterrows():

    ax4.annotate(
        f"{int(row['시간제한_분'])}분",
        (
            row["총공차차량시간_분"],
            row["수요가중커버리지_%"]
        ),
        textcoords="offset points",
        xytext=(5, 7),
        fontsize=9
    )

ax4.set_title(
    "④ 재배치 비용-효과"
)

ax4.set_xlabel(
    "총 공차 차량시간 (분)"
)

ax4.set_ylabel(
    "수요가중 커버리지 (%)"
)

ax4.xaxis.set_major_formatter(
    FuncFormatter(comma_formatter)
)

ax4.grid(
    alpha=0.2
)


fig.suptitle(
    "서울시 UD택시 수요예측 기반 동적 배치 정책 분석",
    fontsize=20,
    fontweight="bold",
    y=0.99
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

dashboard_path = (
    FIGURE_DIR /
    "10_policy_dashboard.png"
)

plt.savefig(
    dashboard_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print(
    f"저장 완료: {dashboard_path}"
)


# ============================================================
# 7. 추가 분석
# ============================================================

print(
    "\n========== 핵심 결과 =========="
)

print(
    f"LightGBM 비용효율 최고 시간제약: "
    f"{BEST_EFFICIENCY_TIME}분"
)

print(
    f"LightGBM 최대 커버리지 시간제약: "
    f"{BEST_COVERAGE_TIME}분"
)

print(
    f"LightGBM 최대 수요가중 커버리지: "
    f"{BEST_COVERAGE:.2f}%"
)


if current_row is not None:

    current_coverage = float(
        current_row[
            "수요가중커버리지_%"
        ]
    )

    improvement = (
        BEST_COVERAGE
        - current_coverage
    )

    print(
        f"현재배치 커버리지: "
        f"{current_coverage:.2f}%"
    )

    print(
        f"현재배치 대비 최대 개선: "
        f"{improvement:+.2f}%p"
    )


if fixed_row is not None:

    fixed_coverage = float(
        fixed_row[
            "수요가중커버리지_%"
        ]
    )

    fixed_improvement = (
        BEST_COVERAGE
        - fixed_coverage
    )

    print(
        f"고정배치 커버리지: "
        f"{fixed_coverage:.2f}%"
    )

    print(
        f"고정배치 대비 최대 개선: "
        f"{fixed_improvement:+.2f}%p"
    )


# ============================================================
# 8. 시간제약별 상세 출력
# ============================================================

print(
    "\n========== 시간제약별 정책 효과 =========="
)

print_columns = [
    "시간제한_분",
    "수요가중커버리지_%",
    "차량없는수요비율_%",
    "총재배치UD",
    "총공차차량시간_분",
    "개선율_per_1000공차분"
]

print(
    eff[
        print_columns
    ].round(3).to_string(
        index=False
    )
)


# ============================================================
# 9. 한계효용 분석
# ============================================================

print(
    "\n========== 재배치 시간 증가 한계효용 =========="
)

marginal = eff[
    [
        "시간제한_분",
        "수요가중커버리지_%",
        "총재배치UD",
        "총공차차량시간_분"
    ]
].copy()

marginal[
    "커버리지증가_%p"
] = (
    marginal[
        "수요가중커버리지_%"
    ].diff()
)

marginal[
    "추가재배치UD"
] = (
    marginal[
        "총재배치UD"
    ].diff()
)

marginal[
    "추가공차시간_분"
] = (
    marginal[
        "총공차차량시간_분"
    ].diff()
)

marginal[
    "추가1000공차분당개선_%p"
] = np.where(
    marginal[
        "추가공차시간_분"
    ] > 0,
    (
        marginal[
            "커버리지증가_%p"
        ]
        /
        marginal[
            "추가공차시간_분"
        ]
        * 1000
    ),
    np.nan
)

print(
    marginal.round(3).to_string(
        index=False
    )
)


# ============================================================
# 10. 완료
# ============================================================

print(
    "\n============================================"
)

print(
    "모든 시각화 생성 완료"
)

print(
    f"저장 폴더:\n{FIGURE_DIR}"
)

print(
    "============================================"
)