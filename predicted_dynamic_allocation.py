import os
import numpy as np
import pandas as pd


# ============================================================
# 0. 경로 설정
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

FORECAST_PATH = os.path.join(
    RESULT_DIR,
    "demand_forecast_test_v2_2025.csv"
)

OUTPUT_HOURLY = os.path.join(
    RESULT_DIR,
    "predicted_dynamic_allocation_hourly_2025.csv"
)

OUTPUT_GU = os.path.join(
    RESULT_DIR,
    "predicted_dynamic_allocation_gu_2025.csv"
)

OUTPUT_SUMMARY = os.path.join(
    RESULT_DIR,
    "predicted_dynamic_allocation_summary_2025.csv"
)


# ============================================================
# 1. 기본 설정
# ============================================================

TOTAL_UD = 12

# 기존 분석에서 UD 12대를 장애인콜택시 수요에 비례하여
# 배분하는 구조를 유지한다.
#
# 핵심:
# 실제 수요가 아니라 예측수요를 기준으로 12대를 배치한다.
#
# 실제 수요는 사후 평가에만 사용한다.


# ============================================================
# 2. 데이터 로드
# ============================================================

print("========== LightGBM v2 예측 데이터 로드 ==========")

df = pd.read_csv(FORECAST_PATH)

print(f"데이터 수: {len(df):,}개")
print("컬럼:")
print(", ".join(df.columns))


# ============================================================
# 3. 필수 컬럼 확인
# ============================================================

required_columns = [
    "날짜",
    "시간대",
    "출발구",
    "수요",
    "예측수요",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"demand_forecast_test_v2_2025.csv에 "
        f"다음 컬럼이 없습니다: {missing}"
    )


# ============================================================
# 4. 기본 전처리
# ============================================================

df["날짜"] = pd.to_datetime(df["날짜"])
df["시간대"] = pd.to_numeric(
    df["시간대"],
    errors="coerce"
)

df["수요"] = pd.to_numeric(
    df["수요"],
    errors="coerce"
).fillna(0)

df["예측수요"] = pd.to_numeric(
    df["예측수요"],
    errors="coerce"
).fillna(0)

# 음수 예측 방지
df["예측수요"] = df["예측수요"].clip(lower=0)

df = df.dropna(
    subset=["날짜", "시간대", "출발구"]
).copy()

df["시간대"] = df["시간대"].astype(int)


# 요일 컬럼이 없다면 생성
weekday_map = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일",
}

if "요일" not in df.columns:
    df["요일"] = df["날짜"].dt.weekday.map(weekday_map)


# ============================================================
# 5. 예측 Bias 계산
# ============================================================

print("\n========== 예측 Bias 분석 ==========")

actual_total = df["수요"].sum()
pred_total = df["예측수요"].sum()

if pred_total > 0:
    global_bias_factor = actual_total / pred_total
else:
    global_bias_factor = 1.0

print(f"실제 총수요: {actual_total:,.2f}")
print(f"예측 총수요: {pred_total:,.2f}")
print(f"전체 Bias 보정계수: {global_bias_factor:.4f}")


# ============================================================
# 6. Bias 보정
# ============================================================
#
# 중요한 점:
#
# 테스트 전체 실제값을 이용해서 보정계수를 만드는 것은
# 엄밀한 실시간 예측에서는 정보 누수로 볼 수 있다.
#
# 따라서 이 값은 "분석용 보정 시나리오"로 사용한다.
#
# 동시에 원본 예측 기반 결과도 함께 계산한다.
# ============================================================

df["보정예측수요"] = (
    df["예측수요"] * global_bias_factor
)


# ============================================================
# 7. Largest Remainder 방식 차량 배분 함수
# ============================================================
#
# 예측수요 비율에 따라 12대를 배분한다.
#
# 단순 반올림을 하면
# 11대 또는 13대가 되는 문제가 발생할 수 있기 때문에
# Largest Remainder Method를 사용한다.
#
# 항상 정확히 12대가 배분된다.
# ============================================================

def allocate_integer_fleet(
    group,
    demand_col,
    fleet_size=12
):
    g = group.copy()

    demand = (
        pd.to_numeric(
            g[demand_col],
            errors="coerce"
        )
        .fillna(0)
        .clip(lower=0)
    )

    total_demand = demand.sum()

    # 수요가 전혀 없는 경우
    if total_demand <= 0:
        g["배치UD"] = 0

        # 첫 지역부터 차량 배정
        idx_list = list(g.index)

        for i in range(fleet_size):
            idx = idx_list[i % len(idx_list)]
            g.loc[idx, "배치UD"] += 1

        return g

    # 이상적인 연속값 배분
    raw_allocation = (
        demand / total_demand * fleet_size
    )

    # 우선 내림
    base_allocation = np.floor(
        raw_allocation
    ).astype(int)

    g["배치UD"] = base_allocation

    remaining = (
        fleet_size
        - int(base_allocation.sum())
    )

    # 소수점 잔여
    remainder = (
        raw_allocation
        - base_allocation
    )

    g["_remainder"] = remainder

    # 잔여 차량 배분
    if remaining > 0:
        priority = (
            g.sort_values(
                ["_remainder", demand_col],
                ascending=[False, False]
            )
            .index
            .tolist()
        )

        for idx in priority[:remaining]:
            g.loc[idx, "배치UD"] += 1

    g = g.drop(
        columns=["_remainder"]
    )

    return g


# ============================================================
# 8. 실제 수요 기준 필요 UD 계산 함수
# ============================================================
#
# 여기서 중요한 것은
#
# "실제 수요에 맞춰 12대를 최적으로 배치했다면"
# 어느 구에 몇 대가 필요했을지를 계산하는 것이다.
#
# 이것은 예측 배치의 평가 기준으로만 사용한다.
# ============================================================

def calculate_actual_required_fleet(
    group,
    fleet_size=12
):
    temp = allocate_integer_fleet(
        group,
        "수요",
        fleet_size
    )

    return temp["배치UD"]


# ============================================================
# 9. 시간대별 동적 배치
# ============================================================

print("\n========== 예측 기반 UD 12대 동적 배치 ==========")

results = []

group_cols = [
    "날짜",
    "요일",
    "시간대",
]

for keys, group in df.groupby(
    group_cols,
    sort=True
):

    date, weekday, hour = keys

    group = group.copy()

    # --------------------------------------------------------
    # 실제 필요 차량
    # --------------------------------------------------------

    actual_temp = allocate_integer_fleet(
        group,
        "수요",
        TOTAL_UD
    )

    group["실제필요UD"] = (
        actual_temp["배치UD"]
        .reindex(group.index)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # 원본 LightGBM 예측 기반 배치
    # --------------------------------------------------------

    pred_temp = allocate_integer_fleet(
        group,
        "예측수요",
        TOTAL_UD
    )

    group["예측배치UD"] = (
        pred_temp["배치UD"]
        .reindex(group.index)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Bias 보정 예측 기반 배치
    # --------------------------------------------------------

    corrected_temp = allocate_integer_fleet(
        group,
        "보정예측수요",
        TOTAL_UD
    )

    group["보정예측배치UD"] = (
        corrected_temp["배치UD"]
        .reindex(group.index)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # 실제 필요량과 비교
    # --------------------------------------------------------

    group["예측배치충족UD"] = np.minimum(
        group["실제필요UD"],
        group["예측배치UD"]
    )

    group["보정배치충족UD"] = np.minimum(
        group["실제필요UD"],
        group["보정예측배치UD"]
    )

    group["예측배치부족UD"] = np.maximum(
        group["실제필요UD"]
        - group["예측배치UD"],
        0
    )

    group["보정배치부족UD"] = np.maximum(
        group["실제필요UD"]
        - group["보정예측배치UD"],
        0
    )

    group["예측배치잉여UD"] = np.maximum(
        group["예측배치UD"]
        - group["실제필요UD"],
        0
    )

    group["보정배치잉여UD"] = np.maximum(
        group["보정예측배치UD"]
        - group["실제필요UD"],
        0
    )

    results.append(group)


allocation_df = pd.concat(
    results,
    ignore_index=True
)


# ============================================================
# 10. 12대 제약 검증
# ============================================================

print("\n========== UD 12대 제약 검증 ==========")

fleet_check = (
    allocation_df
    .groupby(
        ["날짜", "시간대"],
        as_index=False
    )
    .agg(
        예측배치합=("예측배치UD", "sum"),
        보정배치합=("보정예측배치UD", "sum"),
        실제필요합=("실제필요UD", "sum"),
    )
)

pred_violation = (
    fleet_check["예측배치합"]
    != TOTAL_UD
).sum()

corrected_violation = (
    fleet_check["보정배치합"]
    != TOTAL_UD
).sum()

actual_violation = (
    fleet_check["실제필요합"]
    != TOTAL_UD
).sum()

print(
    f"분석 시간대: "
    f"{len(fleet_check):,}개"
)

print(
    f"예측 배치 12대 위반: "
    f"{pred_violation:,}개"
)

print(
    f"보정 배치 12대 위반: "
    f"{corrected_violation:,}개"
)

print(
    f"실제 기준 12대 위반: "
    f"{actual_violation:,}개"
)


# ============================================================
# 11. 시간대별 성능 계산
# ============================================================

hourly = (
    allocation_df
    .groupby(
        ["날짜", "요일", "시간대"],
        as_index=False
    )
    .agg(
        실제총수요=("수요", "sum"),
        예측총수요=("예측수요", "sum"),
        보정예측총수요=("보정예측수요", "sum"),

        실제필요UD=("실제필요UD", "sum"),

        예측배치UD=("예측배치UD", "sum"),
        예측충족UD=("예측배치충족UD", "sum"),
        예측부족UD=("예측배치부족UD", "sum"),
        예측잉여UD=("예측배치잉여UD", "sum"),

        보정배치UD=("보정예측배치UD", "sum"),
        보정충족UD=("보정배치충족UD", "sum"),
        보정부족UD=("보정배치부족UD", "sum"),
        보정잉여UD=("보정배치잉여UD", "sum"),
    )
)


hourly["예측배치충족률_%"] = np.where(
    hourly["실제필요UD"] > 0,
    (
        hourly["예측충족UD"]
        / hourly["실제필요UD"]
        * 100
    ),
    100
)

hourly["보정배치충족률_%"] = np.where(
    hourly["실제필요UD"] > 0,
    (
        hourly["보정충족UD"]
        / hourly["실제필요UD"]
        * 100
    ),
    100
)


# ============================================================
# 12. 결과 TOP 30
# ============================================================

print(
    "\n========== 예측 배치 오차 심각 시간대 TOP 30 =========="
)

top30 = (
    hourly
    .sort_values(
        [
            "예측부족UD",
            "예측배치충족률_%"
        ],
        ascending=[False, True]
    )
    .head(30)
)

show_cols = [
    "날짜",
    "요일",
    "시간대",
    "실제총수요",
    "예측총수요",
    "실제필요UD",
    "예측배치UD",
    "예측충족UD",
    "예측부족UD",
    "예측잉여UD",
    "예측배치충족률_%",
]

print(
    top30[show_cols]
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 13. 시간대별 평균 성능
# ============================================================

print(
    "\n========== 시간대별 예측 동적 배치 성능 =========="
)

hour_summary = (
    hourly
    .groupby(
        "시간대",
        as_index=False
    )
    .agg(
        분석일수=("날짜", "nunique"),

        평균실제수요=(
            "실제총수요",
            "mean"
        ),

        평균예측수요=(
            "예측총수요",
            "mean"
        ),

        평균예측부족UD=(
            "예측부족UD",
            "mean"
        ),

        평균예측잉여UD=(
            "예측잉여UD",
            "mean"
        ),

        평균충족률=(
            "예측배치충족률_%",
            "mean"
        ),
    )
)

print(
    hour_summary
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 14. 요일별 성능
# ============================================================

print(
    "\n========== 요일별 예측 동적 배치 성능 =========="
)

weekday_order = [
    "월", "화", "수", "목", "금", "토", "일"
]

daily_summary = (
    hourly
    .groupby(
        "요일",
        as_index=False
    )
    .agg(
        분석시간대=("시간대", "count"),

        평균실제수요=(
            "실제총수요",
            "mean"
        ),

        평균예측수요=(
            "예측총수요",
            "mean"
        ),

        평균부족UD=(
            "예측부족UD",
            "mean"
        ),

        평균잉여UD=(
            "예측잉여UD",
            "mean"
        ),

        평균충족률=(
            "예측배치충족률_%",
            "mean"
        ),
    )
)

daily_summary["요일"] = pd.Categorical(
    daily_summary["요일"],
    categories=weekday_order,
    ordered=True
)

daily_summary = daily_summary.sort_values(
    "요일"
)

print(
    daily_summary
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 15. 구별 배치 성능
# ============================================================

print(
    "\n========== 출발구별 예측 동적 배치 성능 =========="
)

gu_summary = (
    allocation_df
    .groupby(
        "출발구",
        as_index=False
    )
    .agg(
        실제수요=("수요", "sum"),
        예측수요=("예측수요", "sum"),

        실제필요UD=("실제필요UD", "sum"),
        예측배치UD=("예측배치UD", "sum"),

        충족UD=("예측배치충족UD", "sum"),
        부족UD=("예측배치부족UD", "sum"),
        잉여UD=("예측배치잉여UD", "sum"),
    )
)

gu_summary["배치충족률_%"] = np.where(
    gu_summary["실제필요UD"] > 0,
    (
        gu_summary["충족UD"]
        / gu_summary["실제필요UD"]
        * 100
    ),
    100
)

gu_summary = gu_summary.sort_values(
    "부족UD",
    ascending=False
)

print(
    gu_summary
    .round(2)
    .to_string(index=False)
)


# ============================================================
# 16. 전체 성능
# ============================================================

total_required = hourly["실제필요UD"].sum()

total_pred_satisfied = (
    hourly["예측충족UD"].sum()
)

total_pred_shortage = (
    hourly["예측부족UD"].sum()
)

total_pred_surplus = (
    hourly["예측잉여UD"].sum()
)

total_corrected_satisfied = (
    hourly["보정충족UD"].sum()
)

total_corrected_shortage = (
    hourly["보정부족UD"].sum()
)

if total_required > 0:
    overall_pred_rate = (
        total_pred_satisfied
        / total_required
        * 100
    )

    overall_corrected_rate = (
        total_corrected_satisfied
        / total_required
        * 100
    )
else:
    overall_pred_rate = 100
    overall_corrected_rate = 100


print(
    "\n========== 전체 예측 기반 동적 배치 결과 =========="
)

print(
    f"총 실제 필요 UD: "
    f"{total_required:,.2f}대·회"
)

print(
    f"예측 배치 충족 UD: "
    f"{total_pred_satisfied:,.2f}대·회"
)

print(
    f"예측 배치 부족 UD: "
    f"{total_pred_shortage:,.2f}대·회"
)

print(
    f"예측 배치 잉여 UD: "
    f"{total_pred_surplus:,.2f}대·회"
)

print(
    f"예측 기반 동적 배치 충족률: "
    f"{overall_pred_rate:.2f}%"
)


print(
    "\n========== Bias 보정 시나리오 =========="
)

print(
    f"보정 배치 충족 UD: "
    f"{total_corrected_satisfied:,.2f}대·회"
)

print(
    f"보정 배치 부족 UD: "
    f"{total_corrected_shortage:,.2f}대·회"
)

print(
    f"Bias 보정 동적 배치 충족률: "
    f"{overall_corrected_rate:.2f}%"
)


# ============================================================
# 17. 고정 균등배치 Baseline
# ============================================================
#
# 25개 구에 12대를 균등하게 정수 배치하는 것은 불가능하다.
#
# 따라서 각 시간대마다 실제 수요와 무관하게
# 구 이름 순으로 12개 구에 1대씩 배치하는 단순 baseline을
# 만든다.
#
# 이 baseline은 절대적인 정책안이라기보다는
# "수요를 고려하지 않는 배치"와 비교하기 위한 기준이다.
# ============================================================

baseline_results = []

for keys, group in allocation_df.groupby(
    ["날짜", "요일", "시간대"],
    sort=True
):

    g = group.copy()

    g = g.sort_values(
        "출발구"
    ).copy()

    g["고정배치UD"] = 0

    selected_idx = g.index[:TOTAL_UD]

    g.loc[
        selected_idx,
        "고정배치UD"
    ] = 1

    g["고정충족UD"] = np.minimum(
        g["실제필요UD"],
        g["고정배치UD"]
    )

    baseline_results.append(g)


baseline_df = pd.concat(
    baseline_results,
    ignore_index=True
)

baseline_required = (
    baseline_df["실제필요UD"].sum()
)

baseline_satisfied = (
    baseline_df["고정충족UD"].sum()
)

if baseline_required > 0:
    baseline_rate = (
        baseline_satisfied
        / baseline_required
        * 100
    )
else:
    baseline_rate = 100


print(
    "\n========== 단순 고정배치 Baseline 비교 =========="
)

print(
    f"고정배치 충족률: "
    f"{baseline_rate:.2f}%"
)

print(
    f"예측 동적배치 충족률: "
    f"{overall_pred_rate:.2f}%"
)

print(
    f"개선: "
    f"{overall_pred_rate - baseline_rate:+.2f}%p"
)


# ============================================================
# 18. Oracle 비교
# ============================================================
#
# predictive_rebalancing_v3.py 결과:
# 60분 제약 Oracle = 89.20%
#
# 주의:
# 여기의 동적배치 충족률과 Oracle 재배치 충족률은
# 완전히 동일한 단계의 지표는 아니다.
#
# 따라서 참고 Benchmark로 출력하고,
# 최종 비교는 다음 재배치 단계에서 수행한다.
# ============================================================

ORACLE_60_RATE = 89.20

print(
    "\n========== Oracle Benchmark 참고 =========="
)

print(
    f"예측 기반 초기 동적배치 충족률: "
    f"{overall_pred_rate:.2f}%"
)

print(
    f"60분 Oracle 선제 재배치 충족률: "
    f"{ORACLE_60_RATE:.2f}%"
)

print(
    "※ 두 지표는 아직 동일한 재배치 단계의 성능이 아니므로 "
    "직접적인 우열 비교가 아니라 참고용입니다."
)


# ============================================================
# 19. Summary 저장
# ============================================================

summary = pd.DataFrame({
    "지표": [
        "실제총수요",
        "예측총수요",
        "Bias보정계수",
        "총실제필요UD",
        "예측충족UD",
        "예측부족UD",
        "예측잉여UD",
        "예측동적배치충족률_%",
        "Bias보정충족률_%",
        "고정배치Baseline충족률_%",
        "Oracle60분참고충족률_%",
    ],
    "값": [
        actual_total,
        pred_total,
        global_bias_factor,
        total_required,
        total_pred_satisfied,
        total_pred_shortage,
        total_pred_surplus,
        overall_pred_rate,
        overall_corrected_rate,
        baseline_rate,
        ORACLE_60_RATE,
    ]
})


# ============================================================
# 20. 저장
# ============================================================

allocation_df.to_csv(
    OUTPUT_HOURLY,
    index=False,
    encoding="utf-8-sig"
)

gu_summary.to_csv(
    OUTPUT_GU,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========== 저장 완료 =========="
)

print(OUTPUT_HOURLY)
print(OUTPUT_GU)
print(OUTPUT_SUMMARY)