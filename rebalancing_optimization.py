from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"
STOCK_PATH = RESULT_DIR / "vehicle_stock_detail_2025.csv"

RESULT_DIR.mkdir(exist_ok=True)

TOTAL_UD = 12
MAX_REBALANCE_TIME = 30

DAY_ORDER = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}


# =========================================================
# 2. 실제 운행자료 로드
# =========================================================

usecols = [
    "승차일시",
    "하차일시",
    "출발구",
    "목적구"
]

trip = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=usecols
)

trip["승차일시"] = pd.to_datetime(
    trip["승차일시"],
    errors="coerce"
)

trip["하차일시"] = pd.to_datetime(
    trip["하차일시"],
    errors="coerce"
)

trip["운행시간_분"] = (
    trip["하차일시"] - trip["승차일시"]
).dt.total_seconds() / 60


# =========================================================
# 3. 운행시간 품질 필터
# =========================================================

trip = trip[
    (trip["승차일시"].dt.year == 2025)
    & (trip["운행시간_분"] > 0)
    & (trip["운행시간_분"] <= 180)
].dropna(
    subset=[
        "출발구",
        "목적구"
    ]
).copy()

print("========== 이동시간 데이터 ==========")
print(f"분석 운행: {len(trip):,}건")


# =========================================================
# 4. 출발구 → 목적구 이동시간 계산
# =========================================================
# 이상치 영향을 줄이기 위해 평균 대신 중앙값 사용

travel_time = (
    trip
    .groupby(
        ["출발구", "목적구"],
        as_index=False
    )
    .agg(
        평균이동시간_분=("운행시간_분", "median"),
        운행건수=("운행시간_분", "size")
    )
)

travel_dict = {
    (row["출발구"], row["목적구"]):
        row["평균이동시간_분"]

    for _, row in travel_time.iterrows()
}

# OD 조합이 존재하지 않을 경우 전체 중앙값 사용
DEFAULT_TIME = trip["운행시간_분"].median()

print(f"구간 조합: {len(travel_time):,}개")
print(f"기본 이동시간: {DEFAULT_TIME:.2f}분")


# =========================================================
# 5. 차량 재고 데이터 로드
# =========================================================

stock = pd.read_csv(
    STOCK_PATH,
    encoding="utf-8-sig"
)

required_columns = [
    "요일",
    "승차시간대",
    "출발구",
    "목표UD",
    "시간대시작UD",
    "자연충족UD",
    "부족UD",
    "잉여UD"
]

missing_columns = [
    col
    for col in required_columns
    if col not in stock.columns
]

if missing_columns:
    raise ValueError(
        f"차량 재고 파일에 필요한 컬럼이 없습니다: "
        f"{missing_columns}"
    )

print("\n========== 차량 재고 데이터 ==========")
print(f"분석 조합: {len(stock):,}개")

print("\n사용 컬럼:")
print(", ".join(required_columns))


# =========================================================
# 6. 최적 재배치 함수
# =========================================================
# 같은 요일 × 시간대 안에서
# 잉여 지역 → 부족 지역으로 차량을 이동
#
# 비용 = 실제 운행자료의 구간별 중앙 이동시간
# Hungarian Algorithm으로 총 이동시간 최소화
# =========================================================

def optimize_rebalancing(group):

    # -----------------------------------------------------
    # 6-1. 잉여 / 부족 지역
    # -----------------------------------------------------

    surplus = group[
        group["잉여UD"] > 0
    ].copy()

    shortage = group[
        group["부족UD"] > 0
    ].copy()

    if surplus.empty or shortage.empty:
        return []

    # -----------------------------------------------------
    # 6-2. 차량 단위로 확장
    # -----------------------------------------------------
    # 차량은 실제로 정수 단위이므로
    # 잉여는 floor, 부족은 ceil 처리
    # -----------------------------------------------------

    surplus_units = []

    for _, row in surplus.iterrows():

        n = int(
            np.floor(row["잉여UD"])
        )

        for _ in range(n):
            surplus_units.append(
                row["출발구"]
            )

    shortage_units = []

    for _, row in shortage.iterrows():

        n = int(
            np.ceil(row["부족UD"])
        )

        for _ in range(n):
            shortage_units.append(
                row["출발구"]
            )

    if not surplus_units or not shortage_units:
        return []

    # -----------------------------------------------------
    # 6-3. 이동 비용 행렬 생성
    # -----------------------------------------------------

    cost_matrix = np.zeros(
        (
            len(surplus_units),
            len(shortage_units)
        )
    )

    for i, origin in enumerate(
        surplus_units
    ):

        for j, destination in enumerate(
            shortage_units
        ):

            if origin == destination:
                cost = 0

            else:
                cost = travel_dict.get(
                    (origin, destination),
                    DEFAULT_TIME
                )

            cost_matrix[i, j] = cost

    # -----------------------------------------------------
    # 6-4. 최소 비용 매칭
    # -----------------------------------------------------

    row_idx, col_idx = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    result = []

    for i, j in zip(
        row_idx,
        col_idx
    ):

        origin = surplus_units[i]
        destination = shortage_units[j]

        result.append(
            {
                "요일":
                    group["요일"].iloc[0],

                "승차시간대":
                    group["승차시간대"].iloc[0],

                "재배치출발구":
                    origin,

                "재배치도착구":
                    destination,

                "재배치이동시간_분":
                    cost_matrix[i, j]
            }
        )

    return result


# =========================================================
# 7. 요일 × 시간대별 재배치 최적화
# =========================================================

results = []

for (day, hour), group in stock.groupby(
    ["요일", "승차시간대"]
):

    matches = optimize_rebalancing(
        group
    )

    results.extend(matches)


rebalancing = pd.DataFrame(
    results
)


# =========================================================
# 8. 재배치 결과가 없는 경우
# =========================================================

if rebalancing.empty:

    print(
        "\n재배치 가능한 차량이 없습니다."
    )

    raise SystemExit


# =========================================================
# 9. 동일 재배치 경로 통합
# =========================================================
# 예:
# 노원구 → 강북구 차량이 3대라면
# 3개 행이 아니라 1개 행으로 집계
# =========================================================

rebalancing_summary = (
    rebalancing
    .groupby(
        [
            "요일",
            "승차시간대",
            "재배치출발구",
            "재배치도착구"
        ],
        as_index=False
    )
    .agg(
        재배치대수=(
            "재배치이동시간_분",
            "size"
        ),

        평균재배치시간_분=(
            "재배치이동시간_분",
            "mean"
        )
    )
)

rebalancing_summary[
    "총재배치시간_분"
] = (
    rebalancing_summary[
        "재배치대수"
    ]
    * rebalancing_summary[
        "평균재배치시간_분"
    ]
)


# =========================================================
# 10. 시간대별 재배치 성능
# =========================================================

hourly_rebalancing = (
    rebalancing_summary
    .groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        재배치대수=(
            "재배치대수",
            "sum"
        ),

        총재배치시간_분=(
            "총재배치시간_분",
            "sum"
        )
    )
)

hourly_rebalancing[
    "평균재배치시간_분"
] = (
    hourly_rebalancing[
        "총재배치시간_분"
    ]
    / hourly_rebalancing[
        "재배치대수"
    ]
)


# =========================================================
# 11. 실제 부족 / 잉여량 집계
# =========================================================

required = (
    stock
    .groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        필요재배치UD=(
            "부족UD",
            "sum"
        ),

        총잉여UD=(
            "잉여UD",
            "sum"
        ),

        목표UD=(
            "목표UD",
            "sum"
        ),

        시간대시작UD=(
            "시간대시작UD",
            "max"
        )
    )
)


# =========================================================
# 12. 최적화 결과 결합
# =========================================================

hourly = required.merge(
    hourly_rebalancing,
    on=[
        "요일",
        "승차시간대"
    ],
    how="left"
)

fill_columns = [
    "재배치대수",
    "총재배치시간_분",
    "평균재배치시간_분"
]

hourly[fill_columns] = (
    hourly[fill_columns]
    .fillna(0)
)


# =========================================================
# 13. 재배치 충족률
# =========================================================

hourly["재배치충족률_%"] = np.where(
    hourly["필요재배치UD"] > 0,

    np.minimum(
        hourly["재배치대수"]
        / hourly["필요재배치UD"]
        * 100,

        100
    ),

    100
)


# =========================================================
# 14. 30분 이내 재배치 가능 여부
# =========================================================
# 30분 이내 이동 가능한 경우를
# 실제 단기 재배치 가능 차량으로 간주
# =========================================================

rebalancing["30분이내"] = (
    rebalancing[
        "재배치이동시간_분"
    ]
    <= MAX_REBALANCE_TIME
)

feasible = (
    rebalancing
    .groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        실제가능재배치UD=(
            "30분이내",
            "sum"
        )
    )
)

hourly = hourly.merge(
    feasible,
    on=[
        "요일",
        "승차시간대"
    ],
    how="left"
)

hourly[
    "실제가능재배치UD"
] = (
    hourly[
        "실제가능재배치UD"
    ]
    .fillna(0)
)


# =========================================================
# 15. 시간 제약 충족률
# =========================================================

hourly[
    "시간제약충족률_%"
] = np.where(

    hourly[
        "필요재배치UD"
    ] > 0,

    np.minimum(
        hourly[
            "실제가능재배치UD"
        ]
        / hourly[
            "필요재배치UD"
        ]
        * 100,

        100
    ),

    100
)


# =========================================================
# 16. 결과 확인
# =========================================================

print(
    "\n========== 최적 재배치 경로 TOP 30 =========="
)

route_cols = [
    "요일",
    "승차시간대",
    "재배치출발구",
    "재배치도착구",
    "재배치대수",
    "평균재배치시간_분",
    "총재배치시간_분"
]

print(
    rebalancing_summary
    .sort_values(
        "총재배치시간_분",
        ascending=False
    )
    .head(30)[route_cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 17. 시간대별 결과
# =========================================================

print(
    "\n========== 시간대별 재배치 성능 TOP 30 =========="
)

hour_cols = [
    "요일",
    "승차시간대",
    "목표UD",
    "시간대시작UD",
    "필요재배치UD",
    "총잉여UD",
    "재배치대수",
    "실제가능재배치UD",
    "평균재배치시간_분",
    "재배치충족률_%",
    "시간제약충족률_%"
]

print(
    hourly
    .sort_values(
        "필요재배치UD",
        ascending=False
    )
    .head(30)[hour_cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 18. 요일별 요약
# =========================================================

daily = (
    hourly
    .groupby(
        "요일",
        as_index=False
    )
    .agg(
        분석시간대=(
            "승차시간대",
            "size"
        ),

        평균필요재배치UD=(
            "필요재배치UD",
            "mean"
        ),

        평균잉여UD=(
            "총잉여UD",
            "mean"
        ),

        평균최적재배치UD=(
            "재배치대수",
            "mean"
        ),

        평균30분이내재배치UD=(
            "실제가능재배치UD",
            "mean"
        ),

        평균재배치시간_분=(
            "평균재배치시간_분",
            "mean"
        ),

        평균재배치충족률=(
            "재배치충족률_%",
            "mean"
        ),

        평균시간제약충족률=(
            "시간제약충족률_%",
            "mean"
        )
    )
)

daily["요일순서"] = (
    daily["요일"]
    .map(DAY_ORDER)
)

daily = (
    daily
    .sort_values(
        "요일순서"
    )
    .drop(
        columns="요일순서"
    )
)

print(
    "\n========== 요일별 재배치 최적화 =========="
)

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 19. 전체 요약
# =========================================================

active = hourly[
    hourly["필요재배치UD"] > 0
].copy()

print(
    "\n========== 전체 재배치 최적화 =========="
)

print(
    f"시간당 평균 필요 재배치: "
    f"{active['필요재배치UD'].mean():.2f}대"
)

print(
    f"시간당 평균 최적 재배치: "
    f"{active['재배치대수'].mean():.2f}대"
)

print(
    f"평균 재배치 이동시간: "
    f"{rebalancing['재배치이동시간_분'].mean():.2f}분"
)

print(
    f"30분 이내 재배치 비율: "
    f"{rebalancing['30분이내'].mean() * 100:.2f}%"
)

print(
    f"평균 재배치 충족률: "
    f"{active['재배치충족률_%'].mean():.2f}%"
)

print(
    f"평균 시간제약 충족률: "
    f"{active['시간제약충족률_%'].mean():.2f}%"
)


# =========================================================
# 20. 12대 제약 검증
# =========================================================

print(
    "\n========== UD 12대 제약 검증 =========="
)

print(
    "시간대시작UD 최대:",
    round(
        stock["시간대시작UD"].max(),
        2
    )
)

print(
    "12대 초과 조합:",
    (
        stock["시간대시작UD"]
        > TOTAL_UD
    ).sum()
)


# =========================================================
# 21. 저장
# =========================================================

ROUTE_OUTPUT = (
    RESULT_DIR
    / "rebalancing_routes_2025.csv"
)

HOURLY_OUTPUT = (
    RESULT_DIR
    / "rebalancing_hourly_2025.csv"
)

DAILY_OUTPUT = (
    RESULT_DIR
    / "rebalancing_daily_2025.csv"
)

rebalancing_summary.to_csv(
    ROUTE_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

hourly.to_csv(
    HOURLY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    DAILY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n========== 저장 완료 =========="
)

print(ROUTE_OUTPUT)
print(HOURLY_OUTPUT)
print(DAILY_OUTPUT)