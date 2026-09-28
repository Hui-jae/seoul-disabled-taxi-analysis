from pathlib import Path
import pandas as pd
import numpy as np
import heapq


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

TRIP_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"

# 반드시 3,574개 구 단위 상세 파일
STOCK_PATH = RESULT_DIR / "vehicle_stock_detail_2025.csv"

BASELINE_PATH = RESULT_DIR / "rebalancing_hourly_v2_2025.csv"

OUTPUT_ROUTES = (
    RESULT_DIR / "predictive_rebalancing_routes_v2_2025.csv"
)

OUTPUT_HOURLY = (
    RESULT_DIR / "predictive_rebalancing_hourly_v2_2025.csv"
)

OUTPUT_DAILY = (
    RESULT_DIR / "predictive_rebalancing_daily_v2_2025.csv"
)

OUTPUT_SUMMARY = (
    RESULT_DIR / "predictive_rebalancing_summary_v2_2025.csv"
)


TOTAL_UD = 12

# 선제 재배치 최대 허용 이동시간
MAX_PREDICTIVE_TIME = 60

DAY_ORDER = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}

NEXT_DAY = {
    "월": "화",
    "화": "수",
    "수": "목",
    "목": "금",
    "금": "토",
    "토": "일",
    "일": "월"
}


# =========================================================
# 2. 이동시간 데이터 생성
# =========================================================

print("========== 이동시간 데이터 ==========")

trip_cols = [
    "출발구",
    "목적구",
    "운행시간_분"
]

trip = pd.read_csv(
    TRIP_PATH,
    encoding="utf-8-sig",
    usecols=trip_cols
)

trip["운행시간_분"] = pd.to_numeric(
    trip["운행시간_분"],
    errors="coerce"
)

trip = trip.dropna(
    subset=[
        "출발구",
        "목적구",
        "운행시간_분"
    ]
).copy()

# 비정상 운행시간 제거
trip = trip[
    (trip["운행시간_분"] > 0)
    & (trip["운행시간_분"] <= 180)
].copy()

print(
    f"분석 운행: {len(trip):,}건"
)


# =========================================================
# 3. 구간별 평균 이동시간
# =========================================================

travel_time = (
    trip.groupby(
        ["출발구", "목적구"],
        as_index=False
    )
    .agg(
        평균이동시간_분=(
            "운행시간_분",
            "mean"
        ),
        운행건수=(
            "운행시간_분",
            "size"
        )
    )
)

GLOBAL_TRAVEL_TIME = (
    trip["운행시간_분"].median()
)

print(
    f"구간 조합: {len(travel_time):,}개"
)

print(
    f"기본 이동시간: "
    f"{GLOBAL_TRAVEL_TIME:.2f}분"
)


# =========================================================
# 4. 이동시간 lookup 생성
# =========================================================

travel_lookup = {
    (row["출발구"], row["목적구"]):
        row["평균이동시간_분"]

    for _, row in travel_time.iterrows()
}


def get_travel_time(origin, destination):

    # 같은 구 내 이동
    if origin == destination:

        value = travel_lookup.get(
            (origin, destination),
            GLOBAL_TRAVEL_TIME
        )

        return value

    return travel_lookup.get(
        (origin, destination),
        GLOBAL_TRAVEL_TIME
    )


# =========================================================
# 5. 차량 재고 데이터
# =========================================================

print("\n========== 차량 재고 데이터 ==========")

stock = pd.read_csv(
    STOCK_PATH,
    encoding="utf-8-sig"
)

required_cols = [
    "요일",
    "승차시간대",
    "출발구",
    "목표UD",
    "시간대시작UD",
    "자연충족UD",
    "부족UD",
    "잉여UD"
]

missing = [
    col for col in required_cols
    if col not in stock.columns
]

if missing:

    raise ValueError(
        f"차량 재고 상세 파일에 "
        f"다음 컬럼이 없습니다: {missing}"
    )

print(
    f"분석 조합: {len(stock):,}개"
)


# =========================================================
# 6. 다음 시간대 key 생성
# =========================================================

stock["다음요일"] = stock["요일"]
stock["다음시간대"] = (
    stock["승차시간대"] + 1
)

mask = (
    stock["다음시간대"] >= 24
)

stock.loc[
    mask,
    "다음시간대"
] = 0

stock.loc[
    mask,
    "다음요일"
] = (
    stock.loc[
        mask,
        "요일"
    ].map(NEXT_DAY)
)


# =========================================================
# 7. 다음 시간대 예상 부족량 생성
# =========================================================
# 현재 단계에서는 ML 예측이 아니라
# 2025 요일 × 시간대 × 지역 패턴 기반 예상 부족량
#
# 따라서 결과는 실제 ML 정책 성능이 아니라
# 패턴 기반 선제 재배치 잠재효과로 해석해야 함

future_need = stock[
    [
        "요일",
        "승차시간대",
        "출발구",
        "부족UD"
    ]
].copy()

future_need = future_need.rename(
    columns={
        "요일": "다음요일",
        "승차시간대": "다음시간대",
        "출발구": "부족구",
        "부족UD": "예상부족UD"
    }
)


# =========================================================
# 8. 현재 잉여 차량 생성
# =========================================================

current_surplus = stock[
    [
        "요일",
        "승차시간대",
        "출발구",
        "잉여UD",
        "다음요일",
        "다음시간대"
    ]
].copy()

current_surplus = current_surplus[
    current_surplus["잉여UD"] > 0
].copy()


# =========================================================
# 9. 최소비용 Flow 알고리즘
# =========================================================
# 목적:
#
# 현재 시간대의 잉여 UD를
# 다음 시간대 부족 지역으로 미리 이동
#
# 제약:
# 1) 출발구 잉여량 이하
# 2) 도착구 예상 부족량 이하
# 3) 이동시간 <= 60분
#
# 목적함수:
# 총 재배치 이동시간 최소화
#
# fractional vehicle을 허용하여
# 기존 기대값 기반 분석과 일관성 유지


def min_cost_predictive_flow(
    surplus_df,
    demand_df
):

    if surplus_df.empty or demand_df.empty:
        return []

    surplus = {
        row["출발구"]: float(row["잉여UD"])
        for _, row in surplus_df.iterrows()
        if row["잉여UD"] > 0
    }

    demand = {
        row["부족구"]: float(row["예상부족UD"])
        for _, row in demand_df.iterrows()
        if row["예상부족UD"] > 0
    }

    if not surplus or not demand:
        return []

    # ---------------------------------------------
    # 가능한 경로를 이동시간 기준 min-heap으로 생성
    # ---------------------------------------------

    edges = []

    for origin in surplus.keys():

        for destination in demand.keys():

            travel = get_travel_time(
                origin,
                destination
            )

            if travel <= MAX_PREDICTIVE_TIME:

                heapq.heappush(
                    edges,
                    (
                        float(travel),
                        origin,
                        destination
                    )
                )

    routes = []

    # ---------------------------------------------
    # 최소 이동시간 경로부터 차량 배치
    # ---------------------------------------------

    while edges:

        travel, origin, destination = (
            heapq.heappop(edges)
        )

        available = surplus.get(
            origin,
            0
        )

        needed = demand.get(
            destination,
            0
        )

        if available <= 1e-9:
            continue

        if needed <= 1e-9:
            continue

        flow = min(
            available,
            needed
        )

        if flow <= 1e-9:
            continue

        routes.append(
            {
                "재배치출발구": origin,
                "재배치도착구": destination,
                "선제재배치UD": flow,
                "평균이동시간_분": travel,
                "차량시간_분": (
                    flow * travel
                )
            }
        )

        surplus[origin] -= flow
        demand[destination] -= flow

    return routes


# =========================================================
# 10. 시간대별 선제 재배치 실행
# =========================================================

print(
    "\n========== 공간·시간 제약 선제 재배치 =========="
)

all_routes = []


# 현재 시간대 기준으로 반복
time_keys = (
    stock[
        ["요일", "승차시간대"]
    ]
    .drop_duplicates()
)


for _, key in time_keys.iterrows():

    day = key["요일"]
    hour = int(
        key["승차시간대"]
    )

    # ---------------------------------------------
    # 현재 시간대 잉여 차량
    # ---------------------------------------------

    surplus_group = current_surplus[
        (current_surplus["요일"] == day)
        & (
            current_surplus[
                "승차시간대"
            ] == hour
        )
    ].copy()

    if surplus_group.empty:
        continue

    next_day = (
        surplus_group["다음요일"]
        .iloc[0]
    )

    next_hour = int(
        surplus_group["다음시간대"]
        .iloc[0]
    )

    # ---------------------------------------------
    # 다음 시간대 지역별 부족
    # ---------------------------------------------

    demand_group = future_need[
        (
            future_need["다음요일"]
            == next_day
        )
        & (
            future_need["다음시간대"]
            == next_hour
        )
        & (
            future_need["예상부족UD"]
            > 0
        )
    ].copy()

    if demand_group.empty:
        continue

    # ---------------------------------------------
    # 최소비용 선제 재배치
    # ---------------------------------------------

    routes = min_cost_predictive_flow(
        surplus_group,
        demand_group
    )

    for route in routes:

        route["요일"] = day
        route["승차시간대"] = hour

        route["적용요일"] = next_day
        route["적용시간대"] = next_hour

        all_routes.append(route)


routes_df = pd.DataFrame(
    all_routes
)

print(
    f"선제 재배치 경로: "
    f"{len(routes_df):,}개"
)


# =========================================================
# 11. 선제 재배치 시간대별 집계
# =========================================================

if routes_df.empty:

    predictive_hourly = pd.DataFrame(
        columns=[
            "적용요일",
            "적용시간대",
            "선제재배치UD",
            "평균선제이동시간_분"
        ]
    )

else:

    routes_df[
        "가중이동시간"
    ] = (
        routes_df["선제재배치UD"]
        * routes_df["평균이동시간_분"]
    )

    predictive_hourly = (
        routes_df.groupby(
            [
                "적용요일",
                "적용시간대"
            ],
            as_index=False
        )
        .agg(
            선제재배치UD=(
                "선제재배치UD",
                "sum"
            ),
            총선제차량시간_분=(
                "가중이동시간",
                "sum"
            )
        )
    )

    predictive_hourly[
        "평균선제이동시간_분"
    ] = np.where(
        predictive_hourly[
            "선제재배치UD"
        ] > 0,

        predictive_hourly[
            "총선제차량시간_분"
        ]
        / predictive_hourly[
            "선제재배치UD"
        ],

        0
    )


# =========================================================
# 12. 기존 사후 최소비용 재배치 로드
# =========================================================

baseline = pd.read_csv(
    BASELINE_PATH,
    encoding="utf-8-sig"
)

print(
    "\n========== 기존 사후 재배치 =========="
)

print(
    f"분석 시간대: {len(baseline):,}개"
)


# =========================================================
# 13. 선제 재배치 결과 결합
# =========================================================

result = baseline.merge(
    predictive_hourly,
    left_on=[
        "요일",
        "승차시간대"
    ],
    right_on=[
        "적용요일",
        "적용시간대"
    ],
    how="left"
)

result[
    "선제재배치UD"
] = (
    result["선제재배치UD"]
    .fillna(0)
)

result[
    "평균선제이동시간_분"
] = (
    result["평균선제이동시간_분"]
    .fillna(0)
)

result[
    "총선제차량시간_분"
] = (
    result[
        "총선제차량시간_분"
    ]
    .fillna(0)
)

result = result.drop(
    columns=[
        "적용요일",
        "적용시간대"
    ],
    errors="ignore"
)


# =========================================================
# 14. 실제 선제 효과 제한
# =========================================================
# 선제 재배치 차량이 실제 필요 재배치량보다
# 많을 수 없도록 제한

result["실제선제효과UD"] = np.minimum(
    result["선제재배치UD"],
    result["필요재배치UD"]
)


# =========================================================
# 15. 선제 재배치 후 남은 부족
# =========================================================

result[
    "선제후필요재배치UD"
] = (
    result["필요재배치UD"]
    - result["실제선제효과UD"]
).clip(lower=0)


# =========================================================
# 16. 사후 재배치 적용
# =========================================================
# 기존 v2에서 해당 시간대에 확보한
# 사후 재배치 능력을 남은 부족에 적용
#
# 선제 배치가 먼저 일부 부족을 해결했으므로
# 필요한 만큼만 사후 재배치

result[
    "선제후사후재배치UD"
] = np.minimum(
    result["최적재배치UD"],
    result["선제후필요재배치UD"]
)


# =========================================================
# 17. 최종 미충족
# =========================================================

result["최종미충족UD"] = (
    result["선제후필요재배치UD"]
    - result["선제후사후재배치UD"]
).clip(lower=0)


# =========================================================
# 18. 총 충족량
# =========================================================

result["총충족UD"] = (
    result["실제선제효과UD"]
    + result["선제후사후재배치UD"]
)


# =========================================================
# 19. 최종 충족률
# =========================================================

result[
    "선제재배치충족률_%"
] = np.where(

    result["필요재배치UD"] > 0,

    result["총충족UD"]
    / result["필요재배치UD"]
    * 100,

    100
)


# =========================================================
# 20. 기존 대비 개선
# =========================================================

result["기존미충족UD"] = (
    result["재배치후미충족UD"]
)

result["미충족감소UD"] = (
    result["기존미충족UD"]
    - result["최종미충족UD"]
)

result["미충족감소율_%"] = np.where(

    result["기존미충족UD"] > 0,

    result["미충족감소UD"]
    / result["기존미충족UD"]
    * 100,

    0
)


# =========================================================
# 21. TOP 30
# =========================================================

print(
    "\n========== 공간·시간 제약 선제 재배치 TOP 30 =========="
)

display_cols = [
    "요일",
    "승차시간대",
    "필요재배치UD",
    "실제선제효과UD",
    "평균선제이동시간_분",
    "선제후사후재배치UD",
    "총충족UD",
    "최종미충족UD",
    "재배치충족률_%",
    "선제재배치충족률_%"
]

print(
    result
    .sort_values(
        "미충족감소UD",
        ascending=False
    )
    .head(30)[display_cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 22. 주요 선제 재배치 경로
# =========================================================

if not routes_df.empty:

    route_summary = (
        routes_df.groupby(
            [
                "재배치출발구",
                "재배치도착구"
            ],
            as_index=False
        )
        .agg(
            총선제재배치UD=(
                "선제재배치UD",
                "sum"
            ),
            평균이동시간_분=(
                "평균이동시간_분",
                "mean"
            ),
            총차량시간_분=(
                "차량시간_분",
                "sum"
            )
        )
        .sort_values(
            "총선제재배치UD",
            ascending=False
        )
    )

    print(
        "\n========== 주요 선제 재배치 경로 TOP 30 =========="
    )

    print(
        route_summary
        .head(30)
        .round(2)
        .to_string(index=False)
    )


# =========================================================
# 23. 취약 시간대 개선
# =========================================================

weak_hour = (
    result[
        result["필요재배치UD"] > 0
    ]
    .groupby(
        "승차시간대",
        as_index=False
    )
    .agg(
        발생요일수=(
            "요일",
            "nunique"
        ),
        평균필요재배치=(
            "필요재배치UD",
            "mean"
        ),
        평균선제재배치=(
            "실제선제효과UD",
            "mean"
        ),
        기존평균미충족=(
            "기존미충족UD",
            "mean"
        ),
        선제후평균미충족=(
            "최종미충족UD",
            "mean"
        ),
        기존평균충족률=(
            "재배치충족률_%",
            "mean"
        ),
        선제평균충족률=(
            "선제재배치충족률_%",
            "mean"
        )
    )
)

weak_hour["충족률개선_%p"] = (
    weak_hour["선제평균충족률"]
    - weak_hour["기존평균충족률"]
)

print(
    "\n========== 반복 취약 시간대 개선 =========="
)

print(
    weak_hour
    .sort_values(
        "기존평균미충족",
        ascending=False
    )
    .head(15)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 24. 요일별 결과
# =========================================================

daily = (
    result[
        result["필요재배치UD"] > 0
    ]
    .groupby(
        "요일",
        as_index=False
    )
    .agg(
        분석시간대=(
            "승차시간대",
            "count"
        ),
        평균필요재배치=(
            "필요재배치UD",
            "mean"
        ),
        평균선제재배치=(
            "실제선제효과UD",
            "mean"
        ),
        평균사후재배치=(
            "선제후사후재배치UD",
            "mean"
        ),
        기존평균미충족=(
            "기존미충족UD",
            "mean"
        ),
        선제후평균미충족=(
            "최종미충족UD",
            "mean"
        ),
        기존평균충족률=(
            "재배치충족률_%",
            "mean"
        ),
        선제평균충족률=(
            "선제재배치충족률_%",
            "mean"
        )
    )
)

daily["충족률개선_%p"] = (
    daily["선제평균충족률"]
    - daily["기존평균충족률"]
)

daily["요일순서"] = (
    daily["요일"].map(
        DAY_ORDER
    )
)

daily = (
    daily
    .sort_values("요일순서")
    .drop(columns="요일순서")
)

print(
    "\n========== 요일별 공간·시간 제약 선제 재배치 =========="
)

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 25. 전체 결과
# =========================================================

active = result[
    result["필요재배치UD"] > 0
].copy()

total_need = (
    active["필요재배치UD"]
    .sum()
)

baseline_rebalanced = (
    active["최적재배치UD"]
    .sum()
)

predictive_amount = (
    active["실제선제효과UD"]
    .sum()
)

post_amount = (
    active["선제후사후재배치UD"]
    .sum()
)

total_fulfilled = (
    active["총충족UD"]
    .sum()
)

baseline_unmet = (
    active["기존미충족UD"]
    .sum()
)

final_unmet = (
    active["최종미충족UD"]
    .sum()
)


# =========================================================
# 26. 가중 충족률
# =========================================================

baseline_rate = (
    baseline_rebalanced
    / total_need
    * 100
    if total_need > 0
    else 0
)

predictive_rate = (
    total_fulfilled
    / total_need
    * 100
    if total_need > 0
    else 0
)

improvement = (
    predictive_rate
    - baseline_rate
)

unmet_reduction_rate = (
    (
        baseline_unmet
        - final_unmet
    )
    / baseline_unmet
    * 100
    if baseline_unmet > 0
    else 0
)


# =========================================================
# 27. 이동시간
# =========================================================

if predictive_amount > 0:

    avg_predictive_travel = (
        active[
            "총선제차량시간_분"
        ].sum()
        / predictive_amount
    )

else:

    avg_predictive_travel = 0


# =========================================================
# 28. 전체 출력
# =========================================================

print(
    "\n========== 전체 공간·시간 제약 선제 재배치 결과 =========="
)

print(
    f"총 필요 재배치량: "
    f"{total_need:.2f}대·회"
)

print(
    f"기존 사후 최적 재배치량: "
    f"{baseline_rebalanced:.2f}대·회"
)

print(
    f"선제 재배치량: "
    f"{predictive_amount:.2f}대·회"
)

print(
    f"선제 후 사후 재배치량: "
    f"{post_amount:.2f}대·회"
)

print(
    f"기존 총 미충족량: "
    f"{baseline_unmet:.2f}대·회"
)

print(
    f"최종 미충족량: "
    f"{final_unmet:.2f}대·회"
)

print(
    f"기존 전체 충족률: "
    f"{baseline_rate:.2f}%"
)

print(
    f"공간·시간 제약 선제 충족률: "
    f"{predictive_rate:.2f}%"
)

print(
    f"충족률 개선: "
    f"+{improvement:.2f}%p"
)

print(
    f"미충족 감소율: "
    f"{unmet_reduction_rate:.2f}%"
)

print(
    f"평균 선제 재배치 이동시간: "
    f"{avg_predictive_travel:.2f}분"
)


# =========================================================
# 29. v1 vs v2
# =========================================================

print(
    "\n========== predictive v1 vs v2 =========="
)

print(
    "predictive v1 충족률: 99.06%"
)

print(
    f"predictive v2 충족률: "
    f"{predictive_rate:.2f}%"
)

print(
    f"공간·시간 제약 반영 차이: "
    f"{predictive_rate - 99.06:+.2f}%p"
)


# =========================================================
# 30. 시간대 평균 비교
# =========================================================

baseline_mean = (
    active[
        "재배치충족률_%"
    ].mean()
)

predictive_mean = (
    active[
        "선제재배치충족률_%"
    ].mean()
)

print(
    "\n========== 시간대 평균 기준 =========="
)

print(
    f"기존 평균 충족률: "
    f"{baseline_mean:.2f}%"
)

print(
    f"선제 평균 충족률: "
    f"{predictive_mean:.2f}%"
)

print(
    f"평균 개선: "
    f"{predictive_mean - baseline_mean:+.2f}%p"
)


# =========================================================
# 31. UD 12대 제약
# =========================================================

print(
    "\n========== UD 12대 제약 검증 =========="
)

if "시간대시작UD" in active.columns:

    max_stock = (
        active[
            "시간대시작UD"
        ].max()
    )

    over = (
        active[
            "시간대시작UD"
        ] > TOTAL_UD
    ).sum()

    print(
        f"시간대 시작 UD 최대: "
        f"{max_stock:.2f}대"
    )

    print(
        f"12대 초과: {over}개"
    )


# =========================================================
# 32. Summary
# =========================================================

summary = pd.DataFrame(
    {
        "지표": [
            "총 필요 재배치량",
            "기존 사후 최적 재배치량",
            "선제 재배치량",
            "선제 후 사후 재배치량",
            "기존 총 미충족량",
            "최종 미충족량",
            "기존 전체 충족률",
            "선제 전체 충족률",
            "충족률 개선",
            "미충족 감소율",
            "평균 선제 이동시간"
        ],

        "값": [
            total_need,
            baseline_rebalanced,
            predictive_amount,
            post_amount,
            baseline_unmet,
            final_unmet,
            baseline_rate,
            predictive_rate,
            improvement,
            unmet_reduction_rate,
            avg_predictive_travel
        ]
    }
)


# =========================================================
# 33. 저장
# =========================================================

routes_df.to_csv(
    OUTPUT_ROUTES,
    index=False,
    encoding="utf-8-sig"
)

result.to_csv(
    OUTPUT_HOURLY,
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    OUTPUT_DAILY,
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

print(OUTPUT_ROUTES)
print(OUTPUT_HOURLY)
print(OUTPUT_DAILY)
print(OUTPUT_SUMMARY)