from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"
STOCK_PATH = RESULT_DIR / "vehicle_stock_detail_2025.csv"

RESULT_DIR.mkdir(exist_ok=True)

TOTAL_UD = 12

# 현실적으로 허용할 최대 재배치 시간
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
    subset=["출발구", "목적구"]
).copy()

print("========== 이동시간 데이터 ==========")
print(f"분석 운행: {len(trip):,}건")


# =========================================================
# 4. 자치구 간 이동시간 행렬
# =========================================================
# 평균보다 이상치에 강한 중앙값 사용
# =========================================================

travel_time = (
    trip
    .groupby(
        ["출발구", "목적구"],
        as_index=False
    )
    .agg(
        중앙이동시간_분=("운행시간_분", "median"),
        운행건수=("운행시간_분", "size")
    )
)

travel_dict = {
    (row["출발구"], row["목적구"]):
        row["중앙이동시간_분"]

    for _, row in travel_time.iterrows()
}

DEFAULT_TIME = trip["운행시간_분"].median()

print(f"구간 조합: {len(travel_time):,}개")
print(f"기본 이동시간: {DEFAULT_TIME:.2f}분")


# =========================================================
# 5. 차량 재고 데이터
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

missing = [
    col
    for col in required_columns
    if col not in stock.columns
]

if missing:
    raise ValueError(
        f"필요한 컬럼이 없습니다: {missing}"
    )

print("\n========== 차량 재고 데이터 ==========")
print(f"분석 조합: {len(stock):,}")


# =========================================================
# 6. 최소비용 재배치 최적화
# =========================================================
#
# 목적:
#
# min Σ 이동시간(i,j) × x(i,j)
#
# 단순히 이동시간만 최소화하면
# "아무 차량도 이동하지 않음"이 최적해가 되므로,
#
# 먼저 가능한 최대 재배치량을 계산하고
# 그 재배치량을 반드시 충족시키는 조건에서
# 총 이동시간을 최소화한다.
#
# x(i,j):
# 잉여 지역 i → 부족 지역 j로 이동하는
# 기대 UD 차량 수
#
# 30분을 초과하는 경로는 후보에서 제외
# =========================================================

def optimize_rebalancing(group):

    surplus = group[
        group["잉여UD"] > 1e-9
    ][
        ["출발구", "잉여UD"]
    ].copy()

    shortage = group[
        group["부족UD"] > 1e-9
    ][
        ["출발구", "부족UD"]
    ].copy()

    if surplus.empty or shortage.empty:
        return []

    total_surplus = surplus["잉여UD"].sum()
    total_shortage = shortage["부족UD"].sum()

    # 이 시간대에서 이론적으로 이동 가능한 최대량
    target_flow = min(
        total_surplus,
        total_shortage
    )

    if target_flow <= 1e-9:
        return []

    # -----------------------------------------------------
    # 6-1. 30분 이내 가능한 경로 생성
    # -----------------------------------------------------

    routes = []

    for _, s in surplus.iterrows():

        origin = s["출발구"]

        for _, d in shortage.iterrows():

            destination = d["출발구"]

            # 같은 구는 재배치 경로로 볼 필요 없음
            if origin == destination:
                continue

            travel = travel_dict.get(
                (origin, destination),
                DEFAULT_TIME
            )

            # 30분 초과 경로 제거
            if travel > MAX_REBALANCE_TIME:
                continue

            routes.append(
                {
                    "출발구": origin,
                    "도착구": destination,
                    "이동시간": float(travel)
                }
            )

    if not routes:
        return []

    # -----------------------------------------------------
    # 6-2. 변수
    # -----------------------------------------------------

    n = len(routes)

    # 목적함수:
    # 총 재배치 이동시간 최소화
    c = np.array(
        [
            route["이동시간"]
            for route in routes
        ],
        dtype=float
    )

    A_ub = []
    b_ub = []

    # -----------------------------------------------------
    # 6-3. 출발지별 공급 제약
    #
    # Σ_j x_ij <= 잉여UD_i
    # -----------------------------------------------------

    for _, s in surplus.iterrows():

        origin = s["출발구"]

        row = np.zeros(n)

        for k, route in enumerate(routes):

            if route["출발구"] == origin:
                row[k] = 1

        A_ub.append(row)
        b_ub.append(
            float(s["잉여UD"])
        )

    # -----------------------------------------------------
    # 6-4. 도착지별 수요 제약
    #
    # Σ_i x_ij <= 부족UD_j
    # -----------------------------------------------------

    for _, d in shortage.iterrows():

        destination = d["출발구"]

        row = np.zeros(n)

        for k, route in enumerate(routes):

            if route["도착구"] == destination:
                row[k] = 1

        A_ub.append(row)
        b_ub.append(
            float(d["부족UD"])
        )

    # -----------------------------------------------------
    # 6-5. 가능한 최대 재배치량 계산
    # -----------------------------------------------------
    #
    # 30분 제약 때문에 단순히
    # min(총잉여, 총부족)을 전부 이동할 수 없는 경우가 있다.
    #
    # 따라서 1차 LP:
    # 가능한 총 재배치량 최대화
    # -----------------------------------------------------

    max_flow_result = linprog(
        c=-np.ones(n),
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n,
        method="highs"
    )

    if not max_flow_result.success:
        return []

    max_feasible_flow = (
        max_flow_result.x.sum()
    )

    if max_feasible_flow <= 1e-9:
        return []

    # -----------------------------------------------------
    # 6-6. 최대 재배치량을 유지하면서
    # 이동시간 최소화
    # -----------------------------------------------------

    A_eq = np.ones((1, n))
    b_eq = np.array(
        [max_feasible_flow]
    )

    result = linprog(
        c=c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * n,
        method="highs"
    )

    if not result.success:
        return []

    # -----------------------------------------------------
    # 6-7. 결과 저장
    # -----------------------------------------------------

    output = []

    day = group["요일"].iloc[0]
    hour = group["승차시간대"].iloc[0]

    for route, flow in zip(
        routes,
        result.x
    ):

        if flow <= 1e-8:
            continue

        output.append(
            {
                "요일": day,
                "승차시간대": hour,
                "재배치출발구":
                    route["출발구"],
                "재배치도착구":
                    route["도착구"],
                "재배치UD":
                    flow,
                "재배치이동시간_분":
                    route["이동시간"],
                "재배치차량시간_분":
                    flow
                    * route["이동시간"]
            }
        )

    return output


# =========================================================
# 7. 요일 × 시간대별 최적화
# =========================================================

results = []

for (day, hour), group in stock.groupby(
    ["요일", "승차시간대"],
    sort=False
):

    matches = optimize_rebalancing(
        group
    )

    results.extend(matches)


rebalancing = pd.DataFrame(
    results
)

print(
    "\n========== 최소비용 재배치 최적화 =========="
)

if rebalancing.empty:

    print("가능한 재배치 경로가 없습니다.")

else:

    print(
        f"최적 재배치 경로: "
        f"{len(rebalancing):,}개"
    )


# =========================================================
# 8. 시간대별 기존 필요량
# =========================================================

required = (
    stock
    .groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        목표UD=("목표UD", "sum"),

        시간대시작UD=(
            "시간대시작UD",
            "max"
        ),

        필요재배치UD=(
            "부족UD",
            "sum"
        ),

        가용잉여UD=(
            "잉여UD",
            "sum"
        )
    )
)


# =========================================================
# 9. 시간대별 최적화 결과
# =========================================================

if not rebalancing.empty:

    hourly_opt = (
        rebalancing
        .groupby(
            ["요일", "승차시간대"],
            as_index=False
        )
        .agg(
            최적재배치UD=(
                "재배치UD",
                "sum"
            ),

            총재배치차량시간_분=(
                "재배치차량시간_분",
                "sum"
            )
        )
    )

    hourly_opt[
        "평균재배치시간_분"
    ] = (
        hourly_opt[
            "총재배치차량시간_분"
        ]
        / hourly_opt[
            "최적재배치UD"
        ]
    )

else:

    hourly_opt = pd.DataFrame(
        columns=[
            "요일",
            "승차시간대",
            "최적재배치UD",
            "총재배치차량시간_분",
            "평균재배치시간_분"
        ]
    )


# =========================================================
# 10. 결합
# =========================================================

hourly = required.merge(
    hourly_opt,
    on=["요일", "승차시간대"],
    how="left"
)

fill_cols = [
    "최적재배치UD",
    "총재배치차량시간_분",
    "평균재배치시간_분"
]

hourly[fill_cols] = (
    hourly[fill_cols]
    .fillna(0)
)


# =========================================================
# 11. 재배치 충족률
# =========================================================

hourly["재배치충족률_%"] = np.where(
    hourly["필요재배치UD"] > 0,

    np.minimum(
        hourly["최적재배치UD"]
        / hourly["필요재배치UD"]
        * 100,
        100
    ),

    100
)


# =========================================================
# 12. 재배치 후 미충족 차량
# =========================================================

hourly["재배치후미충족UD"] = (
    hourly["필요재배치UD"]
    - hourly["최적재배치UD"]
).clip(lower=0)


# =========================================================
# 13. 잉여 차량 활용률
# =========================================================

hourly["잉여활용률_%"] = np.where(
    hourly["가용잉여UD"] > 0,

    np.minimum(
        hourly["최적재배치UD"]
        / hourly["가용잉여UD"]
        * 100,
        100
    ),

    0
)


# =========================================================
# 14. 시간대별 결과 출력
# =========================================================

print(
    "\n========== 시간대별 재배치 최적화 TOP 30 =========="
)

cols = [
    "요일",
    "승차시간대",
    "목표UD",
    "시간대시작UD",
    "필요재배치UD",
    "가용잉여UD",
    "최적재배치UD",
    "재배치후미충족UD",
    "평균재배치시간_분",
    "재배치충족률_%",
    "잉여활용률_%"
]

print(
    hourly
    .sort_values(
        "필요재배치UD",
        ascending=False
    )
    .head(30)[cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 15. 최적 재배치 경로
# =========================================================

if not rebalancing.empty:

    print(
        "\n========== 주요 재배치 경로 TOP 30 =========="
    )

    route_summary = (
        rebalancing
        .groupby(
            [
                "재배치출발구",
                "재배치도착구"
            ],
            as_index=False
        )
        .agg(
            총재배치UD=(
                "재배치UD",
                "sum"
            ),

            평균이동시간_분=(
                "재배치이동시간_분",
                "mean"
            ),

            총차량시간_분=(
                "재배치차량시간_분",
                "sum"
            )
        )
    )

    print(
        route_summary
        .sort_values(
            "총재배치UD",
            ascending=False
        )
        .head(30)
        .round(2)
        .to_string(index=False)
    )

else:

    route_summary = pd.DataFrame()


# =========================================================
# 16. 요일별 요약
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

        평균가용잉여UD=(
            "가용잉여UD",
            "mean"
        ),

        평균최적재배치UD=(
            "최적재배치UD",
            "mean"
        ),

        평균미충족UD=(
            "재배치후미충족UD",
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

        평균잉여활용률=(
            "잉여활용률_%",
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
    .sort_values("요일순서")
    .drop(columns="요일순서")
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
# 17. 전체 분석
# =========================================================

active = hourly[
    hourly["필요재배치UD"] > 0
].copy()

print(
    "\n========== 전체 최소비용 재배치 결과 =========="
)

print(
    f"시간당 평균 필요 재배치: "
    f"{active['필요재배치UD'].mean():.2f}대"
)

print(
    f"시간당 평균 가용 잉여: "
    f"{active['가용잉여UD'].mean():.2f}대"
)

print(
    f"시간당 평균 최적 재배치: "
    f"{active['최적재배치UD'].mean():.2f}대"
)

print(
    f"시간당 평균 미충족: "
    f"{active['재배치후미충족UD'].mean():.2f}대"
)

print(
    f"평균 재배치 충족률: "
    f"{active['재배치충족률_%'].mean():.2f}%"
)

if not rebalancing.empty:

    weighted_mean_time = (
        rebalancing[
            "재배치차량시간_분"
        ].sum()
        / rebalancing[
            "재배치UD"
        ].sum()
    )

    print(
        f"평균 재배치 이동시간: "
        f"{weighted_mean_time:.2f}분"
    )


# =========================================================
# 18. v1과 비교
# =========================================================

V1_PATH = (
    RESULT_DIR
    / "rebalancing_hourly_2025.csv"
)

if V1_PATH.exists():

    v1 = pd.read_csv(
        V1_PATH,
        encoding="utf-8-sig"
    )

    v1_active = v1[
        v1["필요재배치UD"] > 0
    ]


    # =========================================================
# 18. v1과 비교
# =========================================================

V1_PATH = RESULT_DIR / "rebalancing_hourly_2025.csv"

if V1_PATH.exists():

    v1 = pd.read_csv(
        V1_PATH,
        encoding="utf-8-sig"
    )

    v1_active = v1[
        v1["필요재배치UD"] > 0
    ].copy()

    v1_rate = v1_active[
        "재배치충족률_%"
    ].mean()

    v2_rate = active[
        "재배치충족률_%"
    ].mean()

    rate_change = v2_rate - v1_rate

    print("\n========== v1 vs v2 ==========")

    print(
        f"v1 평균 재배치 충족률: "
        f"{v1_rate:.2f}%"
    )

    print(
        f"v2 평균 재배치 충족률: "
        f"{v2_rate:.2f}%"
    )

    print(
        f"변화: {rate_change:+.2f}%p"
    )


# =========================================================
# 19. 12대 제약 검증
# =========================================================

print(
    "\n========== UD 12대 제약 검증 =========="
)

max_stock = (
    stock
    .groupby(
        ["요일", "승차시간대"]
    )["시간대시작UD"]
    .max()
    .max()
)

print(
    f"시간대 시작 UD 최대: "
    f"{max_stock:.2f}대"
)

print(
    "12대 초과:",
    int(max_stock > TOTAL_UD)
)


# =========================================================
# 20. 저장
# =========================================================

ROUTE_OUTPUT = (
    RESULT_DIR
    / "rebalancing_routes_v2_2025.csv"
)

HOURLY_OUTPUT = (
    RESULT_DIR
    / "rebalancing_hourly_v2_2025.csv"
)

DAILY_OUTPUT = (
    RESULT_DIR
    / "rebalancing_daily_v2_2025.csv"
)

rebalancing.to_csv(
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