# ============================================================
# forecast_based_rebalancing_v4.py
#
# 목적
# ------------------------------------------------------------
# LightGBM 수요예측을 이용하여 UD택시 12대를
# 서울 25개 구에 선제 배치했을 때,
# 실제 장애인콜택시 수요의 공간분포를 얼마나 잘
# 커버하는지 평가한다.
#
# 핵심 변경점 (v3 -> v4)
# ------------------------------------------------------------
# v3:
# - UD 12대가 실제 장애인콜택시 전체 수요를
#   몇 건 처리할 수 있는지 평가
# - 차량당 시간당 처리용량 가정 필요
#
# v4:
# - 장애인콜택시 수요를 UD 잠재수요 분포로 사용
# - UD 12대의 "공간 배치 품질" 평가
# - 차량당 처리용량 가정 제거
# - 테스트셋 실제값을 이용한 Bias 보정 제거
#
# 비교 정책
# ------------------------------------------------------------
# 1. 고정배치
# 2. 현재/자연배치
# 3. LightGBM 예측 기반 선제 재배치
# 4. Oracle 선제 재배치
#
# 주요 지표
# ------------------------------------------------------------
# - 수요가중 커버리지
# - 수요-차량 분포 일치도
# - L1 분포오차
# - 과소배치 수요비율
# - 재배치 차량 수
# - 공차 차량시간
# - Oracle 대비 달성률
#
# 주의
# ------------------------------------------------------------
# Oracle은 실제 미래수요를 알고 있다고 가정하는
# 이론적 상한선이며 현실 정책이 아니다.
# ============================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. 경로 / 설정
# ============================================================

BASE_DIR = Path(
    "/Users/sinhuijae/Desktop/공모전"
)

DATA_DIR = (
    BASE_DIR
    / "데이터 분석"
)

RESULT_DIR = (
    DATA_DIR
    / "results"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FORECAST_FILE = (
    RESULT_DIR
    / "demand_forecast_test_v2_2025.csv"
)

STOCK_FILE = (
    RESULT_DIR
    / "vehicle_stock_detail_2025.csv"
)

RAW_FILE = (
    DATA_DIR
    / "disabled_taxi_2025.csv"
)


TOTAL_UD = 12


TIME_LIMITS = [
    20,
    30,
    45,
    60
]


SEOUL_GU = [
    "강남구",
    "강동구",
    "강북구",
    "강서구",
    "관악구",
    "광진구",
    "구로구",
    "금천구",
    "노원구",
    "도봉구",
    "동대문구",
    "동작구",
    "마포구",
    "서대문구",
    "서초구",
    "성동구",
    "성북구",
    "송파구",
    "양천구",
    "영등포구",
    "용산구",
    "은평구",
    "종로구",
    "중구",
    "중랑구"
]


WEEKDAY_MAP = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일"
}


# ============================================================
# 1. 기본 함수
# ============================================================

def normalize_gu(x):

    if pd.isna(x):
        return None

    x = str(x).strip()

    for gu in SEOUL_GU:

        if gu in x:
            return gu

    return None


# ============================================================
# 2. Largest Remainder Method
#    수요비율 -> 정수 차량 12대
# ============================================================

def allocate_integer_vehicles(
    demand_dict,
    total_vehicles=TOTAL_UD
):

    values = np.array(
        [
            max(
                float(
                    demand_dict.get(
                        gu,
                        0
                    )
                ),
                0
            )
            for gu in SEOUL_GU
        ],
        dtype=float
    )

    total = values.sum()

    allocation = {
        gu: 0
        for gu in SEOUL_GU
    }

    if total <= 0:
        return allocation

    raw = (
        values
        / total
        * total_vehicles
    )

    base = np.floor(
        raw
    ).astype(int)

    remainder = (
        total_vehicles
        - base.sum()
    )

    fractions = (
        raw
        - base
    )

    order = np.argsort(
        -fractions
    )

    for idx in order[:remainder]:
        base[idx] += 1

    for gu, value in zip(
        SEOUL_GU,
        base
    ):

        allocation[gu] = int(value)

    return allocation


# ============================================================
# 3. 차량 재고 정규화
# ============================================================

def normalize_stock(
    stock_dict,
    total_vehicles=TOTAL_UD
):

    values = np.array(
        [
            max(
                float(
                    stock_dict.get(
                        gu,
                        0
                    )
                ),
                0
            )
            for gu in SEOUL_GU
        ],
        dtype=float
    )

    total = values.sum()

    if total <= 0:

        return {
            gu: 0.0
            for gu in SEOUL_GU
        }

    values = (
        values
        / total
        * total_vehicles
    )

    return {
        gu: float(v)
        for gu, v in zip(
            SEOUL_GU,
            values
        )
    }


# ============================================================
# 4. 재배치
# ============================================================

def rebalance(
    current_stock,
    desired_target,
    travel_time_dict,
    time_limit,
    date=None,
    hour=None,
    policy=None
):

    stock = {
        gu: float(
            current_stock.get(
                gu,
                0
            )
        )
        for gu in SEOUL_GU
    }

    desired = {
        gu: float(
            desired_target.get(
                gu,
                0
            )
        )
        for gu in SEOUL_GU
    }

    surplus = {}
    shortage = {}

    for gu in SEOUL_GU:

        diff = (
            stock[gu]
            - desired[gu]
        )

        if diff > 1e-9:

            surplus[gu] = diff

        elif diff < -1e-9:

            shortage[gu] = -diff


    candidates = []

    for origin in surplus:

        for destination in shortage:

            travel_time = (
                travel_time_dict.get(
                    (
                        origin,
                        destination
                    ),
                    np.inf
                )
            )

            if (
                np.isfinite(
                    travel_time
                )
                and
                travel_time <= time_limit
            ):

                candidates.append(
                    (
                        travel_time,
                        origin,
                        destination
                    )
                )


    candidates.sort(
        key=lambda x: x[0]
    )


    routes = []


    for (
        travel_time,
        origin,
        destination
    ) in candidates:

        available = (
            surplus.get(
                origin,
                0
            )
        )

        needed = (
            shortage.get(
                destination,
                0
            )
        )

        move = min(
            available,
            needed
        )

        if move <= 1e-9:
            continue

        stock[origin] -= move
        stock[destination] += move

        surplus[origin] -= move
        shortage[destination] -= move

        routes.append({

            "날짜":
                date,

            "시간대":
                hour,

            "정책":
                policy,

            "시간제한_분":
                time_limit,

            "재배치출발구":
                origin,

            "재배치도착구":
                destination,

            "재배치UD":
                move,

            "이동시간_분":
                travel_time,

            "공차차량시간_분":
                move
                * travel_time
        })


    total_moved = sum(
        r["재배치UD"]
        for r in routes
    )

    empty_vehicle_minutes = sum(
        r["공차차량시간_분"]
        for r in routes
    )


    return (
        stock,
        total_moved,
        empty_vehicle_minutes,
        routes
    )


# ============================================================
# 5. 배치 품질 평가
# ============================================================

def evaluate_allocation(
    stock,
    actual_demand
):

    """
    실제 수요의 공간분포와
    UD 차량 공간분포의 일치도를 평가한다.

    핵심:

    demand_share
        실제 수요의 지역별 비율

    vehicle_share
        UD 차량의 지역별 비율

    matched_share
        min(demand_share, vehicle_share)

    coverage
        모든 지역의 matched_share 합

    값이 1이면 차량 분포와 실제 수요 분포가
    완벽하게 일치한다.
    """


    demand_values = np.array(
        [
            max(
                float(
                    actual_demand.get(
                        gu,
                        0
                    )
                ),
                0
            )
            for gu in SEOUL_GU
        ],
        dtype=float
    )


    stock_values = np.array(
        [
            max(
                float(
                    stock.get(
                        gu,
                        0
                    )
                ),
                0
            )
            for gu in SEOUL_GU
        ],
        dtype=float
    )


    total_demand = (
        demand_values.sum()
    )

    total_stock = (
        stock_values.sum()
    )


    if total_demand <= 0:

        return {

            "실제총수요":
                0.0,

            "수요가중커버리지_%":
                100.0,

            "분포일치도_%":
                100.0,

            "L1분포오차":
                0.0,

            "과소배치수요비율_%":
                0.0,

            "수요발생지역커버율_%":
                100.0,

            "차량없는수요비율_%":
                0.0
        }


    demand_share = (
        demand_values
        / total_demand
    )


    if total_stock > 0:

        vehicle_share = (
            stock_values
            / total_stock
        )

    else:

        vehicle_share = (
            np.zeros(
                len(SEOUL_GU)
            )
        )


    # --------------------------------------------------------
    # 분포 일치량
    # --------------------------------------------------------

    matched_share = (
        np.minimum(
            demand_share,
            vehicle_share
        )
    )


    distribution_match = (
        matched_share.sum()
    )


    # --------------------------------------------------------
    # L1 distance
    #
    # 완전 일치 = 0
    # 완전 불일치 = 최대 2
    # --------------------------------------------------------

    l1_error = (
        np.abs(
            demand_share
            - vehicle_share
        ).sum()
    )


    # --------------------------------------------------------
    # 과소배치된 수요비율
    #
    # 수요비율 > 차량비율인 지역에서
    # 부족한 비율의 합
    # --------------------------------------------------------

    underallocation = (
        np.maximum(
            demand_share
            - vehicle_share,
            0
        ).sum()
    )


    # --------------------------------------------------------
    # 차량이 최소 1대라도 존재하는 지역에서
    # 발생한 실제 수요 비율
    # --------------------------------------------------------

    covered_mask = (
        stock_values
        > 1e-9
    )


    geographic_coverage = (
        demand_share[
            covered_mask
        ].sum()
    )


    zero_vehicle_demand = (
        demand_share[
            ~covered_mask
        ].sum()
    )


    # --------------------------------------------------------
    # 수요가중 커버리지
    #
    # 핵심 정책 지표
    #
    # 차량 분포와 실제 수요 분포가
    # 얼마나 일치하는지 0~100으로 표현
    # --------------------------------------------------------

    weighted_coverage = (
        distribution_match
        * 100
    )


    return {

        "실제총수요":
            total_demand,

        "수요가중커버리지_%":
            weighted_coverage,

        "분포일치도_%":
            distribution_match
            * 100,

        "L1분포오차":
            l1_error,

        "과소배치수요비율_%":
            underallocation
            * 100,

        "수요발생지역커버율_%":
            geographic_coverage
            * 100,

        "차량없는수요비율_%":
            zero_vehicle_demand
            * 100
    }


# ============================================================
# 6. LightGBM 예측 데이터
# ============================================================

print(
    "\n========== "
    "LightGBM v2 예측 데이터 "
    "=========="
)


forecast = pd.read_csv(
    FORECAST_FILE,
    encoding="utf-8-sig"
)


forecast["날짜"] = pd.to_datetime(
    forecast["날짜"]
)


forecast["출발구"] = (
    forecast["출발구"]
    .apply(
        normalize_gu
    )
)


forecast = forecast[
    forecast["출발구"].isin(
        SEOUL_GU
    )
].copy()


forecast["수요"] = pd.to_numeric(
    forecast["수요"],
    errors="coerce"
).fillna(0)


forecast["예측수요"] = pd.to_numeric(
    forecast["예측수요"],
    errors="coerce"
).fillna(0)


forecast["예측수요"] = (
    forecast["예측수요"]
    .clip(
        lower=0
    )
)


print(
    f"데이터 수: "
    f"{len(forecast):,}개"
)

print(
    "기간:",
    forecast["날짜"].min().date(),
    "~",
    forecast["날짜"].max().date()
)


print(
    "\n※ v4에서는 테스트셋 실제값을 이용한 "
    "Bias 보정을 사용하지 않습니다."
)


# ============================================================
# 7. 예측 자체 성능 확인
# ============================================================

actual_total = (
    forecast["수요"].sum()
)

pred_total = (
    forecast["예측수요"].sum()
)


print(
    "\n========== 예측 총량 확인 =========="
)

print(
    f"실제 총수요: "
    f"{actual_total:,.2f}"
)

print(
    f"예측 총수요: "
    f"{pred_total:,.2f}"
)

print(
    "예측/실제 비율:",
    round(
        pred_total
        / actual_total,
        4
    )
    if actual_total > 0
    else np.nan
)


# ============================================================
# 8. 이동시간 데이터
# ============================================================

print(
    "\n========== 이동시간 데이터 =========="
)


required_raw = [
    "승차일시",
    "하차일시",
    "출발구",
    "목적구"
]


raw = pd.read_csv(
    RAW_FILE,
    usecols=required_raw,
    low_memory=False
)


raw["승차일시"] = pd.to_datetime(
    raw["승차일시"],
    errors="coerce"
)


raw["하차일시"] = pd.to_datetime(
    raw["하차일시"],
    errors="coerce"
)


raw["출발구"] = (
    raw["출발구"]
    .apply(
        normalize_gu
    )
)


raw["목적구"] = (
    raw["목적구"]
    .apply(
        normalize_gu
    )
)


raw = raw[
    raw["승차일시"].notna()
    &
    raw["하차일시"].notna()
    &
    raw["출발구"].isin(
        SEOUL_GU
    )
    &
    raw["목적구"].isin(
        SEOUL_GU
    )
].copy()


raw["운행시간_분"] = (
    raw["하차일시"]
    -
    raw["승차일시"]
).dt.total_seconds() / 60


raw = raw[
    (
        raw["운행시간_분"]
        > 0
    )
    &
    (
        raw["운행시간_분"]
        <= 180
    )
].copy()


default_travel_time = (
    raw["운행시간_분"]
    .median()
)


travel = (
    raw
    .groupby(
        [
            "출발구",
            "목적구"
        ],
        as_index=False
    )[
        "운행시간_분"
    ]
    .median()
)


travel_time_dict = {

    (
        row["출발구"],
        row["목적구"]
    ):

    float(
        row["운행시간_분"]
    )

    for _, row
    in travel.iterrows()
}


# 같은 구 이동은 재배치 불필요
for gu in SEOUL_GU:

    travel_time_dict[
        (
            gu,
            gu
        )
    ] = 0.0


print(
    f"분석 운행: "
    f"{len(raw):,}건"
)

print(
    f"구간 조합: "
    f"{len(travel):,}개"
)

print(
    f"기본 이동시간: "
    f"{default_travel_time:.2f}분"
)


# ============================================================
# 9. 차량 재고 데이터
# ============================================================

print(
    "\n========== 차량 재고 데이터 =========="
)


stock_df = pd.read_csv(
    STOCK_FILE,
    encoding="utf-8-sig"
)


required_stock = [
    "요일",
    "승차시간대",
    "출발구",
    "시간대시작UD"
]


missing_stock = [
    c
    for c in required_stock
    if c not in stock_df.columns
]


if missing_stock:

    raise ValueError(
        "vehicle_stock_detail_2025.csv에 "
        f"필요 컬럼이 없습니다: "
        f"{missing_stock}"
    )


stock_df["출발구"] = (
    stock_df["출발구"]
    .apply(
        normalize_gu
    )
)


stock_df[
    "시간대시작UD"
] = pd.to_numeric(
    stock_df["시간대시작UD"],
    errors="coerce"
).fillna(0)


print(
    f"분석 조합: "
    f"{len(stock_df):,}개"
)


# ============================================================
# 10. 요일 × 시간대 초기 재고
# ============================================================

stock_lookup = {}


for (
    day,
    hour
), group in stock_df.groupby(
    [
        "요일",
        "승차시간대"
    ]
):

    temp = {
        gu: 0.0
        for gu in SEOUL_GU
    }


    for _, row in group.iterrows():

        gu = row["출발구"]

        if gu in temp:

            temp[gu] += float(
                row["시간대시작UD"]
            )


    temp = normalize_stock(
        temp
    )


    stock_lookup[
        (
            day,
            int(hour)
        )
    ] = temp


print(
    f"초기 재고 조합: "
    f"{len(stock_lookup):,}개"
)


# ============================================================
# 11. 날짜 × 시간대별 수요 생성
# ============================================================

print(
    "\n========== "
    "날짜 × 시간대 실제/예측 수요 생성 "
    "=========="
)


time_groups = []


for (
    date,
    hour
), group in forecast.groupby(
    [
        "날짜",
        "시간대"
    ]
):

    actual_demand = {
        gu: 0.0
        for gu in SEOUL_GU
    }

    predicted_demand = {
        gu: 0.0
        for gu in SEOUL_GU
    }


    for _, row in group.iterrows():

        gu = row["출발구"]

        actual_demand[
            gu
        ] += float(
            row["수요"]
        )

        predicted_demand[
            gu
        ] += float(
            row["예측수요"]
        )


    actual_target = (
        allocate_integer_vehicles(
            actual_demand
        )
    )


    predicted_target = (
        allocate_integer_vehicles(
            predicted_demand
        )
    )


    day = WEEKDAY_MAP[
        date.weekday()
    ]


    time_groups.append({

        "날짜":
            date,

        "요일":
            day,

        "시간대":
            int(hour),

        "actual_demand":
            actual_demand,

        "predicted_demand":
            predicted_demand,

        "actual_target":
            actual_target,

        "predicted_target":
            predicted_target,

        "실제총수요":
            sum(
                actual_demand.values()
            ),

        "예측총수요":
            sum(
                predicted_demand.values()
            )
    })


print(
    f"분석 시간대: "
    f"{len(time_groups):,}개"
)


# ============================================================
# 12. 고정배치
# ============================================================

overall_demand = (
    forecast
    .groupby(
        "출발구"
    )["수요"]
    .sum()
    .to_dict()
)


fixed_allocation = (
    allocate_integer_vehicles(
        overall_demand
    )
)


print(
    "\n========== 고정배치 =========="
)


for gu in SEOUL_GU:

    if fixed_allocation[gu] > 0:

        print(
            f"{gu}: "
            f"{fixed_allocation[gu]}대"
        )


print(
    "총 차량:",
    sum(
        fixed_allocation.values()
    )
)


# ============================================================
# 13. 정책 시뮬레이션
# ============================================================

print(
    "\n========== 정책 시뮬레이션 =========="
)


result_rows = []

route_rows = []


def append_result(
    date,
    day,
    hour,
    policy,
    limit,
    evaluation,
    predicted_total,
    moved=0.0,
    empty_minutes=0.0
):

    result_rows.append({

        "날짜":
            date,

        "요일":
            day,

        "시간대":
            hour,

        "정책":
            policy,

        "시간제한_분":
            limit,

        "실제총수요":
            evaluation[
                "실제총수요"
            ],

        "예측총수요":
            predicted_total,

        "수요가중커버리지_%":
            evaluation[
                "수요가중커버리지_%"
            ],

        "분포일치도_%":
            evaluation[
                "분포일치도_%"
            ],

        "L1분포오차":
            evaluation[
                "L1분포오차"
            ],

        "과소배치수요비율_%":
            evaluation[
                "과소배치수요비율_%"
            ],

        "수요발생지역커버율_%":
            evaluation[
                "수요발생지역커버율_%"
            ],

        "차량없는수요비율_%":
            evaluation[
                "차량없는수요비율_%"
            ],

        "재배치UD":
            moved,

        "공차차량시간_분":
            empty_minutes
    })


for item in time_groups:

    date = item["날짜"]

    day = item["요일"]

    hour = item["시간대"]

    actual_demand = (
        item["actual_demand"]
    )

    predicted_target = (
        item["predicted_target"]
    )

    actual_target = (
        item["actual_target"]
    )

    predicted_total_demand = (
        item["예측총수요"]
    )


    # --------------------------------------------------------
    # 현재/자연 재고
    # --------------------------------------------------------

    key = (
        day,
        hour
    )


    if key in stock_lookup:

        initial_stock = {

            gu:
                float(
                    stock_lookup[
                        key
                    ].get(
                        gu,
                        0
                    )
                )

            for gu in SEOUL_GU
        }

    else:

        initial_stock = {

            gu:
                float(
                    fixed_allocation[
                        gu
                    ]
                )

            for gu in SEOUL_GU
        }


    # ========================================================
    # A. 고정배치
    # ========================================================

    fixed_eval = (
        evaluate_allocation(
            fixed_allocation,
            actual_demand
        )
    )


    append_result(
        date,
        day,
        hour,
        "고정배치",
        0,
        fixed_eval,
        predicted_total_demand
    )


    # ========================================================
    # B. 현재 / 자연배치
    # ========================================================

    natural_eval = (
        evaluate_allocation(
            initial_stock,
            actual_demand
        )
    )


    append_result(
        date,
        day,
        hour,
        "현재배치",
        0,
        natural_eval,
        predicted_total_demand
    )


    # ========================================================
    # C. 시간제약별 선제 재배치
    # ========================================================

    for limit in TIME_LIMITS:


        # ----------------------------------------------------
        # LightGBM 선제
        # ----------------------------------------------------

        (
            forecast_stock,
            forecast_move,
            forecast_empty_minutes,
            forecast_routes
        ) = rebalance(

            initial_stock,
            predicted_target,
            travel_time_dict,
            limit,

            date=date,
            hour=hour,

            policy=
                "LightGBM선제"
        )


        forecast_eval = (
            evaluate_allocation(
                forecast_stock,
                actual_demand
            )
        )


        append_result(
            date,
            day,
            hour,
            "LightGBM선제",
            limit,
            forecast_eval,
            predicted_total_demand,
            forecast_move,
            forecast_empty_minutes
        )


        route_rows.extend(
            forecast_routes
        )


        # ----------------------------------------------------
        # Oracle
        # ----------------------------------------------------

        (
            oracle_stock,
            oracle_move,
            oracle_empty_minutes,
            oracle_routes
        ) = rebalance(

            initial_stock,
            actual_target,
            travel_time_dict,
            limit,

            date=date,
            hour=hour,

            policy=
                "Oracle선제"
        )


        oracle_eval = (
            evaluate_allocation(
                oracle_stock,
                actual_demand
            )
        )


        append_result(
            date,
            day,
            hour,
            "Oracle선제",
            limit,
            oracle_eval,
            predicted_total_demand,
            oracle_move,
            oracle_empty_minutes
        )


        route_rows.extend(
            oracle_routes
        )


# ============================================================
# 14. DataFrame
# ============================================================

results = pd.DataFrame(
    result_rows
)


routes = pd.DataFrame(
    route_rows
)


# ============================================================
# 15. 수요가중 정책 성능 요약
# ============================================================

def weighted_average(
    group,
    value_column,
    weight_column="실제총수요"
):

    values = (
        group[value_column]
        .astype(float)
        .to_numpy()
    )

    weights = (
        group[weight_column]
        .astype(float)
        .to_numpy()
    )

    weight_sum = (
        weights.sum()
    )

    if weight_sum <= 0:

        return (
            np.nanmean(
                values
            )
        )

    return (
        np.average(
            values,
            weights=weights
        )
    )


summary_rows = []


for (
    policy,
    limit
), group in results.groupby(
    [
        "정책",
        "시간제한_분"
    ]
):

    summary_rows.append({

        "정책":
            policy,

        "시간제한_분":
            limit,

        "분석시간대":
            len(group),

        "총실제수요":
            group[
                "실제총수요"
            ].sum(),

        "수요가중커버리지_%":
            weighted_average(
                group,
                "수요가중커버리지_%"
            ),

        "분포일치도_%":
            weighted_average(
                group,
                "분포일치도_%"
            ),

        "평균L1분포오차":
            weighted_average(
                group,
                "L1분포오차"
            ),

        "과소배치수요비율_%":
            weighted_average(
                group,
                "과소배치수요비율_%"
            ),

        "수요발생지역커버율_%":
            weighted_average(
                group,
                "수요발생지역커버율_%"
            ),

        "차량없는수요비율_%":
            weighted_average(
                group,
                "차량없는수요비율_%"
            ),

        "총재배치UD":
            group[
                "재배치UD"
            ].sum(),

        "평균재배치UD":
            group[
                "재배치UD"
            ].mean(),

        "총공차차량시간_분":
            group[
                "공차차량시간_분"
            ].sum()
    })


summary = pd.DataFrame(
    summary_rows
)


policy_order = {

    "고정배치": 0,

    "현재배치": 1,

    "LightGBM선제": 2,

    "Oracle선제": 3
}


summary["_order"] = (
    summary["정책"]
    .map(
        policy_order
    )
)


summary = (
    summary
    .sort_values(
        [
            "_order",
            "시간제한_분"
        ]
    )
    .drop(
        columns="_order"
    )
)


# ============================================================
# 16. 전체 정책 비교
# ============================================================

print(
    "\n========== "
    "수요 공간분포 기준 정책 비교 "
    "=========="
)


print(
    summary[
        [
            "정책",
            "시간제한_분",
            "총실제수요",
            "수요가중커버리지_%",
            "수요발생지역커버율_%",
            "차량없는수요비율_%",
            "평균L1분포오차",
            "총재배치UD",
            "총공차차량시간_분"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. 60분 핵심 비교
# ============================================================

print(
    "\n========== "
    "60분 기준 핵심 비교 "
    "=========="
)


fixed_row = (
    summary[
        summary["정책"]
        ==
        "고정배치"
    ]
    .iloc[0]
)


natural_row = (
    summary[
        summary["정책"]
        ==
        "현재배치"
    ]
    .iloc[0]
)


forecast60 = (
    summary[
        (
            summary["정책"]
            ==
            "LightGBM선제"
        )
        &
        (
            summary["시간제한_분"]
            ==
            60
        )
    ]
    .iloc[0]
)


oracle60 = (
    summary[
        (
            summary["정책"]
            ==
            "Oracle선제"
        )
        &
        (
            summary["시간제한_분"]
            ==
            60
        )
    ]
    .iloc[0]
)


fixed_score = (
    fixed_row[
        "수요가중커버리지_%"
    ]
)


natural_score = (
    natural_row[
        "수요가중커버리지_%"
    ]
)


forecast_score = (
    forecast60[
        "수요가중커버리지_%"
    ]
)


oracle_score = (
    oracle60[
        "수요가중커버리지_%"
    ]
)


print(
    f"고정배치: "
    f"{fixed_score:.2f}%"
)

print(
    f"현재배치: "
    f"{natural_score:.2f}%"
)

print(
    f"LightGBM 선제: "
    f"{forecast_score:.2f}%"
)

print(
    f"Oracle 선제: "
    f"{oracle_score:.2f}%"
)


print()


print(
    "고정 → LightGBM 개선: "
    f"{forecast_score - fixed_score:+.2f}%p"
)


print(
    "현재 → LightGBM 개선: "
    f"{forecast_score - natural_score:+.2f}%p"
)


print(
    "LightGBM → Oracle Gap: "
    f"{oracle_score - forecast_score:.2f}%p"
)


# ============================================================
# 18. Oracle 개선 가능량 대비 달성률
# ============================================================

possible_improvement = (
    oracle_score
    - natural_score
)


achieved_improvement = (
    forecast_score
    - natural_score
)


if possible_improvement > 1e-9:

    oracle_achievement = (
        achieved_improvement
        /
        possible_improvement
        * 100
    )

else:

    oracle_achievement = np.nan


print(
    "\n========== "
    "Oracle 대비 성능 "
    "=========="
)


print(
    "Oracle 개선 가능량:",
    f"{possible_improvement:.2f}%p"
)


print(
    "LightGBM 실제 개선량:",
    f"{achieved_improvement:.2f}%p"
)


print(
    "Oracle 개선 가능량 대비 달성률:",
    (
        f"{oracle_achievement:.2f}%"
        if np.isfinite(
            oracle_achievement
        )
        else "계산 불가"
    )
)


# ============================================================
# 19. 시간제약별 LightGBM
# ============================================================

forecast_scenario = (
    summary[
        summary["정책"]
        ==
        "LightGBM선제"
    ][
        [
            "시간제한_분",
            "수요가중커버리지_%",
            "수요발생지역커버율_%",
            "차량없는수요비율_%",
            "평균L1분포오차",
            "총재배치UD",
            "평균재배치UD",
            "총공차차량시간_분"
        ]
    ]
    .sort_values(
        "시간제한_분"
    )
    .copy()
)


forecast_scenario[
    "이전대비개선_%p"
] = (
    forecast_scenario[
        "수요가중커버리지_%"
    ]
    .diff()
)


forecast_scenario[
    "추가재배치UD"
] = (
    forecast_scenario[
        "총재배치UD"
    ]
    .diff()
)


forecast_scenario[
    "추가공차시간_분"
] = (
    forecast_scenario[
        "총공차차량시간_분"
    ]
    .diff()
)


print(
    "\n========== "
    "LightGBM 시간제약별 성능 "
    "=========="
)


print(
    forecast_scenario
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 20. 시간대별 성능
# ============================================================

hour_rows = []


for (
    policy,
    limit,
    hour
), group in results.groupby(
    [
        "정책",
        "시간제한_분",
        "시간대"
    ]
):

    hour_rows.append({

        "정책":
            policy,

        "시간제한_분":
            limit,

        "시간대":
            hour,

        "분석일수":
            group[
                "날짜"
            ].nunique(),

        "총실제수요":
            group[
                "실제총수요"
            ].sum(),

        "수요가중커버리지_%":
            weighted_average(
                group,
                "수요가중커버리지_%"
            ),

        "수요발생지역커버율_%":
            weighted_average(
                group,
                "수요발생지역커버율_%"
            ),

        "차량없는수요비율_%":
            weighted_average(
                group,
                "차량없는수요비율_%"
            ),

        "평균재배치UD":
            group[
                "재배치UD"
            ].mean()
    })


by_hour = pd.DataFrame(
    hour_rows
)


hour60 = by_hour[
    (
        by_hour["정책"]
        ==
        "LightGBM선제"
    )
    &
    (
        by_hour["시간제한_분"]
        ==
        60
    )
]


print(
    "\n========== "
    "60분 LightGBM 시간대별 성능 "
    "=========="
)


print(
    hour60[
        [
            "시간대",
            "분석일수",
            "총실제수요",
            "수요가중커버리지_%",
            "수요발생지역커버율_%",
            "차량없는수요비율_%",
            "평균재배치UD"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 21. 요일별 성능
# ============================================================

day_rows = []


for (
    policy,
    limit,
    day
), group in results.groupby(
    [
        "정책",
        "시간제한_분",
        "요일"
    ]
):

    day_rows.append({

        "정책":
            policy,

        "시간제한_분":
            limit,

        "요일":
            day,

        "총실제수요":
            group[
                "실제총수요"
            ].sum(),

        "수요가중커버리지_%":
            weighted_average(
                group,
                "수요가중커버리지_%"
            ),

        "수요발생지역커버율_%":
            weighted_average(
                group,
                "수요발생지역커버율_%"
            ),

        "평균재배치UD":
            group[
                "재배치UD"
            ].mean()
    })


by_day = pd.DataFrame(
    day_rows
)


day60 = by_day[
    (
        by_day["정책"]
        ==
        "LightGBM선제"
    )
    &
    (
        by_day["시간제한_분"]
        ==
        60
    )
]


print(
    "\n========== "
    "60분 LightGBM 요일별 성능 "
    "=========="
)


print(
    day60
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 22. 실질 운영시간 7~22시
# ============================================================

operating_results = results[
    results["시간대"]
    .between(
        7,
        22
    )
].copy()


operating_rows = []


for (
    policy,
    limit
), group in operating_results.groupby(
    [
        "정책",
        "시간제한_분"
    ]
):

    operating_rows.append({

        "정책":
            policy,

        "시간제한_분":
            limit,

        "총실제수요":
            group[
                "실제총수요"
            ].sum(),

        "수요가중커버리지_%":
            weighted_average(
                group,
                "수요가중커버리지_%"
            ),

        "수요발생지역커버율_%":
            weighted_average(
                group,
                "수요발생지역커버율_%"
            ),

        "차량없는수요비율_%":
            weighted_average(
                group,
                "차량없는수요비율_%"
            ),

        "총재배치UD":
            group[
                "재배치UD"
            ].sum(),

        "총공차차량시간_분":
            group[
                "공차차량시간_분"
            ].sum()
    })


operating_summary = pd.DataFrame(
    operating_rows
)


print(
    "\n========== "
    "실질 운영시간 7~22시 정책 비교 "
    "=========="
)


print(
    operating_summary
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 23. 주요 재배치 경로
# ============================================================

if not routes.empty:

    route_summary = (
        routes
        .groupby(
            [
                "정책",
                "시간제한_분",
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
                "이동시간_분",
                "mean"
            ),

            총공차차량시간_분=(
                "공차차량시간_분",
                "sum"
            )
        )
    )


    top_routes = (
        route_summary[
            (
                route_summary["정책"]
                ==
                "LightGBM선제"
            )
            &
            (
                route_summary[
                    "시간제한_분"
                ]
                ==
                60
            )
        ]
        .sort_values(
            "총재배치UD",
            ascending=False
        )
        .head(30)
    )


    print(
        "\n========== "
        "60분 LightGBM 주요 재배치 경로 TOP 30 "
        "=========="
    )


    print(
        top_routes
        .round(2)
        .to_string(
            index=False
        )
    )


else:

    route_summary = (
        pd.DataFrame()
    )


# ============================================================
# 24. 시간제약 효율성
# ============================================================

efficiency = (
    forecast_scenario.copy()
)


efficiency[
    "커버리지_per_1000공차분"
] = (
    efficiency[
        "수요가중커버리지_%"
    ]
    /
    (
        efficiency[
            "총공차차량시간_분"
        ]
        / 1000
    )
)


print(
    "\n========== "
    "재배치 비용 대비 성능 "
    "=========="
)


print(
    efficiency[
        [
            "시간제한_분",
            "수요가중커버리지_%",
            "총재배치UD",
            "총공차차량시간_분",
            "이전대비개선_%p",
            "커버리지_per_1000공차분"
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# ============================================================
# 25. 권장 시간제약 자동 탐색
# ============================================================

scenario_temp = (
    forecast_scenario
    .copy()
)


min_score = (
    scenario_temp[
        "수요가중커버리지_%"
    ].min()
)


max_score = (
    scenario_temp[
        "수요가중커버리지_%"
    ].max()
)


min_cost = (
    scenario_temp[
        "총공차차량시간_분"
    ].min()
)


max_cost = (
    scenario_temp[
        "총공차차량시간_분"
    ].max()
)


if max_score > min_score:

    scenario_temp[
        "성능정규화"
    ] = (

        scenario_temp[
            "수요가중커버리지_%"
        ]
        - min_score

    ) / (

        max_score
        - min_score

    )

else:

    scenario_temp[
        "성능정규화"
    ] = 1.0


if max_cost > min_cost:

    scenario_temp[
        "비용정규화"
    ] = (

        scenario_temp[
            "총공차차량시간_분"
        ]
        - min_cost

    ) / (

        max_cost
        - min_cost

    )

else:

    scenario_temp[
        "비용정규화"
    ] = 0.0


# 성능 70%, 재배치 비용 30%
scenario_temp[
    "정책효율점수"
] = (

    scenario_temp[
        "성능정규화"
    ]
    * 0.7

    -

    scenario_temp[
        "비용정규화"
    ]
    * 0.3
)


best_scenario = (
    scenario_temp
    .sort_values(
        "정책효율점수",
        ascending=False
    )
    .iloc[0]
)


print(
    "\n========== "
    "권장 재배치 시간제약 "
    "=========="
)


print(
    f"권장 시간제약: "
    f"{int(best_scenario['시간제한_분'])}분"
)


print(
    f"수요가중커버리지: "
    f"{best_scenario['수요가중커버리지_%']:.2f}%"
)


print(
    f"총 재배치량: "
    f"{best_scenario['총재배치UD']:.2f}대·회"
)


print(
    f"총 공차차량시간: "
    f"{best_scenario['총공차차량시간_분']:.2f}분"
)


print(
    "※ 정책효율점수는 "
    "커버리지 성능 70%, "
    "공차 이동비용 30%를 반영한 "
    "비교용 지표입니다."
)


# ============================================================
# 26. 결과 검증
# ============================================================

print(
    "\n========== 결과 검증 =========="
)


for limit in TIME_LIMITS:

    f = (
        summary[
            (
                summary["정책"]
                ==
                "LightGBM선제"
            )
            &
            (
                summary["시간제한_분"]
                ==
                limit
            )
        ][
            "수요가중커버리지_%"
        ]
        .iloc[0]
    )


    o = (
        summary[
            (
                summary["정책"]
                ==
                "Oracle선제"
            )
            &
            (
                summary["시간제한_분"]
                ==
                limit
            )
        ][
            "수요가중커버리지_%"
        ]
        .iloc[0]
    )


    print(
        f"{limit}분: "
        f"LightGBM {f:.2f}% / "
        f"Oracle {o:.2f}% / "
        f"Gap {o - f:.2f}%p"
    )


# ============================================================
# 27. 정책 해석
# ============================================================

print(
    "\n========== 정책 해석 =========="
)


if forecast_score > natural_score:

    print(
        "✓ LightGBM 예측 기반 선제 재배치는 "
        "현재 차량배치보다 실제 수요분포와의 "
        "일치도를 향상시켰습니다."
    )

else:

    print(
        "※ LightGBM 선제 재배치가 "
        "현재 차량배치보다 높은 공간 커버리지를 "
        "보이지 못했습니다."
    )


if forecast_score > fixed_score:

    print(
        "✓ LightGBM 동적 배치는 "
        "고정배치보다 높은 수요가중 커버리지를 "
        "보였습니다."
    )

else:

    print(
        "※ 고정배치 대비 개선 효과가 "
        "확인되지 않았습니다."
    )


print(
    f"Oracle과 LightGBM 차이: "
    f"{oracle_score - forecast_score:.2f}%p"
)


if np.isfinite(
    oracle_achievement
):

    print(
        "Oracle 개선 가능량 대비 "
        "LightGBM 달성률: "
        f"{oracle_achievement:.2f}%"
    )


print(
    "\n※ v4의 커버리지는 "
    "'실제 승객을 몇 명 수송했는가'가 아니라 "
    "'UD 12대를 실제 잠재수요의 공간분포에 "
    "얼마나 적절하게 배치했는가'를 의미합니다."
)


# ============================================================
# 28. 저장
# ============================================================

DETAIL_FILE = (
    RESULT_DIR
    / "allocation_detail_v4_2025.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "allocation_summary_v4_2025.csv"
)

HOUR_FILE = (
    RESULT_DIR
    / "allocation_by_hour_v4_2025.csv"
)

DAY_FILE = (
    RESULT_DIR
    / "allocation_by_day_v4_2025.csv"
)

OPERATING_FILE = (
    RESULT_DIR
    / "allocation_operating_v4_2025.csv"
)

ROUTE_FILE = (
    RESULT_DIR
    / "allocation_routes_v4_2025.csv"
)

SCENARIO_FILE = (
    RESULT_DIR
    / "allocation_time_limit_v4_2025.csv"
)

EFFICIENCY_FILE = (
    RESULT_DIR
    / "allocation_efficiency_v4_2025.csv"
)


results.to_csv(
    DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)


summary.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)


by_hour.to_csv(
    HOUR_FILE,
    index=False,
    encoding="utf-8-sig"
)


by_day.to_csv(
    DAY_FILE,
    index=False,
    encoding="utf-8-sig"
)


operating_summary.to_csv(
    OPERATING_FILE,
    index=False,
    encoding="utf-8-sig"
)


route_summary.to_csv(
    ROUTE_FILE,
    index=False,
    encoding="utf-8-sig"
)


forecast_scenario.to_csv(
    SCENARIO_FILE,
    index=False,
    encoding="utf-8-sig"
)


scenario_temp.to_csv(
    EFFICIENCY_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========== 저장 완료 =========="
)


for path in [
    DETAIL_FILE,
    SUMMARY_FILE,
    HOUR_FILE,
    DAY_FILE,
    OPERATING_FILE,
    ROUTE_FILE,
    SCENARIO_FILE,
    EFFICIENCY_FILE
]:

    print(
        path
    )


print(
    "\n========== "
    "forecast_based_rebalancing_v4 "
    "분석 완료 "
    "=========="
)