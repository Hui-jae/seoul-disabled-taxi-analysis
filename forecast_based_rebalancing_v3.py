# ============================================================
# forecast_based_rebalancing_v3.py
#
# 목적
# ------------------------------------------------------------
# LightGBM 수요예측을 이용한 UD택시 동적 배치 정책을
# 실제 승객 수요 관점에서 평가한다.
#
# 비교 정책
# 1. 고정배치 Baseline
# 2. 사후 재배치
#    - 실제 호출 발생 후 부족 지역으로 차량 이동
#    - 이동시간을 승객 대기시간으로 반영
# 3. LightGBM 예측 기반 선제 재배치
#    - 예측수요를 기반으로 호출 발생 전에 차량 배치
# 4. Oracle 선제 재배치
#    - 실제 미래수요를 알고 있다고 가정하는 이론적 상한선
#
# 핵심 변경점 (v2 -> v3)
# ------------------------------------------------------------
# v2:
# 실제 수요 비율을 12대 목표배치로 변환한 뒤
# 차량 위치와 목표배치의 일치도를 평가
#
# v3:
# 12대 배치는 "차량 위치 결정"에만 사용
# 서비스 성능은 실제 승객 수요를 직접 사용
#
# 주요 평가 지표
# ------------------------------------------------------------
# - 실제수요
# - 즉시충족수요
# - 지연충족수요
# - 총충족수요
# - 미충족수요
# - 수요충족률
# - 즉시충족률
# - 평균예상대기시간
# - 재배치UD
# - 공차차량시간
#
# 주의
# ------------------------------------------------------------
# 본 시뮬레이션은 "시간당 수요량"을 기반으로 한 정책 비교 모델이다.
# 개별 차량의 분 단위 연속 운행을 완전히 재현하는
# microscopic fleet simulation은 아니다.
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


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------

TOTAL_UD = 12

TIME_LIMITS = [
    20,
    30,
    45,
    60
]

# 1대의 UD 차량이 한 시간 동안 처리할 수 있는
# 평균 서비스 용량
#
# 평균 운행시간 약 20분을 고려하면
# 이론적으로 약 3건/시간이지만
# 승하차, 공차 이동, 교통약자 승하차 시간을 고려하여
# 보수적으로 2건/시간 사용
#
# 민감도 분석에서 추후 변경 가능
SERVICE_CAPACITY_PER_VEHICLE = 2.0


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


# ------------------------------------------------------------
# Largest Remainder Method
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 차량 재고 정규화
# ------------------------------------------------------------

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
# 2. 차량 재배치 함수
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
            "날짜": date,
            "시간대": hour,
            "정책": policy,
            "시간제한_분": time_limit,
            "재배치출발구": origin,
            "재배치도착구": destination,
            "재배치UD": move,
            "이동시간_분": travel_time,
            "공차차량시간_분": (
                move
                * travel_time
            )
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
# 3. 실제 승객 수요 서비스 평가
# ============================================================

def evaluate_prepositioned_service(
    stock,
    actual_demand,
    capacity_per_vehicle
):

    """
    선제 배치 정책 평가

    차량이 호출 발생 전에 해당 구에 존재한다고 가정한다.

    따라서 해당 구 차량 용량 내에서 처리되는 수요는
    즉시 서비스 가능한 수요로 본다.
    """

    immediate = 0.0
    unmet = 0.0

    by_gu = {}


    for gu in SEOUL_GU:

        vehicles = max(
            float(
                stock.get(
                    gu,
                    0
                )
            ),
            0
        )

        demand = max(
            float(
                actual_demand.get(
                    gu,
                    0
                )
            ),
            0
        )

        capacity = (
            vehicles
            * capacity_per_vehicle
        )

        served = min(
            demand,
            capacity
        )

        gu_unmet = max(
            demand - served,
            0
        )

        immediate += served
        unmet += gu_unmet

        by_gu[gu] = {
            "차량": vehicles,
            "수요": demand,
            "서비스용량": capacity,
            "즉시충족수요": served,
            "미충족수요": gu_unmet
        }


    total_demand = sum(
        actual_demand.values()
    )

    service_rate = (
        immediate
        / total_demand
        * 100
        if total_demand > 0
        else 100
    )


    return {
        "실제수요": total_demand,
        "즉시충족수요": immediate,
        "지연충족수요": 0.0,
        "총충족수요": immediate,
        "미충족수요": unmet,
        "수요충족률_%": service_rate,
        "즉시충족률_%": service_rate,
        "평균예상대기시간_분": 0.0,
        "by_gu": by_gu
    }


# ============================================================
# 4. 사후 재배치 서비스 평가
# ============================================================

def evaluate_post_rebalancing_service(
    initial_stock,
    actual_demand,
    travel_time_dict,
    time_limit,
    capacity_per_vehicle,
    date=None,
    hour=None
):

    """
    사후 재배치

    1. 현재 위치 차량으로 먼저 서비스
    2. 부족 수요가 있는 지역 확인
    3. 다른 지역의 잉여 차량을 이동
    4. 이동 차량이 처리한 승객은
       이동시간만큼 기다린 것으로 계산
    """

    stock = {
        gu: float(
            initial_stock.get(
                gu,
                0
            )
        )
        for gu in SEOUL_GU
    }


    demand = {
        gu: max(
            float(
                actual_demand.get(
                    gu,
                    0
                )
            ),
            0
        )
        for gu in SEOUL_GU
    }


    immediate_by_gu = {}

    remaining_demand = {}

    spare_vehicles = {}


    total_immediate = 0.0


    for gu in SEOUL_GU:

        local_capacity = (
            stock[gu]
            * capacity_per_vehicle
        )

        served = min(
            demand[gu],
            local_capacity
        )

        total_immediate += served

        remaining = max(
            demand[gu]
            - served,
            0
        )

        remaining_demand[gu] = remaining

        # 실제 사용된 차량 수
        vehicles_used = min(
            stock[gu],
            (
                demand[gu]
                / capacity_per_vehicle
                if capacity_per_vehicle > 0
                else 0
            )
        )

        spare = max(
            stock[gu]
            - vehicles_used,
            0
        )

        spare_vehicles[gu] = spare

        immediate_by_gu[gu] = served


    candidates = []


    for origin in SEOUL_GU:

        if spare_vehicles[origin] <= 1e-9:
            continue

        for destination in SEOUL_GU:

            if remaining_demand[destination] <= 1e-9:
                continue

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

    delayed_served = 0.0

    passenger_wait_minutes = 0.0

    total_moved = 0.0

    empty_vehicle_minutes = 0.0


    for (
        travel_time,
        origin,
        destination
    ) in candidates:

        available_vehicle = (
            spare_vehicles[origin]
        )

        remaining = (
            remaining_demand[
                destination
            ]
        )

        if (
            available_vehicle <= 1e-9
            or
            remaining <= 1e-9
        ):
            continue


        required_vehicle = (
            remaining
            / capacity_per_vehicle
        )


        move_vehicle = min(
            available_vehicle,
            required_vehicle
        )


        served_demand = min(
            remaining,
            move_vehicle
            * capacity_per_vehicle
        )


        if served_demand <= 1e-9:
            continue


        spare_vehicles[
            origin
        ] -= move_vehicle

        remaining_demand[
            destination
        ] -= served_demand


        delayed_served += (
            served_demand
        )

        total_moved += (
            move_vehicle
        )

        passenger_wait_minutes += (
            served_demand
            * travel_time
        )

        empty_vehicle_minutes += (
            move_vehicle
            * travel_time
        )


        routes.append({
            "날짜": date,
            "시간대": hour,
            "정책": "사후재배치",
            "시간제한_분": time_limit,
            "재배치출발구": origin,
            "재배치도착구": destination,
            "재배치UD": move_vehicle,
            "처리수요": served_demand,
            "이동시간_분": travel_time,
            "승객대기시간합_분": (
                served_demand
                * travel_time
            ),
            "공차차량시간_분": (
                move_vehicle
                * travel_time
            )
        })


    total_demand = sum(
        demand.values()
    )

    total_served = (
        total_immediate
        + delayed_served
    )

    unmet = max(
        total_demand
        - total_served,
        0
    )


    service_rate = (
        total_served
        / total_demand
        * 100
        if total_demand > 0
        else 100
    )


    immediate_rate = (
        total_immediate
        / total_demand
        * 100
        if total_demand > 0
        else 100
    )


    # 서비스 받은 전체 승객 기준
    # 즉시 서비스 승객의 대기시간은 0분으로 계산
    avg_wait = (
        passenger_wait_minutes
        / total_served
        if total_served > 0
        else 0
    )


    return {
        "실제수요": total_demand,
        "즉시충족수요": total_immediate,
        "지연충족수요": delayed_served,
        "총충족수요": total_served,
        "미충족수요": unmet,
        "수요충족률_%": service_rate,
        "즉시충족률_%": immediate_rate,
        "평균예상대기시간_분": avg_wait,
        "재배치UD": total_moved,
        "공차차량시간_분": empty_vehicle_minutes,
        "routes": routes
    }


# ============================================================
# 5. LightGBM 예측 데이터
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


# ============================================================
# 6. Bias 보정
# ============================================================

actual_total = (
    forecast["수요"].sum()
)

pred_total = (
    forecast["예측수요"].sum()
)


bias_factor = (
    actual_total
    / pred_total
    if pred_total > 0
    else 1
)


forecast[
    "보정예측수요"
] = (
    forecast["예측수요"]
    * bias_factor
)


print(
    "\n========== 예측 Bias =========="
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
    f"Bias 계수: "
    f"{bias_factor:.4f}"
)


# ============================================================
# 7. 이동시간 데이터
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


raw[
    "운행시간_분"
] = (
    raw["하차일시"]
    -
    raw["승차일시"]
).dt.total_seconds() / 60


raw = raw[
    (
        raw["운행시간_분"] > 0
    )
    &
    (
        raw["운행시간_분"] <= 180
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


for gu in SEOUL_GU:

    if (
        gu,
        gu
    ) not in travel_time_dict:

        travel_time_dict[
            (
                gu,
                gu
            )
        ] = default_travel_time


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
# 8. 차량 재고 데이터
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
# 9. 요일 × 시간대 초기 재고
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
# 10. 날짜 × 시간대별 수요 생성
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
            row["보정예측수요"]
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
        "날짜": date,
        "요일": day,
        "시간대": int(hour),

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
# 11. 고정배치 생성
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
# 12. 정책 시뮬레이션
# ============================================================

print(
    "\n========== 정책 시뮬레이션 =========="
)


result_rows = []

route_rows = []


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

    actual_total_demand = (
        item["실제총수요"]
    )

    predicted_total_demand = (
        item["예측총수요"]
    )


    # --------------------------------------------------------
    # 공통 초기재고
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
        evaluate_prepositioned_service(
            fixed_allocation,
            actual_demand,
            SERVICE_CAPACITY_PER_VEHICLE
        )
    )


    result_rows.append({

        "날짜": date,
        "요일": day,
        "시간대": hour,

        "정책":
            "고정배치",

        "시간제한_분": 0,

        "실제총수요":
            actual_total_demand,

        "예측총수요":
            predicted_total_demand,

        "즉시충족수요":
            fixed_eval[
                "즉시충족수요"
            ],

        "지연충족수요":
            fixed_eval[
                "지연충족수요"
            ],

        "총충족수요":
            fixed_eval[
                "총충족수요"
            ],

        "미충족수요":
            fixed_eval[
                "미충족수요"
            ],

        "수요충족률_%":
            fixed_eval[
                "수요충족률_%"
            ],

        "즉시충족률_%":
            fixed_eval[
                "즉시충족률_%"
            ],

        "평균예상대기시간_분":
            0.0,

        "재배치UD":
            0.0,

        "공차차량시간_분":
            0.0
    })


    # ========================================================
    # 시간제약별 정책
    # ========================================================

    for limit in TIME_LIMITS:


        # ====================================================
        # B. 사후 재배치
        # ====================================================

        post_eval = (
            evaluate_post_rebalancing_service(
                initial_stock,
                actual_demand,
                travel_time_dict,
                limit,
                SERVICE_CAPACITY_PER_VEHICLE,
                date=date,
                hour=hour
            )
        )


        result_rows.append({

            "날짜": date,
            "요일": day,
            "시간대": hour,

            "정책":
                "사후재배치",

            "시간제한_분":
                limit,

            "실제총수요":
                actual_total_demand,

            "예측총수요":
                predicted_total_demand,

            "즉시충족수요":
                post_eval[
                    "즉시충족수요"
                ],

            "지연충족수요":
                post_eval[
                    "지연충족수요"
                ],

            "총충족수요":
                post_eval[
                    "총충족수요"
                ],

            "미충족수요":
                post_eval[
                    "미충족수요"
                ],

            "수요충족률_%":
                post_eval[
                    "수요충족률_%"
                ],

            "즉시충족률_%":
                post_eval[
                    "즉시충족률_%"
                ],

            "평균예상대기시간_분":
                post_eval[
                    "평균예상대기시간_분"
                ],

            "재배치UD":
                post_eval[
                    "재배치UD"
                ],

            "공차차량시간_분":
                post_eval[
                    "공차차량시간_분"
                ]
        })


        route_rows.extend(
            post_eval[
                "routes"
            ]
        )


        # ====================================================
        # C. LightGBM 예측 기반 선제 재배치
        # ====================================================

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
            evaluate_prepositioned_service(
                forecast_stock,
                actual_demand,
                SERVICE_CAPACITY_PER_VEHICLE
            )
        )


        result_rows.append({

            "날짜": date,
            "요일": day,
            "시간대": hour,

            "정책":
                "LightGBM선제",

            "시간제한_분":
                limit,

            "실제총수요":
                actual_total_demand,

            "예측총수요":
                predicted_total_demand,

            "즉시충족수요":
                forecast_eval[
                    "즉시충족수요"
                ],

            "지연충족수요":
                0.0,

            "총충족수요":
                forecast_eval[
                    "총충족수요"
                ],

            "미충족수요":
                forecast_eval[
                    "미충족수요"
                ],

            "수요충족률_%":
                forecast_eval[
                    "수요충족률_%"
                ],

            "즉시충족률_%":
                forecast_eval[
                    "즉시충족률_%"
                ],

            "평균예상대기시간_분":
                0.0,

            "재배치UD":
                forecast_move,

            "공차차량시간_분":
                forecast_empty_minutes
        })


        route_rows.extend(
            forecast_routes
        )


        # ====================================================
        # D. Oracle 선제 재배치
        # ====================================================

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
            evaluate_prepositioned_service(
                oracle_stock,
                actual_demand,
                SERVICE_CAPACITY_PER_VEHICLE
            )
        )


        result_rows.append({

            "날짜": date,
            "요일": day,
            "시간대": hour,

            "정책":
                "Oracle선제",

            "시간제한_분":
                limit,

            "실제총수요":
                actual_total_demand,

            "예측총수요":
                predicted_total_demand,

            "즉시충족수요":
                oracle_eval[
                    "즉시충족수요"
                ],

            "지연충족수요":
                0.0,

            "총충족수요":
                oracle_eval[
                    "총충족수요"
                ],

            "미충족수요":
                oracle_eval[
                    "미충족수요"
                ],

            "수요충족률_%":
                oracle_eval[
                    "수요충족률_%"
                ],

            "즉시충족률_%":
                oracle_eval[
                    "즉시충족률_%"
                ],

            "평균예상대기시간_분":
                0.0,

            "재배치UD":
                oracle_move,

            "공차차량시간_분":
                oracle_empty_minutes
        })


        route_rows.extend(
            oracle_routes
        )


# ============================================================
# 13. DataFrame 생성
# ============================================================

results = pd.DataFrame(
    result_rows
)


routes = pd.DataFrame(
    route_rows
)


# ============================================================
# 14. 전체 정책 성능
# ============================================================

summary = (
    results
    .groupby(
        [
            "정책",
            "시간제한_분"
        ],
        as_index=False
    )
    .agg(

        분석시간대=(
            "시간대",
            "size"
        ),

        총실제수요=(
            "실제총수요",
            "sum"
        ),

        총즉시충족수요=(
            "즉시충족수요",
            "sum"
        ),

        총지연충족수요=(
            "지연충족수요",
            "sum"
        ),

        총충족수요=(
            "총충족수요",
            "sum"
        ),

        총미충족수요=(
            "미충족수요",
            "sum"
        ),

        총재배치UD=(
            "재배치UD",
            "sum"
        ),

        총공차차량시간_분=(
            "공차차량시간_분",
            "sum"
        )
    )
)


summary[
    "전체수요충족률_%"
] = (
    summary["총충족수요"]
    /
    summary["총실제수요"]
    * 100
)


summary[
    "전체즉시충족률_%"
] = (
    summary["총즉시충족수요"]
    /
    summary["총실제수요"]
    * 100
)


# ============================================================
# 15. 평균 대기시간
# ============================================================

wait_summary = (
    results
    .groupby(
        [
            "정책",
            "시간제한_분"
        ],
        as_index=False
    )
    .agg(
        평균예상대기시간_분=(
            "평균예상대기시간_분",
            "mean"
        )
    )
)


summary = summary.merge(
    wait_summary,
    on=[
        "정책",
        "시간제한_분"
    ],
    how="left"
)


# ============================================================
# 16. 결과 출력
# ============================================================

policy_order = {
    "고정배치": 0,
    "사후재배치": 1,
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


print(
    "\n========== "
    "실제 승객 수요 기준 정책 비교 "
    "=========="
)


print(
    summary[
        [
            "정책",
            "시간제한_분",
            "총실제수요",
            "총즉시충족수요",
            "총지연충족수요",
            "총충족수요",
            "총미충족수요",
            "전체수요충족률_%",
            "전체즉시충족률_%",
            "총재배치UD",
            "총공차차량시간_분",
            "평균예상대기시간_분"
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


post60 = (
    summary[
        (
            summary["정책"]
            ==
            "사후재배치"
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


fixed_rate = (
    fixed_row[
        "전체수요충족률_%"
    ]
)

post_rate = (
    post60[
        "전체수요충족률_%"
    ]
)

forecast_rate = (
    forecast60[
        "전체수요충족률_%"
    ]
)

oracle_rate = (
    oracle60[
        "전체수요충족률_%"
    ]
)


print(
    f"고정배치 수요충족률: "
    f"{fixed_rate:.2f}%"
)

print(
    f"사후 재배치 수요충족률: "
    f"{post_rate:.2f}%"
)

print(
    f"LightGBM 선제 수요충족률: "
    f"{forecast_rate:.2f}%"
)

print(
    f"Oracle 선제 수요충족률: "
    f"{oracle_rate:.2f}%"
)


print()

print(
    "고정 → LightGBM 개선: "
    f"{forecast_rate - fixed_rate:+.2f}%p"
)

print(
    "사후 → LightGBM 차이: "
    f"{forecast_rate - post_rate:+.2f}%p"
)

print(
    "LightGBM → Oracle Gap: "
    f"{oracle_rate - forecast_rate:.2f}%p"
)


# ============================================================
# 18. 시간대별 분석
# ============================================================

by_hour = (
    results
    .groupby(
        [
            "정책",
            "시간제한_분",
            "시간대"
        ],
        as_index=False
    )
    .agg(

        분석일수=(
            "날짜",
            "nunique"
        ),

        총실제수요=(
            "실제총수요",
            "sum"
        ),

        총즉시충족수요=(
            "즉시충족수요",
            "sum"
        ),

        총충족수요=(
            "총충족수요",
            "sum"
        ),

        총미충족수요=(
            "미충족수요",
            "sum"
        ),

        평균재배치UD=(
            "재배치UD",
            "mean"
        ),

        평균예상대기시간_분=(
            "평균예상대기시간_분",
            "mean"
        )
    )
)


by_hour[
    "수요충족률_%"
] = (
    by_hour["총충족수요"]
    /
    by_hour["총실제수요"]
    * 100
)


by_hour[
    "즉시충족률_%"
] = (
    by_hour["총즉시충족수요"]
    /
    by_hour["총실제수요"]
    * 100
)


print(
    "\n========== "
    "60분 LightGBM 시간대별 성능 "
    "=========="
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
    hour60[
        [
            "시간대",
            "분석일수",
            "총실제수요",
            "총충족수요",
            "총미충족수요",
            "수요충족률_%",
            "평균재배치UD"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. 요일별 분석
# ============================================================

by_day = (
    results
    .groupby(
        [
            "정책",
            "시간제한_분",
            "요일"
        ],
        as_index=False
    )
    .agg(

        총실제수요=(
            "실제총수요",
            "sum"
        ),

        총충족수요=(
            "총충족수요",
            "sum"
        ),

        총미충족수요=(
            "미충족수요",
            "sum"
        ),

        평균재배치UD=(
            "재배치UD",
            "mean"
        )
    )
)


by_day[
    "수요충족률_%"
] = (
    by_day["총충족수요"]
    /
    by_day["총실제수요"]
    * 100
)


# ============================================================
# 20. 실질 운영시간 7~22시
# ============================================================

operating_results = results[
    results["시간대"]
    .between(
        7,
        22
    )
].copy()


operating_summary = (
    operating_results
    .groupby(
        [
            "정책",
            "시간제한_분"
        ],
        as_index=False
    )
    .agg(

        총실제수요=(
            "실제총수요",
            "sum"
        ),

        총즉시충족수요=(
            "즉시충족수요",
            "sum"
        ),

        총충족수요=(
            "총충족수요",
            "sum"
        ),

        총미충족수요=(
            "미충족수요",
            "sum"
        ),

        총재배치UD=(
            "재배치UD",
            "sum"
        )
    )
)


operating_summary[
    "수요충족률_%"
] = (
    operating_summary[
        "총충족수요"
    ]
    /
    operating_summary[
        "총실제수요"
    ]
    * 100
)


operating_summary[
    "즉시충족률_%"
] = (
    operating_summary[
        "총즉시충족수요"
    ]
    /
    operating_summary[
        "총실제수요"
    ]
    * 100
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
# 21. LightGBM 시간제약 효과
# ============================================================

forecast_scenario = (
    summary[
        summary["정책"]
        ==
        "LightGBM선제"
    ][
        [
            "시간제한_분",
            "전체수요충족률_%",
            "총미충족수요",
            "총재배치UD",
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
        "전체수요충족률_%"
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
# 22. 주요 재배치 경로
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
# 23. 결과 검증
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
            "전체수요충족률_%"
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
            "전체수요충족률_%"
        ]
        .iloc[0]
    )


    status = (
        "정상"
        if f <= o + 1e-9
        else "검토 필요"
    )


    print(
        f"{limit}분: "
        f"LightGBM {f:.2f}% / "
        f"Oracle {o:.2f}% "
        f"→ {status}"
    )


# ============================================================
# 24. 저수요 시간 검증
# ============================================================

print(
    "\n========== "
    "저수요 시간대 검증 "
    "=========="
)


low_demand_check = (
    results[
        (
            results["정책"]
            ==
            "LightGBM선제"
        )
        &
        (
            results["시간제한_분"]
            ==
            60
        )
        &
        (
            results["실제총수요"]
            <= 5
        )
    ]
)


if len(
    low_demand_check
) > 0:

    print(
        "실제수요 5건 이하 시간대:",
        len(
            low_demand_check
        )
    )

    print(
        "평균 수요충족률:",
        round(
            low_demand_check[
                "수요충족률_%"
            ].mean(),
            2
        ),
        "%"
    )

else:

    print(
        "해당 시간대 없음"
    )


# ============================================================
# 25. 정책 해석
# ============================================================

print(
    "\n========== 정책 해석 =========="
)


if forecast_rate > fixed_rate:

    print(
        "✓ LightGBM 예측 기반 선제 재배치는 "
        "고정배치보다 실제 승객 수요 충족률을 "
        "향상시켰습니다."
    )

else:

    print(
        "※ LightGBM 선제 재배치가 "
        "고정배치보다 높은 수요충족률을 "
        "보이지 못했습니다."
    )


if forecast_rate > post_rate:

    print(
        "✓ LightGBM 선제 재배치는 "
        "사후 대응보다 높은 수요충족률을 "
        "보였습니다."
    )

else:

    print(
        "※ 사후 재배치의 총 수요충족률이 "
        "LightGBM 선제 재배치보다 높습니다."
    )


print(
    "※ 사후 재배치는 이동 차량을 기다려야 하므로 "
    "수요충족률뿐 아니라 대기시간을 함께 "
    "비교해야 합니다."
)


print(
    f"Oracle과 LightGBM 차이: "
    f"{oracle_rate - forecast_rate:.2f}%p"
)


# ============================================================
# 26. 저장
# ============================================================

DETAIL_FILE = (
    RESULT_DIR
    / "passenger_service_detail_v3_2025.csv"
)

SUMMARY_FILE = (
    RESULT_DIR
    / "passenger_service_summary_v3_2025.csv"
)

HOUR_FILE = (
    RESULT_DIR
    / "passenger_service_by_hour_v3_2025.csv"
)

DAY_FILE = (
    RESULT_DIR
    / "passenger_service_by_day_v3_2025.csv"
)

OPERATING_FILE = (
    RESULT_DIR
    / "passenger_service_operating_v3_2025.csv"
)

ROUTE_FILE = (
    RESULT_DIR
    / "passenger_service_routes_v3_2025.csv"
)

SCENARIO_FILE = (
    RESULT_DIR
    / "passenger_service_time_limit_v3_2025.csv"
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


print(
    "\n========== 저장 완료 =========="
)

print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)

print(
    HOUR_FILE
)

print(
    DAY_FILE
)

print(
    OPERATING_FILE
)

print(
    ROUTE_FILE
)

print(
    SCENARIO_FILE
)


print(
    "\n========== "
    "forecast_based_rebalancing_v3 "
    "분석 완료 "
    "=========="
)