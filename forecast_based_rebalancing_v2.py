# ============================================================
# forecast_based_rebalancing_v2.py
#
# 목적
# ------------------------------------------------------------
# 동일한 평가 조건에서 다음 정책을 비교한다.
#
# 1. 고정배치 Baseline
# 2. 실제 수요 확인 후 사후 재배치
# 3. LightGBM 예측 기반 선제 재배치
# 4. Oracle 선제 재배치
#
# 공통 조건
# ------------------------------------------------------------
# - UD 차량: 12대
# - 평가기간: LightGBM v2 테스트 기간
# - 공간 단위: 서울 25개 구
# - 시간 단위: 1시간
# - 이동시간 제한: 20 / 30 / 45 / 60분
# - 모든 정책은 동일한 실제 UD 목표수요로 평가
#
# 핵심
# ------------------------------------------------------------
# "Oracle 목표 위치와 얼마나 동일한가"가 아니라
# "실제 UD 목표수요를 얼마나 충족했는가"를 평가한다.
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 0. 경로 / 설정
# ============================================================

BASE_DIR = Path("/Users/sinhuijae/Desktop/공모전")
DATA_DIR = BASE_DIR / "데이터 분석"
RESULT_DIR = DATA_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
# 1. 함수
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
# 12대 정수 배분
#
# 수요 비율을 기준으로 12대를 배분한다.
# Largest Remainder Method 사용
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

    for idx in order[
        :remainder
    ]:

        base[idx] += 1

    for gu, value in zip(
        SEOUL_GU,
        base
    ):

        allocation[gu] = int(
            value
        )

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
        ]
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


# ------------------------------------------------------------
# 목표 배치 충족량 계산
#
# target:
# 실제 수요를 기반으로 계산된
# "실제 UD 목표 배치"
#
# stock:
# 정책 실행 후 실제 차량 위치
#
# 같은 구에 존재하는 차량만 즉시 충족 가능
# ------------------------------------------------------------

def calculate_service(
    stock,
    target
):

    served = 0.0
    unmet = 0.0
    surplus = 0.0

    by_gu = {}

    for gu in SEOUL_GU:

        s = max(
            float(
                stock.get(
                    gu,
                    0
                )
            ),
            0
        )

        t = max(
            float(
                target.get(
                    gu,
                    0
                )
            ),
            0
        )

        gu_served = min(
            s,
            t
        )

        gu_unmet = max(
            t - s,
            0
        )

        gu_surplus = max(
            s - t,
            0
        )

        served += gu_served
        unmet += gu_unmet
        surplus += gu_surplus

        by_gu[gu] = {
            "stock": s,
            "target": t,
            "served": gu_served,
            "unmet": gu_unmet,
            "surplus": gu_surplus
        }

    return (
        served,
        unmet,
        surplus,
        by_gu
    )


# ------------------------------------------------------------
# 최소 이동시간 기반 재배치
# ------------------------------------------------------------

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
                travel_time
                <= time_limit
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
            "이동시간_분": travel_time
        })

    total_moved = sum(
        r["재배치UD"]
        for r in routes
    )

    return (
        stock,
        total_moved,
        routes
    )


# ============================================================
# 2. LightGBM 예측 데이터
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
    forecast[
        "예측수요"
    ]
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
# 3. 예측 Bias 분석
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
    forecast[
        "예측수요"
    ]
    * bias_factor
)


print(
    "\n========== "
    "예측 Bias "
    "=========="
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
# 4. 이동시간 데이터
# ============================================================

print(
    "\n========== "
    "이동시간 데이터 "
    "=========="
)


raw_columns = pd.read_csv(
    RAW_FILE,
    nrows=0
).columns.tolist()


print(
    "원본 컬럼:"
)

print(
    ", ".join(
        raw_columns
    )
)


required_raw = [
    "승차일시",
    "하차일시",
    "출발구",
    "목적구"
]


missing_raw = [
    c
    for c in required_raw
    if c not in raw_columns
]


if missing_raw:

    raise ValueError(
        "원본 운행 데이터에 "
        f"필요 컬럼이 없습니다: "
        f"{missing_raw}"
    )


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
    raw[
        "하차일시"
    ]
    -
    raw[
        "승차일시"
    ]
).dt.total_seconds() / 60


raw = raw[
    (
        raw[
            "운행시간_분"
        ]
        > 0
    )
    &
    (
        raw[
            "운행시간_분"
        ]
        <= 180
    )
].copy()


default_travel_time = (
    raw[
        "운행시간_분"
    ].median()
)


travel = (
    raw.groupby(
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


# 같은 구 내 이동
# 실제 구내 운행시간 median 사용
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
        ] = (
            default_travel_time
        )


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
# 5. 차량 재고 데이터
# ============================================================

print(
    "\n========== "
    "차량 재고 데이터 "
    "=========="
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
    stock_df[
        "시간대시작UD"
    ],
    errors="coerce"
).fillna(0)


print(
    f"분석 조합: "
    f"{len(stock_df):,}개"
)

print(
    "컬럼:"
)

print(
    ", ".join(
        stock_df.columns
    )
)


# ============================================================
# 6. 요일 × 시간대 기준 초기 차량 재고 생성
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
                row[
                    "시간대시작UD"
                ]
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
# 7. 날짜 × 시간대별 목표배치 생성
# ============================================================

print(
    "\n========== "
    "실제 / 예측 UD 목표배치 생성 "
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
            row[
                "수요"
            ]
        )

        predicted_demand[
            gu
        ] += float(
            row[
                "보정예측수요"
            ]
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
        "실제총수요": sum(
            actual_demand.values()
        ),
        "예측총수요": sum(
            predicted_demand.values()
        ),
        "actual_target": actual_target,
        "predicted_target": predicted_target
    })


print(
    f"분석 시간대: "
    f"{len(time_groups):,}개"
)


# ============================================================
# 8. 고정배치 Baseline 생성
# ============================================================

print(
    "\n========== "
    "고정배치 Baseline 생성 "
    "=========="
)


# 테스트 기간 전체 실제 수요 비율을 사용한
# 단일 고정배치
#
# 정책 비교 목적의 단순 baseline
overall_demand = (
    forecast
    .groupby(
        "출발구"
    )[
        "수요"
    ]
    .sum()
    .to_dict()
)


fixed_allocation = (
    allocate_integer_vehicles(
        overall_demand
    )
)


print(
    "고정배치:"
)

for gu in SEOUL_GU:

    if (
        fixed_allocation[
            gu
        ]
        > 0
    ):

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
# 9. 정책 시뮬레이션
# ============================================================

print(
    "\n========== "
    "동일 조건 정책 시뮬레이션 "
    "=========="
)


result_rows = []

route_rows = []


for item in time_groups:

    date = item["날짜"]
    day = item["요일"]
    hour = item["시간대"]

    actual_target = (
        item[
            "actual_target"
        ]
    )

    predicted_target = (
        item[
            "predicted_target"
        ]
    )

    actual_total_demand = (
        item[
            "실제총수요"
        ]
    )

    predicted_total_demand = (
        item[
            "예측총수요"
        ]
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

        # 혹시 데이터가 없으면
        # 전체 고정배치를 fallback으로 사용
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

    (
        fixed_served,
        fixed_unmet,
        fixed_surplus,
        _
    ) = calculate_service(
        fixed_allocation,
        actual_target
    )


    fixed_required = sum(
        actual_target.values()
    )


    fixed_rate = (
        fixed_served
        / fixed_required
        * 100
        if fixed_required > 0
        else 100
    )


    result_rows.append({
        "날짜": date,
        "요일": day,
        "시간대": hour,
        "정책": "고정배치",
        "시간제한_분": 0,
        "실제총수요": actual_total_demand,
        "예측총수요": predicted_total_demand,
        "실제필요UD": fixed_required,
        "충족UD": fixed_served,
        "미충족UD": fixed_unmet,
        "잉여UD": fixed_surplus,
        "재배치UD": 0,
        "충족률_%": fixed_rate
    })


    # ========================================================
    # 시간제약별 정책
    # ========================================================

    for limit in TIME_LIMITS:

        # ====================================================
        # B. 사후 재배치
        #
        # 실제 목표를 알고
        # 수요 발생 후 재배치한다고 가정
        # ====================================================

        (
            post_stock,
            post_move,
            post_routes
        ) = rebalance(
            initial_stock,
            actual_target,
            travel_time_dict,
            limit,
            date=date,
            hour=hour,
            policy="사후재배치"
        )


        (
            post_served,
            post_unmet,
            post_surplus,
            _
        ) = calculate_service(
            post_stock,
            actual_target
        )


        required = sum(
            actual_target.values()
        )


        post_rate = (
            post_served
            / required
            * 100
            if required > 0
            else 100
        )


        result_rows.append({
            "날짜": date,
            "요일": day,
            "시간대": hour,
            "정책": "사후재배치",
            "시간제한_분": limit,
            "실제총수요": actual_total_demand,
            "예측총수요": predicted_total_demand,
            "실제필요UD": required,
            "충족UD": post_served,
            "미충족UD": post_unmet,
            "잉여UD": post_surplus,
            "재배치UD": post_move,
            "충족률_%": post_rate
        })


        route_rows.extend(
            post_routes
        )


        # ====================================================
        # C. LightGBM 예측 기반 선제 재배치
        # ====================================================

        (
            forecast_stock,
            forecast_move,
            forecast_routes
        ) = rebalance(
            initial_stock,
            predicted_target,
            travel_time_dict,
            limit,
            date=date,
            hour=hour,
            policy="LightGBM선제"
        )


        # 중요:
        # 이동 목표는 예측수요지만
        # 평가는 실제 목표수요로 한다.
        (
            forecast_served,
            forecast_unmet,
            forecast_surplus,
            _
        ) = calculate_service(
            forecast_stock,
            actual_target
        )


        forecast_rate = (
            forecast_served
            / required
            * 100
            if required > 0
            else 100
        )


        result_rows.append({
            "날짜": date,
            "요일": day,
            "시간대": hour,
            "정책": "LightGBM선제",
            "시간제한_분": limit,
            "실제총수요": actual_total_demand,
            "예측총수요": predicted_total_demand,
            "실제필요UD": required,
            "충족UD": forecast_served,
            "미충족UD": forecast_unmet,
            "잉여UD": forecast_surplus,
            "재배치UD": forecast_move,
            "충족률_%": forecast_rate
        })


        route_rows.extend(
            forecast_routes
        )


        # ====================================================
        # D. Oracle 선제 재배치
        #
        # 실제 미래 목표를 완전히 안다고 가정
        # ====================================================

        (
            oracle_stock,
            oracle_move,
            oracle_routes
        ) = rebalance(
            initial_stock,
            actual_target,
            travel_time_dict,
            limit,
            date=date,
            hour=hour,
            policy="Oracle선제"
        )


        (
            oracle_served,
            oracle_unmet,
            oracle_surplus,
            _
        ) = calculate_service(
            oracle_stock,
            actual_target
        )


        oracle_rate = (
            oracle_served
            / required
            * 100
            if required > 0
            else 100
        )


        result_rows.append({
            "날짜": date,
            "요일": day,
            "시간대": hour,
            "정책": "Oracle선제",
            "시간제한_분": limit,
            "실제총수요": actual_total_demand,
            "예측총수요": predicted_total_demand,
            "실제필요UD": required,
            "충족UD": oracle_served,
            "미충족UD": oracle_unmet,
            "잉여UD": oracle_surplus,
            "재배치UD": oracle_move,
            "충족률_%": oracle_rate
        })


        route_rows.extend(
            oracle_routes
        )


results = pd.DataFrame(
    result_rows
)

routes = pd.DataFrame(
    route_rows
)


# ============================================================
# 10. 전체 정책 성능
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
        총필요UD=(
            "실제필요UD",
            "sum"
        ),
        총충족UD=(
            "충족UD",
            "sum"
        ),
        총미충족UD=(
            "미충족UD",
            "sum"
        ),
        총재배치UD=(
            "재배치UD",
            "sum"
        ),
        평균시간대충족률=(
            "충족률_%",
            "mean"
        )
    )
)


summary[
    "전체충족률_%"
] = (
    summary[
        "총충족UD"
    ]
    /
    summary[
        "총필요UD"
    ]
    * 100
)


policy_order = {
    "고정배치": 0,
    "사후재배치": 1,
    "LightGBM선제": 2,
    "Oracle선제": 3
}


summary[
    "_order"
] = (
    summary[
        "정책"
    ]
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
        columns=[
            "_order"
        ]
    )
)


print(
    "\n========== "
    "동일 조건 정책 성능 비교 "
    "=========="
)


print(
    summary[
        [
            "정책",
            "시간제한_분",
            "총필요UD",
            "총충족UD",
            "총미충족UD",
            "총재배치UD",
            "전체충족률_%"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. 60분 기준 핵심 정책 비교
# ============================================================

print(
    "\n========== "
    "60분 기준 핵심 정책 효과 "
    "=========="
)


fixed_row = summary[
    summary["정책"]
    ==
    "고정배치"
].iloc[0]


post_row = summary[
    (
        summary["정책"]
        ==
        "사후재배치"
    )
    &
    (
        summary[
            "시간제한_분"
        ]
        ==
        60
    )
].iloc[0]


forecast_row = summary[
    (
        summary["정책"]
        ==
        "LightGBM선제"
    )
    &
    (
        summary[
            "시간제한_분"
        ]
        ==
        60
    )
].iloc[0]


oracle_row = summary[
    (
        summary["정책"]
        ==
        "Oracle선제"
    )
    &
    (
        summary[
            "시간제한_분"
        ]
        ==
        60
    )
].iloc[0]


fixed_rate = (
    fixed_row[
        "전체충족률_%"
    ]
)

post_rate = (
    post_row[
        "전체충족률_%"
    ]
)

forecast_rate = (
    forecast_row[
        "전체충족률_%"
    ]
)

oracle_rate = (
    oracle_row[
        "전체충족률_%"
    ]
)


print(
    f"고정배치: "
    f"{fixed_rate:.2f}%"
)

print(
    f"사후 재배치: "
    f"{post_rate:.2f}%"
)

print(
    f"LightGBM 선제 재배치: "
    f"{forecast_rate:.2f}%"
)

print(
    f"Oracle 선제 재배치: "
    f"{oracle_rate:.2f}%"
)


print()

print(
    "고정배치 → LightGBM 개선: "
    f"{forecast_rate - fixed_rate:+.2f}%p"
)

print(
    "사후배치 → LightGBM 차이: "
    f"{forecast_rate - post_rate:+.2f}%p"
)

print(
    "LightGBM → Oracle Gap: "
    f"{oracle_rate - forecast_rate:.2f}%p"
)


oracle_possible_gain = (
    oracle_rate
    - fixed_rate
)

forecast_gain = (
    forecast_rate
    - fixed_rate
)


if oracle_possible_gain > 0:

    achievement = (
        forecast_gain
        / oracle_possible_gain
        * 100
    )

else:

    achievement = np.nan


print(
    "Oracle 개선 가능량 대비 "
    "LightGBM 달성률: "
    f"{achievement:.2f}%"
)


# ============================================================
# 12. 시간제약별 LightGBM 효과
# ============================================================

print(
    "\n========== "
    "LightGBM 시간제약별 성능 "
    "=========="
)


forecast_scenario = summary[
    summary["정책"]
    ==
    "LightGBM선제"
][
    [
        "시간제한_분",
        "총재배치UD",
        "총미충족UD",
        "전체충족률_%"
    ]
].copy()


print(
    forecast_scenario
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 13. 한계효용 분석
# ============================================================

forecast_scenario = (
    forecast_scenario
    .sort_values(
        "시간제한_분"
    )
    .copy()
)


forecast_scenario[
    "이전대비개선_%p"
] = (
    forecast_scenario[
        "전체충족률_%"
    ]
    .diff()
)


print(
    "\n========== "
    "시간제약 증가 한계효용 "
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
# 14. 시간대별 정책 비교
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
        평균실제수요=(
            "실제총수요",
            "mean"
        ),
        평균예측수요=(
            "예측총수요",
            "mean"
        ),
        총필요UD=(
            "실제필요UD",
            "sum"
        ),
        총충족UD=(
            "충족UD",
            "sum"
        ),
        총미충족UD=(
            "미충족UD",
            "sum"
        ),
        평균재배치UD=(
            "재배치UD",
            "mean"
        )
    )
)


by_hour[
    "충족률_%"
] = (
    by_hour[
        "총충족UD"
    ]
    /
    by_hour[
        "총필요UD"
    ]
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
        by_hour[
            "시간제한_분"
        ]
        ==
        60
    )
]


print(
    hour60[
        [
            "시간대",
            "분석일수",
            "평균실제수요",
            "평균예측수요",
            "총미충족UD",
            "평균재배치UD",
            "충족률_%"
        ]
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 15. 요일별 정책 비교
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
        분석시간대=(
            "시간대",
            "size"
        ),
        총필요UD=(
            "실제필요UD",
            "sum"
        ),
        총충족UD=(
            "충족UD",
            "sum"
        ),
        총미충족UD=(
            "미충족UD",
            "sum"
        ),
        평균재배치UD=(
            "재배치UD",
            "mean"
        )
    )
)


by_day[
    "충족률_%"
] = (
    by_day[
        "총충족UD"
    ]
    /
    by_day[
        "총필요UD"
    ]
    * 100
)


# 요일 순서
day_order = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}


by_day[
    "_day_order"
] = (
    by_day[
        "요일"
    ]
    .map(
        day_order
    )
)


by_day = (
    by_day
    .sort_values(
        [
            "정책",
            "시간제한_분",
            "_day_order"
        ]
    )
    .drop(
        columns=[
            "_day_order"
        ]
    )
)


# ============================================================
# 16. 운영시간 구간별 분석
# ============================================================

def operation_period(hour):

    if 0 <= hour <= 6:
        return "저수요_0_6"

    elif 7 <= hour <= 19:
        return "주간_7_19"

    elif 20 <= hour <= 22:
        return "야간_20_22"

    else:
        return "심야_23"


results[
    "운영구간"
] = (
    results[
        "시간대"
    ]
    .apply(
        operation_period
    )
)


period_summary = (
    results
    .groupby(
        [
            "정책",
            "시간제한_분",
            "운영구간"
        ],
        as_index=False
    )
    .agg(
        총필요UD=(
            "실제필요UD",
            "sum"
        ),
        총충족UD=(
            "충족UD",
            "sum"
        ),
        총미충족UD=(
            "미충족UD",
            "sum"
        ),
        총재배치UD=(
            "재배치UD",
            "sum"
        )
    )
)


period_summary[
    "충족률_%"
] = (
    period_summary[
        "총충족UD"
    ]
    /
    period_summary[
        "총필요UD"
    ]
    * 100
)


print(
    "\n========== "
    "60분 LightGBM 운영시간 구간별 성능 "
    "=========="
)


period60 = period_summary[
    (
        period_summary[
            "정책"
        ]
        ==
        "LightGBM선제"
    )
    &
    (
        period_summary[
            "시간제한_분"
        ]
        ==
        60
    )
]


print(
    period60
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 17. 7~22시 별도 정책평가
# ============================================================

operating_results = results[
    results[
        "시간대"
    ].between(
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
        총필요UD=(
            "실제필요UD",
            "sum"
        ),
        총충족UD=(
            "충족UD",
            "sum"
        ),
        총미충족UD=(
            "미충족UD",
            "sum"
        ),
        총재배치UD=(
            "재배치UD",
            "sum"
        )
    )
)


operating_summary[
    "충족률_%"
] = (
    operating_summary[
        "총충족UD"
    ]
    /
    operating_summary[
        "총필요UD"
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
# 18. LightGBM 실패 분석
# ============================================================

forecast_failure = results[
    (
        results[
            "정책"
        ]
        ==
        "LightGBM선제"
    )
    &
    (
        results[
            "시간제한_분"
        ]
        ==
        60
    )
].copy()


oracle60 = results[
    (
        results[
            "정책"
        ]
        ==
        "Oracle선제"
    )
    &
    (
        results[
            "시간제한_분"
        ]
        ==
        60
    )
][
    [
        "날짜",
        "시간대",
        "충족UD",
        "미충족UD",
        "충족률_%"
    ]
].copy()


oracle60 = oracle60.rename(
    columns={
        "충족UD":
        "Oracle충족UD",

        "미충족UD":
        "Oracle미충족UD",

        "충족률_%":
        "Oracle충족률_%"
    }
)


forecast_failure = (
    forecast_failure
    .merge(
        oracle60,
        on=[
            "날짜",
            "시간대"
        ],
        how="left"
    )
)


forecast_failure[
    "Oracle대비손실UD"
] = (
    forecast_failure[
        "Oracle충족UD"
    ]
    -
    forecast_failure[
        "충족UD"
    ]
)


forecast_failure[
    "Oracle대비충족률Gap_%p"
] = (
    forecast_failure[
        "Oracle충족률_%"
    ]
    -
    forecast_failure[
        "충족률_%"
    ]
)


forecast_failure = (
    forecast_failure
    .sort_values(
        "Oracle대비손실UD",
        ascending=False
    )
)


print(
    "\n========== "
    "LightGBM 예측 실패 영향 TOP 30 "
    "=========="
)


print(
    forecast_failure[
        [
            "날짜",
            "요일",
            "시간대",
            "실제총수요",
            "예측총수요",
            "실제필요UD",
            "충족UD",
            "미충족UD",
            "충족률_%",
            "Oracle충족률_%",
            "Oracle대비손실UD",
            "Oracle대비충족률Gap_%p"
        ]
    ]
    .head(30)
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. 주요 재배치 경로
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
            )
        )
    )


    route_summary[
        "총차량시간_분"
    ] = (
        route_summary[
            "총재배치UD"
        ]
        *
        route_summary[
            "평균이동시간_분"
        ]
    )


    print(
        "\n========== "
        "60분 LightGBM 주요 재배치 경로 TOP 30 "
        "=========="
    )


    top_routes = (
        route_summary[
            (
                route_summary[
                    "정책"
                ]
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
        top_routes
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    route_summary = pd.DataFrame()


# ============================================================
# 20. 검증
# ============================================================

print(
    "\n========== "
    "결과 검증 "
    "=========="
)


# Oracle은 같은 초기재고에서
# 실제 미래 목표를 알고 있으므로
# LightGBM보다 낮으면 이상
for limit in TIME_LIMITS:

    f = summary[
        (
            summary[
                "정책"
            ]
            ==
            "LightGBM선제"
        )
        &
        (
            summary[
                "시간제한_분"
            ]
            ==
            limit
        )
    ][
        "전체충족률_%"
    ].iloc[0]


    o = summary[
        (
            summary[
                "정책"
            ]
            ==
            "Oracle선제"
        )
        &
        (
            summary[
                "시간제한_분"
            ]
            ==
            limit
        )
    ][
        "전체충족률_%"
    ].iloc[0]


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


# ------------------------------------------------------------
# 차량 총량 검증
# ------------------------------------------------------------

invalid_target = 0


for item in time_groups:

    if (
        sum(
            item[
                "actual_target"
            ].values()
        )
        != TOTAL_UD
    ):

        invalid_target += 1


print(
    f"실제 목표배치 "
    f"12대 불일치: "
    f"{invalid_target}개"
)


# ============================================================
# 21. 정책 해석 자동 출력
# ============================================================

print(
    "\n========== "
    "정책 해석 "
    "=========="
)


if forecast_rate > fixed_rate:

    print(
        "✓ LightGBM 예측 기반 선제 재배치는 "
        "고정배치보다 높은 실제 목표수요 "
        "충족률을 보였습니다."
    )

else:

    print(
        "※ LightGBM 선제 재배치가 "
        "고정배치를 개선하지 못했습니다."
    )


if forecast_rate > post_rate:

    print(
        "✓ LightGBM 선제 재배치가 "
        "동일한 60분 이동제약에서 "
        "사후 재배치보다 우수했습니다."
    )

else:

    print(
        "※ 동일한 60분 조건에서는 "
        "사후 재배치의 충족률이 "
        "LightGBM 선제 재배치보다 높습니다."
    )


print(
    f"Oracle과의 남은 성능 차이: "
    f"{oracle_rate - forecast_rate:.2f}%p"
)


# ============================================================
# 22. 저장
# ============================================================

RESULT_DETAIL_FILE = (
    RESULT_DIR
    / "policy_comparison_detail_v2_2025.csv"
)

RESULT_SUMMARY_FILE = (
    RESULT_DIR
    / "policy_comparison_v2_2025.csv"
)

RESULT_HOUR_FILE = (
    RESULT_DIR
    / "policy_comparison_by_hour_v2_2025.csv"
)

RESULT_DAY_FILE = (
    RESULT_DIR
    / "policy_comparison_by_day_v2_2025.csv"
)

RESULT_PERIOD_FILE = (
    RESULT_DIR
    / "policy_comparison_period_v2_2025.csv"
)

RESULT_OPERATING_FILE = (
    RESULT_DIR
    / "policy_comparison_operating_hours_v2_2025.csv"
)

RESULT_ROUTE_FILE = (
    RESULT_DIR
    / "forecast_rebalancing_routes_v2_2025.csv"
)

RESULT_FAILURE_FILE = (
    RESULT_DIR
    / "policy_failure_analysis_v2_2025.csv"
)


results.to_csv(
    RESULT_DETAIL_FILE,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    RESULT_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig"
)

by_hour.to_csv(
    RESULT_HOUR_FILE,
    index=False,
    encoding="utf-8-sig"
)

by_day.to_csv(
    RESULT_DAY_FILE,
    index=False,
    encoding="utf-8-sig"
)

period_summary.to_csv(
    RESULT_PERIOD_FILE,
    index=False,
    encoding="utf-8-sig"
)

operating_summary.to_csv(
    RESULT_OPERATING_FILE,
    index=False,
    encoding="utf-8-sig"
)

route_summary.to_csv(
    RESULT_ROUTE_FILE,
    index=False,
    encoding="utf-8-sig"
)

forecast_failure.to_csv(
    RESULT_FAILURE_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========== "
    "저장 완료 "
    "=========="
)

print(
    RESULT_DETAIL_FILE
)

print(
    RESULT_SUMMARY_FILE
)

print(
    RESULT_HOUR_FILE
)

print(
    RESULT_DAY_FILE
)

print(
    RESULT_PERIOD_FILE
)

print(
    RESULT_OPERATING_FILE
)

print(
    RESULT_ROUTE_FILE
)

print(
    RESULT_FAILURE_FILE
)


print(
    "\n========== "
    "forecast_based_rebalancing_v2 "
    "분석 완료 "
    "=========="
)