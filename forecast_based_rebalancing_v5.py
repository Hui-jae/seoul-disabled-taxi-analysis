# ============================================================
# forecast_based_rebalancing_v5.py
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 0. 경로
# ============================================================

BASE_DIR = Path(
    "/Users/sinhuijae/Desktop/공모전/데이터 분석"
)

RESULT_DIR = BASE_DIR / "results"

FORECAST_FILE = (
    RESULT_DIR / "demand_forecast_test_v2_2025.csv"
)

RAW_FILE = (
    BASE_DIR / "disabled_taxi_2025.csv"
)

STOCK_FILE = (
    RESULT_DIR / "vehicle_stock_detail_2025.csv"
)

OUTPUT_DETAIL = RESULT_DIR / "allocation_detail_v5_2025.csv"
OUTPUT_SUMMARY = RESULT_DIR / "allocation_summary_v5_2025.csv"
OUTPUT_HOUR = RESULT_DIR / "allocation_by_hour_v5_2025.csv"
OUTPUT_DAY = RESULT_DIR / "allocation_by_day_v5_2025.csv"
OUTPUT_OPERATING = RESULT_DIR / "allocation_operating_v5_2025.csv"
OUTPUT_ROUTES = RESULT_DIR / "allocation_routes_v5_2025.csv"
OUTPUT_TIME_LIMIT = RESULT_DIR / "allocation_time_limit_v5_2025.csv"
OUTPUT_EFFICIENCY = RESULT_DIR / "allocation_efficiency_v5_2025.csv"


# ============================================================
# 1. 기본 설정
# ============================================================

TOTAL_UD = 12

TIME_LIMITS = [
    20,
    30,
    45,
    60
]

OPERATING_START = 7
OPERATING_END = 22

EPS = 1e-9

DAY_ORDER = [
    "월",
    "화",
    "수",
    "목",
    "금",
    "토",
    "일"
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


# ============================================================
# 2. 함수
# ============================================================

def normalize_gu(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    value = value.replace(
        "서울특별시",
        ""
    )

    value = value.replace(
        "서울시",
        ""
    )

    return value.strip()


def normalize_day(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    mapping = {
        "월요일": "월",
        "화요일": "화",
        "수요일": "수",
        "목요일": "목",
        "금요일": "금",
        "토요일": "토",
        "일요일": "일"
    }

    return mapping.get(
        value,
        value
    )


def find_existing_column(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:
            return col

    return None


def safe_divide(
    a,
    b
):

    if abs(b) <= EPS:
        return 0.0

    return a / b


def normalize_stock(
    stock_dict,
    total=TOTAL_UD
):

    values = {
        k: max(
            float(v),
            0.0
        )
        for k, v
        in stock_dict.items()
    }

    total_now = sum(
        values.values()
    )

    if total_now <= EPS:

        raise ValueError(
            "차량 재고 총합이 0입니다. "
            "vehicle_stock_detail_2025.csv의 "
            "요일/시간대/출발구 데이터를 확인하세요."
        )

    factor = (
        total
        / total_now
    )

    return {
        k: v * factor
        for k, v
        in values.items()
    }


def demand_distribution(
    demand_dict,
    gu_list
):

    values = np.array(
        [
            max(
                float(
                    demand_dict.get(
                        g,
                        0
                    )
                ),
                0
            )
            for g
            in gu_list
        ],
        dtype=float
    )

    total = values.sum()

    if total <= EPS:

        return np.zeros(
            len(gu_list)
        )

    return (
        values
        / total
    )


def stock_distribution(
    stock_dict,
    gu_list
):

    values = np.array(
        [
            max(
                float(
                    stock_dict.get(
                        g,
                        0
                    )
                ),
                0
            )
            for g
            in gu_list
        ],
        dtype=float
    )

    total = values.sum()

    if total <= EPS:

        return np.zeros(
            len(gu_list)
        )

    return (
        values
        / total
    )


def distribution_coverage(
    stock_dict,
    demand_dict,
    gu_list
):

    demand_share = (
        demand_distribution(
            demand_dict,
            gu_list
        )
    )

    vehicle_share = (
        stock_distribution(
            stock_dict,
            gu_list
        )
    )

    return float(
        np.minimum(
            demand_share,
            vehicle_share
        ).sum()
    )


def l1_distribution_error(
    stock_dict,
    demand_dict,
    gu_list
):

    demand_share = (
        demand_distribution(
            demand_dict,
            gu_list
        )
    )

    vehicle_share = (
        stock_distribution(
            stock_dict,
            gu_list
        )
    )

    return float(
        np.abs(
            demand_share
            - vehicle_share
        ).sum()
    )


def demand_region_coverage(
    stock_dict,
    demand_dict,
    gu_list
):

    total_demand = sum(
        max(
            float(
                demand_dict.get(
                    g,
                    0
                )
            ),
            0
        )
        for g
        in gu_list
    )

    if total_demand <= EPS:
        return 1.0

    covered = 0.0

    for g in gu_list:

        demand = max(
            float(
                demand_dict.get(
                    g,
                    0
                )
            ),
            0
        )

        stock = max(
            float(
                stock_dict.get(
                    g,
                    0
                )
            ),
            0
        )

        if stock > EPS:
            covered += demand

    return (
        covered
        / total_demand
    )


# ============================================================
# 빠른 Marginal Gain Greedy
# ============================================================

def greedy_rebalance(
    current_stock,
    decision_demand,
    gu_list,
    travel_time,
    time_limit,
    move_unit=0.5,
    max_iterations=50
):

    stock = normalize_stock(
        current_stock.copy()
    )

    routes = []

    current_score = (
        distribution_coverage(
            stock,
            decision_demand,
            gu_list
        )
    )

    initial_score = current_score

    demand_share = (
        demand_distribution(
            decision_demand,
            gu_list
        )
    )

    demand_share_dict = {
        g: float(
            demand_share[i]
        )
        for i, g
        in enumerate(gu_list)
    }

    iterations = 0

    while (
        iterations
        < max_iterations
    ):

        iterations += 1

        total_stock = sum(
            max(
                float(
                    stock.get(
                        g,
                        0
                    )
                ),
                0
            )
            for g
            in gu_list
        )

        if total_stock <= EPS:
            break

        stock_share = {
            g: (
                max(
                    float(
                        stock.get(
                            g,
                            0
                        )
                    ),
                    0
                )
                / total_stock
            )
            for g
            in gu_list
        }

        balance = {
            g: (
                stock_share[g]
                - demand_share_dict[g]
            )
            for g
            in gu_list
        }

        origins = [
            g
            for g
            in gu_list
            if (
                stock.get(
                    g,
                    0
                ) > EPS
                and balance[g] > EPS
            )
        ]

        destinations = [
            g
            for g
            in gu_list
            if balance[g] < -EPS
        ]

        if (
            not origins
            or not destinations
        ):
            break

        origins.sort(
            key=lambda g: balance[g],
            reverse=True
        )

        destinations.sort(
            key=lambda g: balance[g]
        )

        best_move = None
        best_gain = 0.0
        best_efficiency = 0.0

        for origin in origins:

            available = max(
                float(
                    stock.get(
                        origin,
                        0
                    )
                ),
                0
            )

            if available <= EPS:
                continue

            excess_vehicle = (
                balance[origin]
                * total_stock
            )

            if excess_vehicle <= EPS:
                continue

            for destination in destinations:

                if origin == destination:
                    continue

                t = travel_time.get(
                    (
                        origin,
                        destination
                    ),
                    np.inf
                )

                if not np.isfinite(t):
                    continue

                if t > time_limit:
                    continue

                shortage_vehicle = (
                    -balance[destination]
                    * total_stock
                )

                if shortage_vehicle <= EPS:
                    continue

                amount = min(
                    move_unit,
                    available,
                    excess_vehicle,
                    shortage_vehicle
                )

                if amount <= EPS:
                    continue

                candidate = stock.copy()

                candidate[origin] -= amount
                candidate[destination] += amount

                new_score = (
                    distribution_coverage(
                        candidate,
                        decision_demand,
                        gu_list
                    )
                )

                gain = (
                    new_score
                    - current_score
                )

                if gain <= EPS:
                    continue

                efficiency = (
                    gain
                    / max(
                        float(t),
                        1.0
                    )
                )

                if (
                    gain
                    > best_gain + EPS
                    or (
                        abs(
                            gain
                            - best_gain
                        ) <= EPS
                        and efficiency
                        > best_efficiency
                    )
                ):

                    best_gain = gain

                    best_efficiency = (
                        efficiency
                    )

                    best_move = {
                        "origin":
                            origin,

                        "destination":
                            destination,

                        "amount":
                            amount,

                        "travel_time":
                            float(t),

                        "new_score":
                            new_score
                    }

        if best_move is None:
            break

        origin = best_move[
            "origin"
        ]

        destination = best_move[
            "destination"
        ]

        amount = best_move[
            "amount"
        ]

        stock[origin] -= amount
        stock[destination] += amount

        if stock[origin] < EPS:
            stock[origin] = 0.0

        current_score = (
            best_move[
                "new_score"
            ]
        )

        routes.append(
            {
                "재배치출발구":
                    origin,

                "재배치도착구":
                    destination,

                "재배치UD":
                    amount,

                "이동시간_분":
                    best_move[
                        "travel_time"
                    ],

                "예측커버리지증가":
                    best_gain
            }
        )

    stock = normalize_stock(
        stock
    )

    total_move = sum(
        x["재배치UD"]
        for x
        in routes
    )

    total_empty_time = sum(
        (
            x["재배치UD"]
            * x["이동시간_분"]
        )
        for x
        in routes
    )

    return {
        "stock":
            stock,

        "routes":
            routes,

        "initial_score":
            initial_score,

        "final_score":
            current_score,

        "predicted_gain":
            (
                current_score
                - initial_score
            ),

        "total_move":
            total_move,

        "total_empty_time":
            total_empty_time,

        "iterations":
            iterations
    }


# ============================================================
# 3. LightGBM 예측 데이터
# ============================================================

print(
    "\n========== LightGBM v2 예측 데이터 =========="
)

forecast = pd.read_csv(
    FORECAST_FILE,
    encoding="utf-8-sig"
)

forecast["날짜"] = (
    pd.to_datetime(
        forecast["날짜"]
    )
)

forecast["출발구"] = (
    forecast["출발구"]
    .apply(normalize_gu)
)

forecast["요일"] = (
    forecast["요일"]
    .apply(normalize_day)
)

forecast["시간대"] = (
    pd.to_numeric(
        forecast["시간대"],
        errors="coerce"
    )
)

forecast["수요"] = (
    pd.to_numeric(
        forecast["수요"],
        errors="coerce"
    )
    .fillna(0)
)

forecast["예측수요"] = (
    pd.to_numeric(
        forecast["예측수요"],
        errors="coerce"
    )
    .fillna(0)
    .clip(lower=0)
)

# 서울 25개 구만 사용
forecast = forecast[
    forecast[
        "출발구"
    ].isin(
        SEOUL_GU
    )
].copy()

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
    "\n※ v5에서는 테스트셋 실제값을 이용한 "
    "Bias 보정을 사용하지 않습니다."
)

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
    "예측/실제 비율: "
    f"{safe_divide(pred_total, actual_total):.4f}"
)


# ============================================================
# 4. 이동시간 데이터
# ============================================================

print(
    "\n========== 이동시간 데이터 =========="
)

raw_columns = pd.read_csv(
    RAW_FILE,
    nrows=0,
    encoding="utf-8-sig"
).columns.tolist()

dummy = pd.DataFrame(
    columns=raw_columns
)

origin_col = find_existing_column(
    dummy,
    [
        "출발구",
        "출발지구",
        "승차구"
    ]
)

destination_col = find_existing_column(
    dummy,
    [
        "목적구",
        "목적지구",
        "하차구"
    ]
)

pickup_col = find_existing_column(
    dummy,
    [
        "승차일시",
        "승차시간"
    ]
)

dropoff_col = find_existing_column(
    dummy,
    [
        "하차일시",
        "하차시간"
    ]
)

if (
    origin_col is None
    or destination_col is None
    or pickup_col is None
    or dropoff_col is None
):

    raise ValueError(
        "이동시간 계산에 필요한 컬럼을 "
        "원본 데이터에서 찾지 못했습니다."
    )

raw = pd.read_csv(
    RAW_FILE,
    usecols=[
        origin_col,
        destination_col,
        pickup_col,
        dropoff_col
    ],
    encoding="utf-8-sig",
    low_memory=False
)

raw[origin_col] = (
    raw[origin_col]
    .apply(normalize_gu)
)

raw[destination_col] = (
    raw[destination_col]
    .apply(normalize_gu)
)

raw[pickup_col] = (
    pd.to_datetime(
        raw[pickup_col],
        errors="coerce"
    )
)

raw[dropoff_col] = (
    pd.to_datetime(
        raw[dropoff_col],
        errors="coerce"
    )
)

raw["이동시간_분"] = (
    (
        raw[dropoff_col]
        - raw[pickup_col]
    )
    .dt
    .total_seconds()
    / 60
)

# 서울 -> 서울만
raw = raw[
    raw[
        origin_col
    ].isin(
        SEOUL_GU
    )
    & raw[
        destination_col
    ].isin(
        SEOUL_GU
    )
    & raw[
        "이동시간_분"
    ].notna()
    & (
        raw[
            "이동시간_분"
        ] > 0
    )
    & (
        raw[
            "이동시간_분"
        ] <= 180
    )
].copy()

travel_df = (
    raw
    .groupby(
        [
            origin_col,
            destination_col
        ],
        as_index=False
    )[
        "이동시간_분"
    ]
    .median()
)

default_travel_time = (
    raw[
        "이동시간_분"
    ].median()
)

travel_time = {
    (
        row[origin_col],
        row[destination_col]
    ):
        float(
            row["이동시간_분"]
        )

    for _, row
    in travel_df.iterrows()
}

# 동일 구 이동시간은 0
for g in SEOUL_GU:

    travel_time[
        (g, g)
    ] = 0.0


print(
    f"분석 운행: "
    f"{len(raw):,}건"
)

print(
    f"구간 조합: "
    f"{len(travel_df):,}개"
)

print(
    f"기본 이동시간: "
    f"{default_travel_time:.2f}분"
)


# ============================================================
# 5. 차량 재고
# ============================================================

print(
    "\n========== 차량 재고 데이터 =========="
)

stock_df = pd.read_csv(
    STOCK_FILE,
    encoding="utf-8-sig"
)

required_stock_columns = [
    "요일",
    "승차시간대",
    "출발구",
    "시간대시작UD"
]

missing = [
    c
    for c
    in required_stock_columns
    if c not in stock_df.columns
]

if missing:

    raise ValueError(
        f"차량 재고 데이터에 필요한 컬럼이 없습니다: "
        f"{missing}"
    )

stock_df["요일"] = (
    stock_df["요일"]
    .apply(normalize_day)
)

stock_df["출발구"] = (
    stock_df["출발구"]
    .apply(normalize_gu)
)

stock_df["승차시간대"] = (
    pd.to_numeric(
        stock_df[
            "승차시간대"
        ],
        errors="coerce"
    )
)

stock_df["시간대시작UD"] = (
    pd.to_numeric(
        stock_df[
            "시간대시작UD"
        ],
        errors="coerce"
    )
    .fillna(0)
)

stock_df = stock_df[
    stock_df[
        "출발구"
    ].isin(
        SEOUL_GU
    )
].copy()

print(
    f"분석 조합: "
    f"{len(stock_df):,}개"
)


# ============================================================
# 6. 분석 지역
# ============================================================

gu_list = sorted(
    set(
        forecast[
            "출발구"
        ].dropna()
    )
)

print(
    f"분석 지역: "
    f"{len(gu_list)}개 구"
)

if len(gu_list) != 25:

    print(
        "⚠ 분석 지역이 25개 구가 아닙니다:",
        gu_list
    )


# ============================================================
# 7. 날짜 × 시간대 패널
# ============================================================

print(
    "\n========== 날짜 × 시간대 패널 생성 =========="
)

groups = list(
    forecast.groupby(
        [
            "날짜",
            "시간대"
        ],
        sort=True
    )
)

print(
    f"분석 시간대: "
    f"{len(groups):,}개"
)


# ============================================================
# 8. 현재 재고 조회
# ============================================================

def get_current_stock(
    day,
    hour
):

    temp = stock_df[
        (
            stock_df[
                "요일"
            ] == day
        )
        & (
            stock_df[
                "승차시간대"
            ] == hour
        )
    ]

    result = {
        g: 0.0
        for g
        in gu_list
    }

    for _, row in temp.iterrows():

        gu = row[
            "출발구"
        ]

        if gu in result:

            result[gu] += float(
                row[
                    "시간대시작UD"
                ]
            )

    stock_sum = sum(
        result.values()
    )

    if stock_sum <= EPS:

        raise ValueError(
            f"{day}요일 {hour}시 "
            "차량 재고를 찾지 못했습니다."
        )

    return normalize_stock(
        result
    )


# ============================================================
# 9. 고정배치
# ============================================================

overall_demand = (
    forecast
    .groupby(
        "출발구"
    )[
        "수요"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

top12 = (
    overall_demand
    .head(
        TOTAL_UD
    )
    .index
    .tolist()
)

fixed_stock = {
    g: 0.0
    for g
    in gu_list
}

for g in top12:
    fixed_stock[g] = 1.0

print(
    "\n========== 고정배치 =========="
)

for g in gu_list:

    if fixed_stock[g] > 0:

        print(
            f"{g}: "
            f"{fixed_stock[g]:.0f}대"
        )

print(
    f"총 차량: "
    f"{sum(fixed_stock.values()):.0f}"
)


# ============================================================
# 10. 정책 시뮬레이션
# ============================================================

print(
    "\n========== v5 Marginal Gain 정책 시뮬레이션 =========="
)

detail_rows = []
route_rows = []

total_groups = len(
    groups
)

for group_idx, (
    (date, hour),
    temp
) in enumerate(
    groups,
    start=1
):

    if (
        group_idx == 1
        or group_idx % 50 == 0
        or group_idx == total_groups
    ):

        progress = (
            group_idx
            / total_groups
            * 100
        )

        print(
            f"[진행] "
            f"{group_idx:,}/{total_groups:,} "
            f"({progress:.1f}%) "
            f"- {date.date()} "
            f"{int(hour)}시"
        )

    day = normalize_day(
        temp[
            "요일"
        ].iloc[0]
    )

    actual_demand = {
        g: 0.0
        for g
        in gu_list
    }

    predicted_demand = {
        g: 0.0
        for g
        in gu_list
    }

    for _, row in temp.iterrows():

        gu = row[
            "출발구"
        ]

        if gu not in actual_demand:
            continue

        actual_demand[gu] = float(
            row[
                "수요"
            ]
        )

        predicted_demand[gu] = float(
            row[
                "예측수요"
            ]
        )

    actual_total_hour = sum(
        actual_demand.values()
    )

    current_stock = (
        get_current_stock(
            day,
            hour
        )
    )

    # --------------------------------------------------------
    # 고정배치
    # --------------------------------------------------------

    fixed_cov = (
        distribution_coverage(
            fixed_stock,
            actual_demand,
            gu_list
        )
    )

    fixed_region_cov = (
        demand_region_coverage(
            fixed_stock,
            actual_demand,
            gu_list
        )
    )

    fixed_l1 = (
        l1_distribution_error(
            fixed_stock,
            actual_demand,
            gu_list
        )
    )

    detail_rows.append(
        {
            "날짜":
                date,

            "요일":
                day,

            "시간대":
                hour,

            "정책":
                "고정배치",

            "시간제한_분":
                0,

            "총실제수요":
                actual_total_hour,

            "수요가중커버리지":
                fixed_cov,

            "수요발생지역커버율":
                fixed_region_cov,

            "차량없는수요비율":
                1 - fixed_region_cov,

            "L1분포오차":
                fixed_l1,

            "재배치UD":
                0.0,

            "공차차량시간_분":
                0.0,

            "예측커버리지개선":
                0.0
        }
    )

    # --------------------------------------------------------
    # 현재배치
    # --------------------------------------------------------

    current_cov_hour = (
        distribution_coverage(
            current_stock,
            actual_demand,
            gu_list
        )
    )

    current_region_cov = (
        demand_region_coverage(
            current_stock,
            actual_demand,
            gu_list
        )
    )

    current_l1 = (
        l1_distribution_error(
            current_stock,
            actual_demand,
            gu_list
        )
    )

    detail_rows.append(
        {
            "날짜":
                date,

            "요일":
                day,

            "시간대":
                hour,

            "정책":
                "현재배치",

            "시간제한_분":
                0,

            "총실제수요":
                actual_total_hour,

            "수요가중커버리지":
                current_cov_hour,

            "수요발생지역커버율":
                current_region_cov,

            "차량없는수요비율":
                1 - current_region_cov,

            "L1분포오차":
                current_l1,

            "재배치UD":
                0.0,

            "공차차량시간_분":
                0.0,

            "예측커버리지개선":
                0.0
        }
    )

    # --------------------------------------------------------
    # LightGBM / Oracle
    # --------------------------------------------------------

    for limit in TIME_LIMITS:

        # LightGBM
        result = greedy_rebalance(
            current_stock=
                current_stock,

            decision_demand=
                predicted_demand,

            gu_list=
                gu_list,

            travel_time=
                travel_time,

            time_limit=
                limit
        )

        final_stock = (
            result[
                "stock"
            ]
        )

        actual_cov = (
            distribution_coverage(
                final_stock,
                actual_demand,
                gu_list
            )
        )

        region_cov = (
            demand_region_coverage(
                final_stock,
                actual_demand,
                gu_list
            )
        )

        l1 = (
            l1_distribution_error(
                final_stock,
                actual_demand,
                gu_list
            )
        )

        detail_rows.append(
            {
                "날짜":
                    date,

                "요일":
                    day,

                "시간대":
                    hour,

                "정책":
                    "LightGBM선제",

                "시간제한_분":
                    limit,

                "총실제수요":
                    actual_total_hour,

                "수요가중커버리지":
                    actual_cov,

                "수요발생지역커버율":
                    region_cov,

                "차량없는수요비율":
                    1 - region_cov,

                "L1분포오차":
                    l1,

                "재배치UD":
                    result[
                        "total_move"
                    ],

                "공차차량시간_분":
                    result[
                        "total_empty_time"
                    ],

                "예측커버리지개선":
                    result[
                        "predicted_gain"
                    ]
            }
        )

        for route in result[
            "routes"
        ]:

            route_rows.append(
                {
                    "날짜":
                        date,

                    "요일":
                        day,

                    "시간대":
                        hour,

                    "정책":
                        "LightGBM선제",

                    "시간제한_분":
                        limit,

                    **route
                }
            )

        # Oracle
        oracle = greedy_rebalance(
            current_stock=
                current_stock,

            decision_demand=
                actual_demand,

            gu_list=
                gu_list,

            travel_time=
                travel_time,

            time_limit=
                limit
        )

        oracle_stock = (
            oracle[
                "stock"
            ]
        )

        oracle_cov = (
            distribution_coverage(
                oracle_stock,
                actual_demand,
                gu_list
            )
        )

        oracle_region_cov = (
            demand_region_coverage(
                oracle_stock,
                actual_demand,
                gu_list
            )
        )

        oracle_l1 = (
            l1_distribution_error(
                oracle_stock,
                actual_demand,
                gu_list
            )
        )

        detail_rows.append(
            {
                "날짜":
                    date,

                "요일":
                    day,

                "시간대":
                    hour,

                "정책":
                    "Oracle선제",

                "시간제한_분":
                    limit,

                "총실제수요":
                    actual_total_hour,

                "수요가중커버리지":
                    oracle_cov,

                "수요발생지역커버율":
                    oracle_region_cov,

                "차량없는수요비율":
                    1 - oracle_region_cov,

                "L1분포오차":
                    oracle_l1,

                "재배치UD":
                    oracle[
                        "total_move"
                    ],

                "공차차량시간_분":
                    oracle[
                        "total_empty_time"
                    ],

                "예측커버리지개선":
                    oracle[
                        "predicted_gain"
                    ]
            }
        )

        for route in oracle[
            "routes"
        ]:

            route_rows.append(
                {
                    "날짜":
                        date,

                    "요일":
                        day,

                    "시간대":
                        hour,

                    "정책":
                        "Oracle선제",

                    "시간제한_분":
                        limit,

                    **route
                }
            )


# ============================================================
# 11. DataFrame
# ============================================================

detail = pd.DataFrame(
    detail_rows
)

routes = pd.DataFrame(
    route_rows
)


# ============================================================
# 12. 요약 함수
# ============================================================

def summarize_policy(df):

    rows = []

    for (
        policy,
        limit
    ), temp in df.groupby(
        [
            "정책",
            "시간제한_분"
        ]
    ):

        total_demand = (
            temp[
                "총실제수요"
            ].sum()
        )

        if total_demand > 0:

            weights = (
                temp[
                    "총실제수요"
                ]
            )

            weighted_cov = (
                np.average(
                    temp[
                        "수요가중커버리지"
                    ],
                    weights=weights
                )
            )

            region_cov = (
                np.average(
                    temp[
                        "수요발생지역커버율"
                    ],
                    weights=weights
                )
            )

            l1 = (
                np.average(
                    temp[
                        "L1분포오차"
                    ],
                    weights=weights
                )
            )

        else:

            weighted_cov = 0
            region_cov = 0
            l1 = 0

        rows.append(
            {
                "정책":
                    policy,

                "시간제한_분":
                    limit,

                "총실제수요":
                    total_demand,

                "수요가중커버리지_%":
                    weighted_cov * 100,

                "수요발생지역커버율_%":
                    region_cov * 100,

                "차량없는수요비율_%":
                    (
                        1 - region_cov
                    ) * 100,

                "평균L1분포오차":
                    l1,

                "총재배치UD":
                    temp[
                        "재배치UD"
                    ].sum(),

                "평균재배치UD":
                    temp[
                        "재배치UD"
                    ].mean(),

                "총공차차량시간_분":
                    temp[
                        "공차차량시간_분"
                    ].sum(),

                "평균예측커버리지개선_%p":
                    temp[
                        "예측커버리지개선"
                    ].mean() * 100
            }
        )

    return pd.DataFrame(
        rows
    )


summary = summarize_policy(
    detail
)

print(
    "\n========== v5 수요 공간분포 기준 정책 비교 =========="
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 13. 시간제약별 LightGBM
# ============================================================

time_limit_df = (
    summary[
        summary[
            "정책"
        ] == "LightGBM선제"
    ]
    .copy()
    .sort_values(
        "시간제한_분"
    )
    .reset_index(
        drop=True
    )
)

time_limit_df[
    "이전대비개선_%p"
] = (
    time_limit_df[
        "수요가중커버리지_%"
    ].diff()
)

time_limit_df[
    "추가재배치UD"
] = (
    time_limit_df[
        "총재배치UD"
    ].diff()
)

time_limit_df[
    "추가공차시간_분"
] = (
    time_limit_df[
        "총공차차량시간_분"
    ].diff()
)

print(
    "\n========== LightGBM 시간제약별 성능 =========="
)

print(
    time_limit_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 14. Metric 조회
# ============================================================

def get_metric(
    policy,
    limit,
    metric
):

    temp = summary[
        (
            summary[
                "정책"
            ] == policy
        )
        & (
            summary[
                "시간제한_분"
            ] == limit
        )
    ]

    if len(temp) == 0:
        return np.nan

    return float(
        temp.iloc[0][
            metric
        ]
    )


current_cov = get_metric(
    "현재배치",
    0,
    "수요가중커버리지_%"
)

fixed_cov = get_metric(
    "고정배치",
    0,
    "수요가중커버리지_%"
)

lgb60 = get_metric(
    "LightGBM선제",
    60,
    "수요가중커버리지_%"
)

oracle60 = get_metric(
    "Oracle선제",
    60,
    "수요가중커버리지_%"
)

print(
    "\n========== 60분 기준 핵심 비교 =========="
)

print(
    f"고정배치: "
    f"{fixed_cov:.2f}%"
)

print(
    f"현재배치: "
    f"{current_cov:.2f}%"
)

print(
    f"LightGBM 선제: "
    f"{lgb60:.2f}%"
)

print(
    f"Oracle 선제: "
    f"{oracle60:.2f}%"
)

print(
    "\n고정 → LightGBM 개선:",
    f"{lgb60-fixed_cov:+.2f}%p"
)

print(
    "현재 → LightGBM 개선:",
    f"{lgb60-current_cov:+.2f}%p"
)

print(
    "LightGBM → Oracle Gap:",
    f"{oracle60-lgb60:.2f}%p"
)


# ============================================================
# 15. 시간대별
# ============================================================

hour_rows = []

for (
    policy,
    limit,
    hour
), temp in detail.groupby(
    [
        "정책",
        "시간제한_분",
        "시간대"
    ]
):

    total_demand = (
        temp[
            "총실제수요"
        ].sum()
    )

    if total_demand > 0:

        cov = np.average(
            temp[
                "수요가중커버리지"
            ],
            weights=temp[
                "총실제수요"
            ]
        )

        region_cov = np.average(
            temp[
                "수요발생지역커버율"
            ],
            weights=temp[
                "총실제수요"
            ]
        )

    else:

        cov = 0
        region_cov = 0

    hour_rows.append(
        {
            "정책":
                policy,

            "시간제한_분":
                limit,

            "시간대":
                hour,

            "분석일수":
                len(temp),

            "총실제수요":
                total_demand,

            "수요가중커버리지_%":
                cov * 100,

            "수요발생지역커버율_%":
                region_cov * 100,

            "평균재배치UD":
                temp[
                    "재배치UD"
                ].mean()
        }
    )

by_hour = pd.DataFrame(
    hour_rows
)

print(
    "\n========== 60분 LightGBM 시간대별 성능 =========="
)

temp_hour = by_hour[
    (
        by_hour[
            "정책"
        ] == "LightGBM선제"
    )
    & (
        by_hour[
            "시간제한_분"
        ] == 60
    )
]

print(
    temp_hour.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 16. 요일별
# ============================================================

day_rows = []

for (
    policy,
    limit,
    day
), temp in detail.groupby(
    [
        "정책",
        "시간제한_분",
        "요일"
    ]
):

    total_demand = (
        temp[
            "총실제수요"
        ].sum()
    )

    if total_demand > 0:

        cov = np.average(
            temp[
                "수요가중커버리지"
            ],
            weights=temp[
                "총실제수요"
            ]
        )

        region_cov = np.average(
            temp[
                "수요발생지역커버율"
            ],
            weights=temp[
                "총실제수요"
            ]
        )

    else:

        cov = 0
        region_cov = 0

    day_rows.append(
        {
            "정책":
                policy,

            "시간제한_분":
                limit,

            "요일":
                day,

            "총실제수요":
                total_demand,

            "수요가중커버리지_%":
                cov * 100,

            "수요발생지역커버율_%":
                region_cov * 100,

            "평균재배치UD":
                temp[
                    "재배치UD"
                ].mean()
        }
    )

by_day = pd.DataFrame(
    day_rows
)

by_day["요일"] = (
    pd.Categorical(
        by_day[
            "요일"
        ],
        categories=DAY_ORDER,
        ordered=True
    )
)

by_day = (
    by_day.sort_values(
        [
            "정책",
            "시간제한_분",
            "요일"
        ]
    )
)

print(
    "\n========== 60분 LightGBM 요일별 성능 =========="
)

temp_day = by_day[
    (
        by_day[
            "정책"
        ] == "LightGBM선제"
    )
    & (
        by_day[
            "시간제한_분"
        ] == 60
    )
]

print(
    temp_day.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 17. 운영시간 7~22시
# ============================================================

operating = detail[
    (
        detail[
            "시간대"
        ] >= OPERATING_START
    )
    & (
        detail[
            "시간대"
        ] <= OPERATING_END
    )
].copy()

operating_summary = (
    summarize_policy(
        operating
    )
)

print(
    "\n========== 실질 운영시간 7~22시 정책 비교 =========="
)

print(
    operating_summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.2f}"
    )
)


# ============================================================
# 18. 재배치 경로
# ============================================================

if len(routes) > 0:

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

            총예측커버리지증가=(
                "예측커버리지증가",
                "sum"
            )
        )
    )

    route_summary[
        "총공차차량시간_분"
    ] = (
        route_summary[
            "총재배치UD"
        ]
        * route_summary[
            "평균이동시간_분"
        ]
    )

else:

    route_summary = (
        pd.DataFrame(
            columns=[
                "정책",
                "시간제한_분",
                "재배치출발구",
                "재배치도착구",
                "총재배치UD",
                "평균이동시간_분",
                "총예측커버리지증가",
                "총공차차량시간_분"
            ]
        )
    )


print(
    "\n========== 60분 LightGBM 주요 재배치 경로 TOP 30 =========="
)

if len(route_summary) > 0:

    top_routes = (
        route_summary[
            (
                route_summary[
                    "정책"
                ] == "LightGBM선제"
            )
            & (
                route_summary[
                    "시간제한_분"
                ] == 60
            )
        ]
        .sort_values(
            "총재배치UD",
            ascending=False
        )
        .head(30)
    )

    print(
        top_routes.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}"
        )
    )

else:

    print(
        "재배치 경로 없음"
    )


# ============================================================
# 19. 비용 대비 성능
# ============================================================

efficiency = (
    time_limit_df.copy()
)

efficiency[
    "현재배치대비개선_%p"
] = (
    efficiency[
        "수요가중커버리지_%"
    ]
    - current_cov
)

efficiency[
    "개선율_per_1000공차분"
] = np.where(
    efficiency[
        "총공차차량시간_분"
    ] > EPS,

    efficiency[
        "현재배치대비개선_%p"
    ]
    / (
        efficiency[
            "총공차차량시간_분"
        ]
        / 1000
    ),

    0
)

print(
    "\n========== 재배치 비용 대비 성능 =========="
)

print(
    efficiency[
        [
            "시간제한_분",
            "수요가중커버리지_%",
            "현재배치대비개선_%p",
            "총재배치UD",
            "총공차차량시간_분",
            "개선율_per_1000공차분"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.3f}"
    )
)


# ============================================================
# 20. 권장 시간제약
# ============================================================

positive = efficiency[
    efficiency[
        "현재배치대비개선_%p"
    ] > 0
].copy()

print(
    "\n========== 권장 재배치 시간제약 =========="
)

if len(positive) > 0:

    max_gain = positive[
        "현재배치대비개선_%p"
    ].max()

    max_cost = positive[
        "총공차차량시간_분"
    ].max()

    if max_gain > EPS:

        positive[
            "성능점수"
        ] = (
            positive[
                "현재배치대비개선_%p"
            ]
            / max_gain
        )

    else:

        positive[
            "성능점수"
        ] = 0

    if max_cost > EPS:

        positive[
            "비용점수"
        ] = (
            1
            - (
                positive[
                    "총공차차량시간_분"
                ]
                / max_cost
            )
        )

    else:

        positive[
            "비용점수"
        ] = 1

    positive[
        "정책효율점수"
    ] = (
        0.7
        * positive[
            "성능점수"
        ]
        + 0.3
        * positive[
            "비용점수"
        ]
    )

    recommended = (
        positive
        .sort_values(
            "정책효율점수",
            ascending=False
        )
        .iloc[0]
    )

    print(
        "권장 시간제약:",
        f"{int(recommended['시간제한_분'])}분"
    )

    print(
        "수요가중커버리지:",
        f"{recommended['수요가중커버리지_%']:.2f}%"
    )

    print(
        "현재배치 대비 개선:",
        f"{recommended['현재배치대비개선_%p']:+.2f}%p"
    )

    print(
        "총 재배치량:",
        f"{recommended['총재배치UD']:.2f}대·회"
    )

    print(
        "총 공차차량시간:",
        f"{recommended['총공차차량시간_분']:.2f}분"
    )

else:

    print(
        "현재배치보다 실제 커버리지를 개선한 "
        "시간제약 시나리오가 없습니다."
    )


# ============================================================
# 21. Oracle 비교
# ============================================================

print(
    "\n========== 결과 검증 =========="
)

for limit in TIME_LIMITS:

    lgb = get_metric(
        "LightGBM선제",
        limit,
        "수요가중커버리지_%"
    )

    oracle = get_metric(
        "Oracle선제",
        limit,
        "수요가중커버리지_%"
    )

    print(
        f"{limit}분: "
        f"현재 {current_cov:.2f}% / "
        f"LightGBM {lgb:.2f}% / "
        f"Oracle {oracle:.2f}% / "
        f"예측 Gap {oracle-lgb:.2f}%p"
    )


# ============================================================
# 22. 핵심 정책 해석
# ============================================================

print(
    "\n========== 정책 해석 =========="
)

best_lgb = (
    summary[
        summary[
            "정책"
        ] == "LightGBM선제"
    ]
    .sort_values(
        "수요가중커버리지_%",
        ascending=False
    )
    .iloc[0]
)

best_oracle = (
    summary[
        summary[
            "정책"
        ] == "Oracle선제"
    ]
    .sort_values(
        "수요가중커버리지_%",
        ascending=False
    )
    .iloc[0]
)

print(
    "현재배치 커버리지:",
    f"{current_cov:.2f}%"
)

print(
    "최고 LightGBM 커버리지:",
    f"{best_lgb['수요가중커버리지_%']:.2f}%"
)

print(
    "최고 Oracle 커버리지:",
    f"{best_oracle['수요가중커버리지_%']:.2f}%"
)

if (
    best_lgb[
        "수요가중커버리지_%"
    ] > current_cov
):

    print(
        "✓ LightGBM 예측 기반 Marginal Gain "
        "재배치가 현재배치보다 실제 수요분포 "
        "커버리지를 개선했습니다."
    )

else:

    print(
        "※ LightGBM 재배치가 현재배치를 "
        "개선하지 못했습니다."
    )

if (
    best_oracle[
        "수요가중커버리지_%"
    ] >= current_cov
):

    print(
        "✓ Oracle 재배치는 현재배치 이상의 "
        "성능을 보였습니다."
    )

else:

    print(
        "⚠ Oracle조차 현재배치보다 낮습니다. "
        "커버리지 정의 또는 알고리즘을 "
        "추가 검토해야 합니다."
    )

print(
    "\n※ 현재 차량 위치에서 이동했을 때 "
    "예측 커버리지가 증가하는 경우에만 "
    "선제 재배치를 수행합니다."
)

print(
    "※ LightGBM은 예측수요만 이용해 "
    "재배치를 결정하며 실제수요는 "
    "사후 성능평가에만 사용합니다."
)


# ============================================================
# 23. 저장
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

detail.to_csv(
    OUTPUT_DETAIL,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

by_hour.to_csv(
    OUTPUT_HOUR,
    index=False,
    encoding="utf-8-sig"
)

by_day.to_csv(
    OUTPUT_DAY,
    index=False,
    encoding="utf-8-sig"
)

operating_summary.to_csv(
    OUTPUT_OPERATING,
    index=False,
    encoding="utf-8-sig"
)

route_summary.to_csv(
    OUTPUT_ROUTES,
    index=False,
    encoding="utf-8-sig"
)

time_limit_df.to_csv(
    OUTPUT_TIME_LIMIT,
    index=False,
    encoding="utf-8-sig"
)

efficiency.to_csv(
    OUTPUT_EFFICIENCY,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========== 저장 완료 =========="
)

for path in [
    OUTPUT_DETAIL,
    OUTPUT_SUMMARY,
    OUTPUT_HOUR,
    OUTPUT_DAY,
    OUTPUT_OPERATING,
    OUTPUT_ROUTES,
    OUTPUT_TIME_LIMIT,
    OUTPUT_EFFICIENCY
]:

    print(path)


print(
    "\n========== forecast_based_rebalancing_v5 분석 완료 =========="
)