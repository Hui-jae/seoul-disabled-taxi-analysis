# ============================================================
# forecast_based_rebalancing.py
#
# LightGBM v2 수요예측 + 차량재고 + 이동시간 기반
# 실제 예측(Non-Oracle) 선제 재배치 시뮬레이션
#
# 비교 대상
# 1. 기존 사후 최소비용 재배치
# 2. LightGBM 예측 기반 선제 재배치
# 3. Oracle 선제 재배치
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd

from scipy.optimize import linprog

warnings.filterwarnings("ignore")


# ============================================================
# 0. 경로 설정
# ============================================================

BASE_DIR = "/Users/sinhuijae/Desktop/공모전/데이터 분석"
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

FORECAST_FILE = os.path.join(
    RESULT_DIR,
    "demand_forecast_test_v2_2025.csv"
)

STOCK_FILE = os.path.join(
    RESULT_DIR,
    "vehicle_stock_detail_2025.csv"
)

REBALANCING_FILE = os.path.join(
    RESULT_DIR,
    "rebalancing_hourly_v2_2025.csv"
)

ORACLE_FILE = os.path.join(
    RESULT_DIR,
    "predictive_rebalancing_scenarios_v3_2025.csv"
)

RAW_FILE = os.path.join(
    BASE_DIR,
    "disabled_taxi_2025.csv"
)


TIME_LIMITS = [20, 30, 45, 60]

TOTAL_UD = 12

SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구"
]


# ============================================================
# 1. 유틸 함수
# ============================================================

def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def normalize_gu(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    for gu in SEOUL_GU:
        if gu in x:
            return gu

    return np.nan


def allocate_integer_vehicles(demand, total_vehicles=12):
    """
    지역별 수요 비율에 따라 정수 차량을 배분한다.
    Largest Remainder Method 사용.

    단, 총수요가 0이면 모두 0으로 둔다.
    """

    demand = np.asarray(demand, dtype=float)
    demand = np.maximum(demand, 0)

    result = np.zeros(len(demand), dtype=int)

    total_demand = demand.sum()

    if total_demand <= 0:
        return result

    raw = demand / total_demand * total_vehicles

    floor_alloc = np.floor(raw).astype(int)

    remainder = total_vehicles - floor_alloc.sum()

    result = floor_alloc.copy()

    if remainder > 0:
        fractional = raw - floor_alloc
        order = np.argsort(-fractional)

        for idx in order[:remainder]:
            result[idx] += 1

    return result


# ============================================================
# 2. LightGBM v2 예측 데이터
# ============================================================

print("========== LightGBM v2 예측 데이터 ==========")

forecast = pd.read_csv(FORECAST_FILE)

forecast["날짜"] = pd.to_datetime(forecast["날짜"])

forecast["출발구"] = forecast["출발구"].apply(normalize_gu)

forecast = forecast[
    forecast["출발구"].isin(SEOUL_GU)
].copy()

forecast["예측수요"] = pd.to_numeric(
    forecast["예측수요"],
    errors="coerce"
).fillna(0)

forecast["수요"] = pd.to_numeric(
    forecast["수요"],
    errors="coerce"
).fillna(0)

forecast["예측수요"] = forecast["예측수요"].clip(lower=0)

print(f"데이터 수: {len(forecast):,}개")
print(
    f"기간: {forecast['날짜'].min().date()} "
    f"~ {forecast['날짜'].max().date()}"
)


# ============================================================
# 3. 예측 Bias 확인
# ============================================================

actual_total = forecast["수요"].sum()
pred_total = forecast["예측수요"].sum()

bias_factor = safe_div(actual_total, pred_total)

print("\n========== 예측 Bias ==========")

print(f"실제 총수요: {actual_total:,.2f}")
print(f"예측 총수요: {pred_total:,.2f}")
print(f"Bias 계수: {bias_factor:.4f}")

# 주의:
# 전체 Bias를 모든 지역에 동일하게 곱하면
# 지역별 상대적 비율은 변하지 않는다.
#
# 여기서는 절대 수요 수준 보정용으로만 저장한다.

forecast["보정예측수요"] = (
    forecast["예측수요"] * bias_factor
)

# ============================================================
# 4. 실제 운행 데이터에서 OD 이동시간 생성
# ============================================================

print("\n========== 이동시간 데이터 ==========")

# 실제 disabled_taxi_2025.csv 컬럼명 사용
usecols = [
    "승차일시",
    "하차일시",
    "출발구",
    "목적구"
]

raw = pd.read_csv(
    RAW_FILE,
    usecols=usecols,
    low_memory=False
)

# ------------------------------------------------------------
# 날짜/시간 변환
# ------------------------------------------------------------

raw["승차일시"] = pd.to_datetime(
    raw["승차일시"],
    errors="coerce"
)

raw["하차일시"] = pd.to_datetime(
    raw["하차일시"],
    errors="coerce"
)

# ------------------------------------------------------------
# 서울 25개 구 정규화
# ------------------------------------------------------------

raw["출발구"] = raw["출발구"].apply(normalize_gu)
raw["도착구"] = raw["목적구"].apply(normalize_gu)

# ------------------------------------------------------------
# 운행시간 계산
# ------------------------------------------------------------

raw["운행시간_분"] = (
    raw["하차일시"] - raw["승차일시"]
).dt.total_seconds() / 60

# ------------------------------------------------------------
# 기존 분석과 동일하게
# 서울 → 서울 + 정상 운행시간만 사용
# ------------------------------------------------------------

raw = raw[
    raw["출발구"].isin(SEOUL_GU)
    & raw["도착구"].isin(SEOUL_GU)
    & raw["운행시간_분"].between(
        1,
        180
    )
].copy()

# ------------------------------------------------------------
# 전체 기본 이동시간
# ------------------------------------------------------------

default_travel_time = raw[
    "운행시간_분"
].median()

# ------------------------------------------------------------
# OD별 이동시간
#
# 이전 predictive_rebalancing_v3와 동일하게
# 중앙값 기반 이동시간 사용
# ------------------------------------------------------------

travel = (
    raw
    .groupby(
        [
            "출발구",
            "도착구"
        ],
        as_index=False
    )
    .agg(
        평균이동시간_분=(
            "운행시간_분",
            "median"
        )
    )
)

# ------------------------------------------------------------
# 딕셔너리 변환
# ------------------------------------------------------------

travel_time = {}

for _, r in travel.iterrows():

    travel_time[
        (
            r["출발구"],
            r["도착구"]
        )
    ] = float(
        r["평균이동시간_분"]
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

print("\n========== 차량 재고 데이터 ==========")

stock = pd.read_csv(
    STOCK_FILE,
    encoding="utf-8-sig"
)

print(f"분석 조합: {len(stock):,}개")
print("컬럼:")
print(", ".join(stock.columns))


# 컬럼 자동 탐색
gu_candidates = [
    "출발구",
    "구",
    "지역"
]

stock_candidates = [
    "시간대시작UD",
    "시작UD",
    "보유UD",
    "가용UD"
]


def find_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


stock_gu_col = find_existing_column(
    stock,
    gu_candidates
)

stock_value_col = find_existing_column(
    stock,
    stock_candidates
)


if stock_gu_col is None:
    raise ValueError(
        "vehicle_stock_detail_2025.csv에서 "
        "지역 컬럼을 찾을 수 없습니다."
    )


if stock_value_col is None:
    raise ValueError(
        "vehicle_stock_detail_2025.csv에서 "
        "시간대 시작 UD 재고 컬럼을 찾을 수 없습니다."
    )


stock["출발구"] = stock[
    stock_gu_col
].apply(normalize_gu)

stock["시간대시작UD"] = pd.to_numeric(
    stock[stock_value_col],
    errors="coerce"
).fillna(0)


# ============================================================
# 6. 예측 기반 목표 차량 배치 생성
# ============================================================

print(
    "\n========== 예측 기반 목표 배치 생성 =========="
)

allocation_rows = []


for (date, hour), g in forecast.groupby(
    [
        "날짜",
        "시간대"
    ]
):

    g = (
        g
        .set_index("출발구")
        .reindex(SEOUL_GU)
    )


    # --------------------------------------------------------
    # 예측 수요
    # --------------------------------------------------------

    pred = (
        g["예측수요"]
        .fillna(0)
        .clip(lower=0)
        .values
    )


    # --------------------------------------------------------
    # 실제 수요
    # --------------------------------------------------------

    actual = (
        g["수요"]
        .fillna(0)
        .clip(lower=0)
        .values
    )


    # --------------------------------------------------------
    # 예측 기반 12대 배치
    # --------------------------------------------------------

    predicted_allocation = (
        allocate_integer_vehicles(
            pred,
            TOTAL_UD
        )
    )


    # --------------------------------------------------------
    # Oracle 12대 배치
    # --------------------------------------------------------

    oracle_allocation = (
        allocate_integer_vehicles(
            actual,
            TOTAL_UD
        )
    )


    # --------------------------------------------------------
    # 배치 총량 검증
    #
    # 수요가 존재하는 시간대라면
    # 반드시 12대가 배치되어야 한다.
    # --------------------------------------------------------

    if pred.sum() > 0:

        if (
            predicted_allocation.sum()
            != TOTAL_UD
        ):

            raise ValueError(
                f"{pd.Timestamp(date).date()} "
                f"{int(hour)}시 "
                "예측 목표 배치가 "
                f"{TOTAL_UD}대가 아닙니다. "
                f"현재: "
                f"{predicted_allocation.sum()}"
            )


    if actual.sum() > 0:

        if (
            oracle_allocation.sum()
            != TOTAL_UD
        ):

            raise ValueError(
                f"{pd.Timestamp(date).date()} "
                f"{int(hour)}시 "
                "Oracle 목표 배치가 "
                f"{TOTAL_UD}대가 아닙니다. "
                f"현재: "
                f"{oracle_allocation.sum()}"
            )


    for i, gu in enumerate(SEOUL_GU):

        allocation_rows.append(
            {
                "날짜":
                    date,

                "시간대":
                    int(hour),

                "출발구":
                    gu,

                "실제수요":
                    float(actual[i]),

                "예측수요":
                    float(pred[i]),

                "예측목표UD":
                    int(
                        predicted_allocation[i]
                    ),

                "Oracle목표UD":
                    int(
                        oracle_allocation[i]
                    )
            }
        )


allocation = pd.DataFrame(
    allocation_rows
)


# ------------------------------------------------------------
# 생성 결과 검증
# ------------------------------------------------------------

allocation_time_count = (
    allocation[
        [
            "날짜",
            "시간대"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


print(
    f"시간대: "
    f"{allocation_time_count:,}개"
)


# 각 날짜×시간대 25개 구 확인
allocation_gu_check = (
    allocation
    .groupby(
        [
            "날짜",
            "시간대"
        ]
    )["출발구"]
    .nunique()
)


if (
    allocation_gu_check
    != len(SEOUL_GU)
).any():

    raise ValueError(
        "예측 목표 배치 데이터 중 "
        "25개 구가 모두 존재하지 않는 "
        "시간대가 있습니다."
    )


# 예측 목표 총량
pred_target_check = (
    allocation
    .groupby(
        [
            "날짜",
            "시간대"
        ],
        as_index=False
    )
    .agg(
        예측목표합=(
            "예측목표UD",
            "sum"
        ),

        Oracle목표합=(
            "Oracle목표UD",
            "sum"
        ),

        실제총수요=(
            "실제수요",
            "sum"
        ),

        예측총수요=(
            "예측수요",
            "sum"
        )
    )
)


invalid_pred_target = (
    pred_target_check[
        (
            pred_target_check[
                "예측총수요"
            ] > 0
        )
        &
        (
            pred_target_check[
                "예측목표합"
            ] != TOTAL_UD
        )
    ]
)


invalid_oracle_target = (
    pred_target_check[
        (
            pred_target_check[
                "실제총수요"
            ] > 0
        )
        &
        (
            pred_target_check[
                "Oracle목표합"
            ] != TOTAL_UD
        )
    ]
)


if len(invalid_pred_target) > 0:

    raise ValueError(
        "예측 목표 차량 총량이 "
        "12대가 아닌 시간대가 있습니다."
    )


if len(invalid_oracle_target) > 0:

    raise ValueError(
        "Oracle 목표 차량 총량이 "
        "12대가 아닌 시간대가 있습니다."
    )


print(
    "예측/Oracle 목표 배치 검증 완료"
)

# ============================================================
# 7. 재배치 최적화 함수
# ============================================================

def optimize_rebalancing(
    current_stock,
    target_stock,
    time_limit
):
    """
    현재 차량 위치에서 예측 목표 위치까지
    최소 이동시간으로 차량을 재배치한다.

    이동시간이 time_limit 이하인 경로만 허용.
    """

    surplus = {}
    shortage = {}

    for gu in SEOUL_GU:

        current = current_stock.get(gu, 0)
        target = target_stock.get(gu, 0)

        diff = current - target

        if diff > 0:
            surplus[gu] = diff

        elif diff < 0:
            shortage[gu] = -diff


    # 같은 지역 차량은 이동할 필요 없음
    new_stock = current_stock.copy()

    routes = []

    if not surplus or not shortage:
        return new_stock, routes


    route_candidates = []

    for origin, supply in surplus.items():

        for destination, demand in shortage.items():

            if origin == destination:
                continue

            t = travel_time.get(
                (origin, destination),
                default_travel_time
            )

            if t <= time_limit:

                route_candidates.append(
                    (
                        origin,
                        destination,
                        t
                    )
                )


    if not route_candidates:
        return new_stock, routes


    n = len(route_candidates)

    c = np.array(
        [
            x[2]
            for x in route_candidates
        ],
        dtype=float
    )


    # 미충족량을 강하게 벌점
    #
    # 단순 이동시간 최소화만 하면
    # 차량을 안 움직이는 것이 최적이 되므로
    # 실제로는 먼저 최대 충족량을 확보하고
    # 그 안에서 이동시간을 최소화해야 한다.
    #
    # 여기서는 LP를 2단계로 수행한다.


    # --------------------------------------------------------
    # Stage 1
    # 최대 이동 가능 차량량 계산
    # --------------------------------------------------------

    c_stage1 = -np.ones(n)

    A_ub = []
    b_ub = []


    # 출발지 공급 제약
    for origin in surplus:

        row = np.zeros(n)

        for i, (
            o,
            d,
            t
        ) in enumerate(route_candidates):

            if o == origin:
                row[i] = 1

        A_ub.append(row)
        b_ub.append(surplus[origin])


    # 도착지 필요량 제약
    for destination in shortage:

        row = np.zeros(n)

        for i, (
            o,
            d,
            t
        ) in enumerate(route_candidates):

            if d == destination:
                row[i] = 1

        A_ub.append(row)
        b_ub.append(shortage[destination])


    result1 = linprog(
        c_stage1,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, None)] * n,
        method="highs"
    )


    if not result1.success:
        return new_stock, routes


    max_flow = result1.x.sum()


    if max_flow <= 1e-8:
        return new_stock, routes


    # --------------------------------------------------------
    # Stage 2
    # 최대 충족량을 유지하면서 이동시간 최소화
    # --------------------------------------------------------

    A_eq = [
        np.ones(n)
    ]

    b_eq = [
        max_flow
    ]


    result2 = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=[(0, None)] * n,
        method="highs"
    )


    if not result2.success:
        x = result1.x

    else:
        x = result2.x


    for i, amount in enumerate(x):

        if amount <= 1e-6:
            continue

        origin, destination, t = (
            route_candidates[i]
        )

        new_stock[origin] -= amount
        new_stock[destination] += amount

        routes.append(
            {
                "재배치출발구": origin,
                "재배치도착구": destination,
                "재배치UD": amount,
                "이동시간_분": t
            }
        )


    return new_stock, routes


# ============================================================
# 8. 예측 기반 선제 재배치 시뮬레이션
# ============================================================

print(
    "\n========== LightGBM 예측 기반 "
    "선제 재배치 =========="
)


scenario_rows = []
hourly_rows = []
route_rows = []


# 여기서는 각 시간제약을 독립적으로 평가
for time_limit in TIME_LIMITS:

    print(
        f"\n----- {time_limit}분 제한 -----"
    )


    grouped_times = (
        allocation[
            ["날짜", "시간대"]
        ]
        .drop_duplicates()
        .sort_values(
            ["날짜", "시간대"]
        )
    )


    total_required = 0.0
    total_satisfied = 0.0
    total_shortage = 0.0
    total_moved = 0.0


    for _, time_row in grouped_times.iterrows():

        date = time_row["날짜"]
        hour = int(time_row["시간대"])


        temp = allocation[
            (allocation["날짜"] == date)
            & (allocation["시간대"] == hour)
        ].copy()


        # ----------------------------------------------------
        # 현재 재고
        #
        # 기존 vehicle_stock 데이터는
        # 요일 + 시간대 기반일 가능성이 높으므로
        # 날짜의 요일을 계산해 연결
        # ----------------------------------------------------

        weekday_map = {
            0: "월",
            1: "화",
            2: "수",
            3: "목",
            4: "금",
            5: "토",
            6: "일"
        }

        weekday = weekday_map[
            pd.Timestamp(date).weekday()
        ]


        current = stock.copy()

        if "요일" in current.columns:
            current = current[
                current["요일"] == weekday
            ]

        if "승차시간대" in current.columns:
            current = current[
                current["승차시간대"] == hour
            ]

        elif "시간대" in current.columns:
            current = current[
                current["시간대"] == hour
            ]


        current_stock = dict(
            zip(
                current["출발구"],
                current["시간대시작UD"]
            )
        )


        # 누락 지역 0
        current_stock = {
            gu: float(
                current_stock.get(gu, 0)
            )
            for gu in SEOUL_GU
        }


        # ============================================================
        # 차량 재고가 없는 시간대 보완
        # ============================================================

        stock_sum = sum(current_stock.values())

        if stock_sum <= 0:

            # 현재 날짜의 요일
            current_dow = date.dayofweek
            dow_map = {
                0: "월",
                1: "화",
                2: "수",
                3: "목",
                4: "금",
                5: "토",
                6: "일"
            }

            dow_name = dow_map[current_dow]

            # 같은 요일에서 현재 시간보다 이전 시간대 검색
            available_hours = (
                stock[
                    (stock["요일"] == dow_name) &
                    (stock["승차시간대"] < hour)
                ]["승차시간대"]
                .dropna()
                .unique()
            )

            if len(available_hours) > 0:

                nearest_hour = max(available_hours)

                fallback_stock = stock[
                    (stock["요일"] == dow_name) &
                    (stock["승차시간대"] == nearest_hour)
                ]

                current_stock = {
                    gu: 0.0
                    for gu in SEOUL_GU
                }

                for _, row in fallback_stock.iterrows():

                    gu = row["출발구"]

                    if gu in current_stock:
                        current_stock[gu] = float(
                            row["시간대시작UD"]
                        )

                print(
                    f"[재고 보완] {date.date()} {hour}시 → "
                    f"{dow_name}요일 {nearest_hour}시 재고 사용"
                )

            else:

                # 이전 시간대도 없다면
                # 같은 요일에서 가장 가까운 시간대 사용
                same_day_stock = stock[
                    stock["요일"] == dow_name
                ].copy()

                if len(same_day_stock) == 0:
                    raise ValueError(
                        f"{dow_name}요일 차량 재고 데이터가 없습니다."
                    )

                available_hours = (
                    same_day_stock["승차시간대"]
                    .dropna()
                    .unique()
                )

                nearest_hour = min(
                    available_hours,
                    key=lambda x: abs(x - hour)
                )

                fallback_stock = same_day_stock[
                    same_day_stock["승차시간대"] == nearest_hour
                ]

                current_stock = {
                    gu: 0.0
                    for gu in SEOUL_GU
                }

                for _, row in fallback_stock.iterrows():

                    gu = row["출발구"]

                    if gu in current_stock:
                        current_stock[gu] = float(
                            row["시간대시작UD"]
                        )

                print(
                    f"[재고 보완] {date.date()} {hour}시 → "
                    f"{dow_name}요일 {nearest_hour}시 재고 사용"
                )


        # ============================================================
        # 최종 재고 검증
        # ============================================================

        stock_sum = sum(current_stock.values())

        if stock_sum <= 0:
            raise ValueError(
                f"{date.date()} {hour}시 "
                "보완 후에도 차량 재고가 0입니다."
            )


        # ----------------------------------------------------
        # 예측 목표
        # ----------------------------------------------------

        target_stock = dict(
            zip(
                temp["출발구"],
                temp["예측목표UD"]
            )
        )


        # ----------------------------------------------------
        # 선제 재배치
        # ----------------------------------------------------

        after_stock, routes = (
            optimize_rebalancing(
                current_stock,
                target_stock,
                time_limit
            )
        )


        moved = sum(
            r["재배치UD"]
            for r in routes
        )


        # ----------------------------------------------------
        # 실제 수요 기준 Oracle 필요 위치와 비교
        #
        # 실제 수요 비율로 만든 12대 목표와
        # 선제 재배치 후 차량 위치 비교
        # ----------------------------------------------------

        oracle_target = dict(
            zip(
                temp["출발구"],
                temp["Oracle목표UD"]
            )
        )


        required = 0.0
        satisfied = 0.0
        shortage = 0.0


        for gu in SEOUL_GU:

            need = oracle_target.get(
                gu,
                0
            )

            have = after_stock.get(
                gu,
                0
            )

            required += need

            sat = min(
                need,
                have
            )

            satisfied += sat

            shortage += max(
                need - have,
                0
            )


        rate = (
            safe_div(
                satisfied,
                required
            )
            * 100
        )


        total_required += required
        total_satisfied += satisfied
        total_shortage += shortage
        total_moved += moved


        hourly_rows.append(
            {
                "시간제한_분":
                    time_limit,

                "날짜":
                    date,

                "요일":
                    weekday,

                "시간대":
                    hour,

                "실제총수요":
                    temp["실제수요"].sum(),

                "예측총수요":
                    temp["예측수요"].sum(),

                "실제필요UD":
                    required,

                "예측선제충족UD":
                    satisfied,

                "예측선제미충족UD":
                    shortage,

                "실제선제재배치UD":
                    moved,

                "충족률_%":
                    rate
            }
        )


        for r in routes:

            route_rows.append(
                {
                    "시간제한_분":
                        time_limit,

                    "날짜":
                        date,

                    "요일":
                        weekday,

                    "시간대":
                        hour,

                    **r
                }
            )


    overall_rate = (
        safe_div(
            total_satisfied,
            total_required
        )
        * 100
    )


    scenario_rows.append(
        {
            "시간제한_분":
                time_limit,

            "총필요UD":
                total_required,

            "총충족UD":
                total_satisfied,

            "총미충족UD":
                total_shortage,

            "총선제재배치UD":
                total_moved,

            "전체충족률_%":
                overall_rate
        }
    )


# ============================================================
# 9. 결과 DataFrame
# ============================================================

scenario_df = pd.DataFrame(
    scenario_rows
)

hourly_df = pd.DataFrame(
    hourly_rows
)

routes_df = pd.DataFrame(
    route_rows
)


print(
    "\n========== 예측 기반 "
    "선제 재배치 결과 =========="
)

print(
    scenario_df.to_string(
        index=False,
        formatters={
            "총필요UD":
                "{:.2f}".format,

            "총충족UD":
                "{:.2f}".format,

            "총미충족UD":
                "{:.2f}".format,

            "총선제재배치UD":
                "{:.2f}".format,

            "전체충족률_%":
                "{:.2f}".format
        }
    )
)


# ============================================================
# 10. 시간대 평균 성능
# ============================================================

hour_summary = (
    hourly_df
    .groupby(
        [
            "시간제한_분",
            "시간대"
        ],
        as_index=False
    )
    .agg(
        분석일수=(
            "날짜",
            "count"
        ),

        평균실제수요=(
            "실제총수요",
            "mean"
        ),

        평균예측수요=(
            "예측총수요",
            "mean"
        ),

        평균미충족UD=(
            "예측선제미충족UD",
            "mean"
        ),

        평균선제재배치UD=(
            "실제선제재배치UD",
            "mean"
        ),

        평균충족률=(
            "충족률_%",
            "mean"
        )
    )
)


print(
    "\n========== 60분 제한 "
    "시간대별 성능 =========="
)

print(
    hour_summary[
        hour_summary["시간제한_분"]
        == 60
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 11. 요일별 성능
# ============================================================

daily_summary = (
    hourly_df
    .groupby(
        [
            "시간제한_분",
            "요일"
        ],
        as_index=False
    )
    .agg(
        분석시간대=(
            "시간대",
            "count"
        ),

        평균미충족UD=(
            "예측선제미충족UD",
            "mean"
        ),

        평균선제재배치UD=(
            "실제선제재배치UD",
            "mean"
        ),

        평균충족률=(
            "충족률_%",
            "mean"
        )
    )
)


weekday_order = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}

daily_summary["요일순서"] = (
    daily_summary["요일"]
    .map(weekday_order)
)

daily_summary = (
    daily_summary
    .sort_values(
        [
            "시간제한_분",
            "요일순서"
        ]
    )
    .drop(
        columns="요일순서"
    )
)


print(
    "\n========== 60분 제한 "
    "요일별 성능 =========="
)

print(
    daily_summary[
        daily_summary["시간제한_분"]
        == 60
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 12. 주요 재배치 경로
# ============================================================

if len(routes_df) > 0:

    route_summary = (
        routes_df
        .groupby(
            [
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


    top_routes = (
        route_summary[
            route_summary["시간제한_분"]
            == 60
        ]
        .sort_values(
            "총재배치UD",
            ascending=False
        )
        .head(30)
    )


    print(
        "\n========== 60분 제한 "
        "주요 예측 선제 재배치 경로 TOP 30 =========="
    )

    print(
        top_routes.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

else:

    route_summary = pd.DataFrame()


# ============================================================
# 13. 기존 사후 재배치 결과
# ============================================================

POST_RATE = 70.84
ORACLE_RATE = 89.20


forecast_60 = scenario_df.loc[
    scenario_df["시간제한_분"] == 60,
    "전체충족률_%"
].iloc[0]


print(
    "\n========== 정책 성능 비교 =========="
)

print(
    f"기존 사후 재배치 충족률: "
    f"{POST_RATE:.2f}%"
)

print(
    f"LightGBM 예측 기반 60분 선제 재배치: "
    f"{forecast_60:.2f}%"
)

print(
    f"Oracle 60분 선제 재배치: "
    f"{ORACLE_RATE:.2f}%"
)

print(
    f"기존 대비 개선: "
    f"{forecast_60 - POST_RATE:+.2f}%p"
)

print(
    f"Oracle 대비 차이: "
    f"{forecast_60 - ORACLE_RATE:+.2f}%p"
)


if ORACLE_RATE > POST_RATE:

    oracle_gain = (
        ORACLE_RATE
        - POST_RATE
    )

    forecast_gain = (
        forecast_60
        - POST_RATE
    )

    attainment = (
        safe_div(
            forecast_gain,
            oracle_gain
        )
        * 100
    )

    print(
        f"Oracle 개선 가능량 대비 달성률: "
        f"{attainment:.2f}%"
    )


# ============================================================
# 14. 검증
# ============================================================

print(
    "\n========== 결과 검증 =========="
)

if forecast_60 > ORACLE_RATE + 1e-6:

    print(
        "⚠️ WARNING: "
        "예측 기반 성능이 Oracle보다 높습니다."
    )

    print(
        "평가 정의 또는 차량 재고 연결을 "
        "다시 확인해야 합니다."
    )

else:

    print(
        "예측 기반 성능 <= Oracle 성능: 정상"
    )


if forecast_60 < POST_RATE:

    print(
        "※ 예측 기반 선제 재배치가 "
        "기존 사후 재배치보다 낮습니다."
    )

    print(
        "예측 오차 또는 시간제약의 영향을 "
        "추가 분석할 필요가 있습니다."
    )

else:

    print(
        "예측 기반 선제 재배치가 "
        "기존 사후 재배치보다 개선되었습니다."
    )


# ============================================================
# 15. 저장
# ============================================================

scenario_path = os.path.join(
    RESULT_DIR,
    "forecast_based_rebalancing_scenarios_2025.csv"
)

hourly_path = os.path.join(
    RESULT_DIR,
    "forecast_based_rebalancing_hourly_2025.csv"
)

daily_path = os.path.join(
    RESULT_DIR,
    "forecast_based_rebalancing_daily_2025.csv"
)

routes_path = os.path.join(
    RESULT_DIR,
    "forecast_based_rebalancing_routes_2025.csv"
)

hour_summary_path = os.path.join(
    RESULT_DIR,
    "forecast_based_rebalancing_by_hour_2025.csv"
)


scenario_df.to_csv(
    scenario_path,
    index=False,
    encoding="utf-8-sig"
)

hourly_df.to_csv(
    hourly_path,
    index=False,
    encoding="utf-8-sig"
)

daily_summary.to_csv(
    daily_path,
    index=False,
    encoding="utf-8-sig"
)

hour_summary.to_csv(
    hour_summary_path,
    index=False,
    encoding="utf-8-sig"
)

routes_df.to_csv(
    routes_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "\n========== 저장 완료 =========="
)

print(scenario_path)
print(hourly_path)
print(daily_path)
print(routes_path)
print(hour_summary_path)