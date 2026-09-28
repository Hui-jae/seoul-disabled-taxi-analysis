from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"

STOCK_PATH = RESULT_DIR / "vehicle_stock_detail_2025.csv"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

TIME_LIMITS = [20, 30, 45, 60]

DAY_ORDER = {
    "월": 0, "화": 1, "수": 2, "목": 3,
    "금": 4, "토": 5, "일": 6
}


# =========================================================
# 2. 실제 이동시간 데이터 로드
# =========================================================

usecols = [
    "출발구",
    "목적구",
    "승차일시",
    "운행시간_분"
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

trip["운행시간_분"] = pd.to_numeric(
    trip["운행시간_분"],
    errors="coerce"
)

trip = trip[
    (trip["승차일시"].dt.year == 2025)
    & trip["출발구"].notna()
    & trip["목적구"].notna()
    & trip["운행시간_분"].notna()
    & (trip["운행시간_분"] > 0)
    & (trip["운행시간_분"] <= 180)
].copy()

print("========== 이동시간 데이터 ==========")
print(f"분석 운행: {len(trip):,}건")


# =========================================================
# 3. 구간별 평균 이동시간
# =========================================================

travel_time = (
    trip.groupby(
        ["출발구", "목적구"],
        as_index=False
    )
    .agg(
        평균이동시간_분=("운행시간_분", "median"),
        운행건수=("운행시간_분", "size")
    )
)

default_travel_time = trip["운행시간_분"].median()

print(f"구간 조합: {len(travel_time):,}개")
print(f"기본 이동시간: {default_travel_time:.2f}분")


# =========================================================
# 4. 차량 재고 데이터 로드
# =========================================================

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
        f"{STOCK_PATH.name}에 다음 컬럼이 없습니다: {missing}"
    )

for col in [
    "목표UD",
    "시간대시작UD",
    "자연충족UD",
    "부족UD",
    "잉여UD"
]:
    stock[col] = pd.to_numeric(
        stock[col],
        errors="coerce"
    ).fillna(0)

print("\n========== 차량 재고 데이터 ==========")
print(f"분석 조합: {len(stock):,}개")


# =========================================================
# 5. 동일 구 현위치 유지 계산
# =========================================================
# 기존 v2에서는 같은 구 -> 같은 구도
# '재배치'로 계산될 가능성이 있었음.
#
# v3에서는 해당 구에 이미 존재하는 차량이
# 다음 시간대에도 해당 구에서 필요하다면
# 이동이 아닌 '현위치 유지'로 처리한다.
# =========================================================

stock["현위치유지UD"] = np.minimum(
    stock["잉여UD"],
    stock["부족UD"]
)

# 논리적으로 한 행에서 부족과 잉여가 동시에 발생하지 않는 구조라면
# 대부분 0이 된다.
#
# 실제 동일구 유지 효과는 다음 시간대 부족과
# 이전 시간대 잉여를 연결해서 계산한다.


# =========================================================
# 6. 이전 시간대 잉여 -> 다음 시간대 부족 연결
# =========================================================

def previous_hour(day, hour):

    if hour > 0:
        return day, hour - 1

    prev_day = {
        "월": "일",
        "화": "월",
        "수": "화",
        "목": "수",
        "금": "목",
        "토": "금",
        "일": "토"
    }

    return prev_day[day], 23


# =========================================================
# 7. 선제 재배치 최적화
# =========================================================

def optimize_predictive_rebalancing(
    target_day,
    target_hour,
    time_limit
):

    prev_day, prev_hour = previous_hour(
        target_day,
        target_hour
    )

    # -----------------------------------------------------
    # 다음 시간대 부족 지역
    # -----------------------------------------------------

    demand = stock[
        (stock["요일"] == target_day)
        & (stock["승차시간대"] == target_hour)
        & (stock["부족UD"] > 0)
    ][
        [
            "출발구",
            "부족UD"
        ]
    ].copy()

    if demand.empty:
        return [], {
            "현위치유지UD": 0,
            "선제재배치UD": 0,
            "미충족UD": 0
        }

    # -----------------------------------------------------
    # 이전 시간대 잉여 지역
    # -----------------------------------------------------

    supply = stock[
        (stock["요일"] == prev_day)
        & (stock["승차시간대"] == prev_hour)
        & (stock["잉여UD"] > 0)
    ][
        [
            "출발구",
            "잉여UD"
        ]
    ].copy()

    total_need = demand["부족UD"].sum()

    if supply.empty:
        return [], {
            "현위치유지UD": 0,
            "선제재배치UD": 0,
            "미충족UD": total_need
        }

    demand_dict = dict(
        zip(
            demand["출발구"],
            demand["부족UD"]
        )
    )

    supply_dict = dict(
        zip(
            supply["출발구"],
            supply["잉여UD"]
        )
    )

    # =====================================================
    # 7-1. 동일 구 현위치 유지
    # =====================================================

    stay_records = []
    total_stay = 0

    common_gu = set(demand_dict).intersection(
        supply_dict
    )

    for gu in common_gu:

        stay = min(
            demand_dict[gu],
            supply_dict[gu]
        )

        if stay <= 0:
            continue

        demand_dict[gu] -= stay
        supply_dict[gu] -= stay

        total_stay += stay

        stay_records.append({
            "요일": target_day,
            "승차시간대": target_hour,
            "시간제한_분": time_limit,
            "재배치출발구": gu,
            "재배치도착구": gu,
            "구분": "현위치유지",
            "배치UD": stay,
            "이동시간_분": 0.0
        })

    # =====================================================
    # 7-2. 실제 지역 간 이동 후보 생성
    # =====================================================

    candidates = []

    for origin, available in supply_dict.items():

        if available <= 0:
            continue

        for destination, need in demand_dict.items():

            if need <= 0:
                continue

            # 동일 구는 이미 위에서 처리
            if origin == destination:
                continue

            tt = travel_time[
                (travel_time["출발구"] == origin)
                & (travel_time["목적구"] == destination)
            ]

            if len(tt) > 0:
                move_time = float(
                    tt.iloc[0]["평균이동시간_분"]
                )
            else:
                move_time = float(
                    default_travel_time
                )

            # 시간 제약
            if move_time > time_limit:
                continue

            candidates.append(
                (
                    move_time,
                    origin,
                    destination
                )
            )

    # 이동시간이 짧은 경로부터 배치
    candidates.sort(
        key=lambda x: x[0]
    )

    move_records = []
    total_move = 0

    for move_time, origin, destination in candidates:

        available = supply_dict.get(
            origin,
            0
        )

        need = demand_dict.get(
            destination,
            0
        )

        if available <= 0 or need <= 0:
            continue

        amount = min(
            available,
            need
        )

        if amount <= 0:
            continue

        supply_dict[origin] -= amount
        demand_dict[destination] -= amount

        total_move += amount

        move_records.append({
            "요일": target_day,
            "승차시간대": target_hour,
            "시간제한_분": time_limit,
            "재배치출발구": origin,
            "재배치도착구": destination,
            "구분": "선제재배치",
            "배치UD": amount,
            "이동시간_분": move_time
        })

    remaining_need = sum(
        max(v, 0)
        for v in demand_dict.values()
    )

    records = stay_records + move_records

    summary = {
        "현위치유지UD": total_stay,
        "선제재배치UD": total_move,
        "미충족UD": remaining_need
    }

    return records, summary


# =========================================================
# 8. 시간제약별 시뮬레이션
# =========================================================

all_routes = []
hourly_results = []

active_hours = (
    stock[
        stock["부족UD"] > 0
    ][
        ["요일", "승차시간대"]
    ]
    .drop_duplicates()
)

print(
    "\n========== 시간제약별 선제 재배치 시뮬레이션 =========="
)

for time_limit in TIME_LIMITS:

    print(
        f"\n----- {time_limit}분 제한 -----"
    )

    for _, row in active_hours.iterrows():

        day = row["요일"]
        hour = int(
            row["승차시간대"]
        )

        target = stock[
            (stock["요일"] == day)
            & (stock["승차시간대"] == hour)
        ]

        total_need = target[
            "부족UD"
        ].sum()

        routes, summary = (
            optimize_predictive_rebalancing(
                day,
                hour,
                time_limit
            )
        )

        all_routes.extend(routes)

        stay = summary[
            "현위치유지UD"
        ]

        move = summary[
            "선제재배치UD"
        ]

        unmet = summary[
            "미충족UD"
        ]

        total_fulfilled = stay + move

        fulfillment_rate = (
            total_fulfilled
            / total_need
            * 100
            if total_need > 0
            else 100
        )

        move_only = [
            r for r in routes
            if r["구분"] == "선제재배치"
        ]

        if move_only:

            weighted_time = sum(
                r["배치UD"]
                * r["이동시간_분"]
                for r in move_only
            )

            move_amount = sum(
                r["배치UD"]
                for r in move_only
            )

            avg_move_time = (
                weighted_time / move_amount
                if move_amount > 0
                else 0
            )

        else:
            avg_move_time = 0

        hourly_results.append({
            "요일": day,
            "승차시간대": hour,
            "시간제한_분": time_limit,
            "필요재배치UD": total_need,
            "현위치유지UD": stay,
            "실제선제재배치UD": move,
            "총선제충족UD": total_fulfilled,
            "최종미충족UD": unmet,
            "선제충족률_%": fulfillment_rate,
            "평균실제이동시간_분": avg_move_time
        })


# =========================================================
# 9. DataFrame 변환
# =========================================================

routes_df = pd.DataFrame(
    all_routes
)

hourly = pd.DataFrame(
    hourly_results
)


# =========================================================
# 10. 시간제약별 전체 성능
# =========================================================

scenario_summary = (
    hourly.groupby(
        "시간제한_분",
        as_index=False
    )
    .agg(
        총필요재배치UD=(
            "필요재배치UD",
            "sum"
        ),
        총현위치유지UD=(
            "현위치유지UD",
            "sum"
        ),
        총실제선제재배치UD=(
            "실제선제재배치UD",
            "sum"
        ),
        총미충족UD=(
            "최종미충족UD",
            "sum"
        ),
        평균시간대충족률=(
            "선제충족률_%",
            "mean"
        ),
        평균실제이동시간=(
            "평균실제이동시간_분",
            "mean"
        )
    )
)

scenario_summary["전체충족률_%"] = np.where(
    scenario_summary[
        "총필요재배치UD"
    ] > 0,
    (
        scenario_summary[
            "총현위치유지UD"
        ]
        + scenario_summary[
            "총실제선제재배치UD"
        ]
    )
    / scenario_summary[
        "총필요재배치UD"
    ]
    * 100,
    100
)

scenario_summary[
    "실제이동비율_%"
] = np.where(
    scenario_summary[
        "총필요재배치UD"
    ] > 0,
    scenario_summary[
        "총실제선제재배치UD"
    ]
    / scenario_summary[
        "총필요재배치UD"
    ]
    * 100,
    0
)

scenario_summary[
    "현위치유지비율_%"
] = np.where(
    scenario_summary[
        "총필요재배치UD"
    ] > 0,
    scenario_summary[
        "총현위치유지UD"
    ]
    / scenario_summary[
        "총필요재배치UD"
    ]
    * 100,
    0
)


# =========================================================
# 11. 출력
# =========================================================

print(
    "\n========== 시간제약별 Oracle 선제 재배치 결과 =========="
)

print(
    scenario_summary
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 12. 시간제약별 미충족량 비교
# =========================================================

print(
    "\n========== 시간제약별 정책 성능 비교 =========="
)

for _, row in scenario_summary.iterrows():

    limit = int(
        row["시간제한_분"]
    )

    print(
        f"\n[{limit}분 이내 선제 이동]"
    )

    print(
        f"전체 충족률: "
        f"{row['전체충족률_%']:.2f}%"
    )

    print(
        f"현위치 유지: "
        f"{row['총현위치유지UD']:.2f}대·회"
    )

    print(
        f"실제 지역간 이동: "
        f"{row['총실제선제재배치UD']:.2f}대·회"
    )

    print(
        f"최종 미충족: "
        f"{row['총미충족UD']:.2f}대·회"
    )


# =========================================================
# 13. 60분 Oracle benchmark
# =========================================================

oracle = scenario_summary[
    scenario_summary[
        "시간제한_분"
    ] == 60
]

if not oracle.empty:

    oracle = oracle.iloc[0]

    print(
        "\n========== Oracle Benchmark =========="
    )

    print(
        "※ 미래 지역별 차량 부족을 정확히 "
        "알고 있다고 가정한 성능 상한"
    )

    print(
        f"Oracle 전체 충족률: "
        f"{oracle['전체충족률_%']:.2f}%"
    )

    print(
        f"Oracle 미충족량: "
        f"{oracle['총미충족UD']:.2f}대·회"
    )

    print(
        f"현위치 유지 비율: "
        f"{oracle['현위치유지비율_%']:.2f}%"
    )

    print(
        f"실제 이동 비율: "
        f"{oracle['실제이동비율_%']:.2f}%"
    )


# =========================================================
# 14. 주요 실제 재배치 경로
# =========================================================

if not routes_df.empty:

    actual_moves = routes_df[
        routes_df["구분"]
        == "선제재배치"
    ].copy()

    if not actual_moves.empty:

        major_routes = (
            actual_moves.groupby(
                [
                    "시간제한_분",
                    "재배치출발구",
                    "재배치도착구"
                ],
                as_index=False
            )
            .agg(
                총재배치UD=(
                    "배치UD",
                    "sum"
                ),
                평균이동시간_분=(
                    "이동시간_분",
                    "mean"
                )
            )
        )

        major_60 = (
            major_routes[
                major_routes[
                    "시간제한_분"
                ] == 60
            ]
            .sort_values(
                "총재배치UD",
                ascending=False
            )
            .head(30)
        )

        print(
            "\n========== Oracle 주요 실제 재배치 경로 TOP 30 =========="
        )

        print(
            major_60
            .round(2)
            .to_string(index=False)
        )


# =========================================================
# 15. 동일구 유지 TOP 30
# =========================================================

if not routes_df.empty:

    stay_df = routes_df[
        (routes_df["구분"] == "현위치유지")
        & (routes_df["시간제한_분"] == 60)
    ]

    if not stay_df.empty:

        stay_summary = (
            stay_df.groupby(
                "재배치출발구",
                as_index=False
            )
            .agg(
                총현위치유지UD=(
                    "배치UD",
                    "sum"
                )
            )
            .sort_values(
                "총현위치유지UD",
                ascending=False
            )
        )

        print(
            "\n========== 현위치 유지 지역 =========="
        )

        print(
            stay_summary
            .head(30)
            .round(2)
            .to_string(index=False)
        )


# =========================================================
# 16. 요일별 60분 Oracle
# =========================================================

oracle_hourly = hourly[
    hourly[
        "시간제한_분"
    ] == 60
].copy()

daily = (
    oracle_hourly.groupby(
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
        평균현위치유지=(
            "현위치유지UD",
            "mean"
        ),
        평균실제선제재배치=(
            "실제선제재배치UD",
            "mean"
        ),
        평균미충족=(
            "최종미충족UD",
            "mean"
        ),
        평균충족률=(
            "선제충족률_%",
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
    "\n========== 요일별 Oracle 선제 재배치 =========="
)

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 17. 저장
# =========================================================

routes_df.to_csv(
    RESULT_DIR
    / "predictive_rebalancing_routes_v3_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

hourly.to_csv(
    RESULT_DIR
    / "predictive_rebalancing_hourly_v3_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

scenario_summary.to_csv(
    RESULT_DIR
    / "predictive_rebalancing_scenarios_v3_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    RESULT_DIR
    / "predictive_rebalancing_daily_v3_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n========== 저장 완료 =========="
)

print(
    RESULT_DIR
    / "predictive_rebalancing_routes_v3_2025.csv"
)

print(
    RESULT_DIR
    / "predictive_rebalancing_hourly_v3_2025.csv"
)

print(
    RESULT_DIR
    / "predictive_rebalancing_scenarios_v3_2025.csv"
)

print(
    RESULT_DIR
    / "predictive_rebalancing_daily_v3_2025.csv"
)