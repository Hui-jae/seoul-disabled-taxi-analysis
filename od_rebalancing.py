from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"
ALLOCATION_PATH = RESULT_DIR / "ud_allocation_2025.csv"

TOTAL_UD = 12

DAY_ORDER = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}

DAY_MAP = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일"
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
# 2. OD 데이터 로드
# =========================================================

trip = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=[
        "승차일시",
        "출발구",
        "목적구"
    ]
)

trip["승차일시"] = pd.to_datetime(
    trip["승차일시"],
    errors="coerce"
)

trip = (
    trip[
        trip["승차일시"].dt.year == 2025
    ]
    .dropna(
        subset=[
            "승차일시",
            "출발구",
            "목적구"
        ]
    )
    .copy()
)

trip["요일"] = (
    trip["승차일시"]
    .dt.dayofweek
    .map(DAY_MAP)
)

trip["승차시간대"] = (
    trip["승차일시"].dt.hour
)

print("========== OD 데이터 로드 ==========")
print(f"전체 운행: {len(trip):,}건")


# =========================================================
# 3. 시간대별 OD 확률 계산
# =========================================================
# P(목적구 | 요일, 시간대, 출발구)

od = (
    trip
    .groupby(
        [
            "요일",
            "승차시간대",
            "출발구",
            "목적구"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "OD건수"
        }
    )
)

od["출발구총건수"] = (
    od
    .groupby(
        [
            "요일",
            "승차시간대",
            "출발구"
        ]
    )["OD건수"]
    .transform("sum")
)

od["OD확률"] = (
    od["OD건수"]
    / od["출발구총건수"]
)

print(f"OD 조합: {len(od):,}개")


# =========================================================
# 4. 기존 UD 배치 결과 로드
# =========================================================

allocation = pd.read_csv(
    ALLOCATION_PATH,
    encoding="utf-8-sig"
)

allocation = (
    allocation[
        allocation["UD배치대수"] > 0
    ]
    .copy()
)

print("\n========== 기존 UD 배치 ==========")
print(f"UD 투입 조합: {len(allocation):,}개")


# =========================================================
# 5. UD 운행 후 예상 차량 위치 계산
# =========================================================
# 특정 출발구에 배치된 UD 차량이
# 해당 지역의 기존 OD 이동 패턴을 따른다고 가정
#
# 예:
# 노원구 → 노원구 55%
# 노원구 → 도봉구 9%
# ...
#
# UD 3대가 노원구에서 운행한다면
# OD 확률에 따라 목적구별 예상 차량 위치를 계산

vehicle_flow = (
    allocation[
        [
            "요일",
            "승차시간대",
            "출발구",
            "UD배치대수"
        ]
    ]
    .merge(
        od[
            [
                "요일",
                "승차시간대",
                "출발구",
                "목적구",
                "OD확률"
            ]
        ],
        on=[
            "요일",
            "승차시간대",
            "출발구"
        ],
        how="left"
    )
)

vehicle_flow["예상도착UD"] = (
    vehicle_flow["UD배치대수"]
    * vehicle_flow["OD확률"]
)

vehicle_flow = (
    vehicle_flow
    .dropna(
        subset=["예상도착UD"]
    )
    .copy()
)


# =========================================================
# 6. 시간대 종료 후 목적구별 예상 차량 수
# =========================================================

arrivals = (
    vehicle_flow
    .groupby(
        [
            "요일",
            "승차시간대",
            "목적구"
        ],
        as_index=False
    )
    ["예상도착UD"]
    .sum()
)

# 목적구는 다음 시간대의 차량 출발 위치가 됨
arrivals = arrivals.rename(
    columns={
        "목적구": "출발구",
        "예상도착UD": "이전시간대도착UD"
    }
)


# =========================================================
# 7. t시간 운행 → t+1시간 공급으로 이동
# =========================================================

arrivals["승차시간대"] += 1

# 23시 → 다음 날 0시
next_day_mask = (
    arrivals["승차시간대"] == 24
)

arrivals.loc[
    next_day_mask,
    "승차시간대"
] = 0

arrivals.loc[
    next_day_mask,
    "요일"
] = (
    arrivals.loc[
        next_day_mask,
        "요일"
    ]
    .map(NEXT_DAY)
)


# =========================================================
# 8. 다음 시간대 목표 배치와 실제 예상 위치 비교
# =========================================================

target = allocation[
    [
        "요일",
        "승차시간대",
        "출발구",
        "UD배치대수",
        "최종배치점수",
        "배치유형"
    ]
].copy()

compare = target.merge(
    arrivals,
    on=[
        "요일",
        "승차시간대",
        "출발구"
    ],
    how="left"
)

compare["이전시간대도착UD"] = (
    compare["이전시간대도착UD"]
    .fillna(0)
)


# =========================================================
# 9. 재배치 필요 차량 계산
# =========================================================

compare["차량위치차이"] = (
    compare["UD배치대수"]
    - compare["이전시간대도착UD"]
)

# 목표 차량보다 부족한 경우
compare["재배치필요UD"] = (
    compare["차량위치차이"]
    .clip(lower=0)
)

# 목표보다 차량이 많이 도착한 경우
compare["잉여UD"] = (
    (-compare["차량위치차이"])
    .clip(lower=0)
)


# =========================================================
# 10. 시간대별 위치 일치도
# =========================================================

hourly = (
    compare
    .groupby(
        [
            "요일",
            "승차시간대"
        ],
        as_index=False
    )
    .agg(
        목표UD배치=(
            "UD배치대수",
            "sum"
        ),
        자연도착UD=(
            "이전시간대도착UD",
            "sum"
        ),
        재배치필요UD=(
            "재배치필요UD",
            "sum"
        ),
        잉여UD=(
            "잉여UD",
            "sum"
        )
    )
)

hourly["자연배치충족UD"] = (
    hourly["목표UD배치"]
    - hourly["재배치필요UD"]
).clip(lower=0)

hourly["위치일치율_%"] = np.where(
    hourly["목표UD배치"] > 0,
    (
        hourly["자연배치충족UD"]
        / hourly["목표UD배치"]
        * 100
    ),
    0
)

hourly["재배치필요율_%"] = np.where(
    hourly["목표UD배치"] > 0,
    (
        hourly["재배치필요UD"]
        / hourly["목표UD배치"]
        * 100
    ),
    0
)


# =========================================================
# 11. 12대 제약 검증
# =========================================================

print("\n========== 시간대별 UD 위치 검증 ==========")

print(
    hourly["목표UD배치"]
    .describe()
    .round(2)
)

over_12 = (
    hourly["목표UD배치"] > TOTAL_UD
).sum()

print(
    f"\n12대 초과: {over_12}개 시간대"
)


# =========================================================
# 12. 재배치가 많이 필요한 시간대
# =========================================================

print(
    "\n========== 재배치 필요 시간대 TOP 30 =========="
)

cols = [
    "요일",
    "승차시간대",
    "목표UD배치",
    "자연도착UD",
    "자연배치충족UD",
    "재배치필요UD",
    "잉여UD",
    "위치일치율_%",
    "재배치필요율_%"
]

print(
    hourly
    .sort_values(
        "재배치필요UD",
        ascending=False
    )
    .head(30)[cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 13. 요일별 요약
# =========================================================

daily = (
    hourly
    .groupby(
        "요일",
        as_index=False
    )
    .agg(
        평균목표UD=(
            "목표UD배치",
            "mean"
        ),
        평균자연도착UD=(
            "자연도착UD",
            "mean"
        ),
        평균재배치필요UD=(
            "재배치필요UD",
            "mean"
        ),
        평균위치일치율=(
            "위치일치율_%",
            "mean"
        ),
        평균재배치필요율=(
            "재배치필요율_%",
            "mean"
        )
    )
)

daily["요일순서"] = (
    daily["요일"].map(DAY_ORDER)
)

daily = (
    daily
    .sort_values("요일순서")
    .drop(columns="요일순서")
)


print(
    "\n========== 요일별 차량 위치 연속성 =========="
)

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 14. 전체 요약
# =========================================================

active = hourly[
    hourly["목표UD배치"] > 0
].copy()

print(
    "\n========== 전체 OD 재배치 분석 =========="
)

print(
    f"평균 위치 일치율: "
    f"{active['위치일치율_%'].mean():.2f}%"
)

print(
    f"평균 재배치 필요율: "
    f"{active['재배치필요율_%'].mean():.2f}%"
)

print(
    f"시간당 평균 재배치 필요 차량: "
    f"{active['재배치필요UD'].mean():.2f}대"
)


# =========================================================
# 15. 저장
# =========================================================

DETAIL_PATH = (
    RESULT_DIR
    / "od_rebalancing_detail_2025.csv"
)

HOURLY_PATH = (
    RESULT_DIR
    / "od_rebalancing_hourly_2025.csv"
)

DAILY_PATH = (
    RESULT_DIR
    / "od_rebalancing_daily_2025.csv"
)

compare.to_csv(
    DETAIL_PATH,
    index=False,
    encoding="utf-8-sig"
)

hourly.to_csv(
    HOURLY_PATH,
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    DAILY_PATH,
    index=False,
    encoding="utf-8-sig"
)


print("\n========== 저장 완료 ==========")

print(DETAIL_PATH)
print(HOURLY_PATH)
print(DAILY_PATH)