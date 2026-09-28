from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"

ALLOCATION_PATH = RESULT_DIR / "ud_allocation_2025.csv"

TOTAL_UD = 12
DEST_COL = "목적구"

DAY_MAP = {
    0: "월", 1: "화", 2: "수", 3: "목",
    4: "금", 5: "토", 6: "일"
}

DAY_ORDER = {
    "월": 0, "화": 1, "수": 2, "목": 3,
    "금": 4, "토": 5, "일": 6
}


# =========================================================
# 2. 실제 운행 데이터 로드
# =========================================================

usecols = [
    "승차일시",
    "하차일시",
    "출발구",
    DEST_COL
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

trip = trip[
    trip["승차일시"].dt.year == 2025
].dropna(
    subset=[
        "승차일시",
        "하차일시",
        "출발구",
        DEST_COL
    ]
).copy()


# =========================================================
# 3. 운행시간 품질 필터
# =========================================================

trip["운행시간_분"] = (
    trip["하차일시"] - trip["승차일시"]
).dt.total_seconds() / 60

trip = trip[
    (trip["운행시간_분"] > 0)
    & (trip["운행시간_분"] <= 180)
].copy()

trip["출발요일번호"] = trip["승차일시"].dt.dayofweek
trip["출발요일"] = (
    trip["출발요일번호"].map(DAY_MAP)
)

trip["출발시간대"] = trip["승차일시"].dt.hour

# 실제 하차 시점
trip["도착요일번호"] = trip["하차일시"].dt.dayofweek
trip["도착요일"] = (
    trip["도착요일번호"].map(DAY_MAP)
)

trip["도착시간대"] = trip["하차일시"].dt.hour


print("========== 실제 운행 데이터 ==========")
print(f"분석 운행: {len(trip):,}건")

print(
    f"평균 운행시간: "
    f"{trip['운행시간_분'].mean():.2f}분"
)


# =========================================================
# 4. 하차시간을 반영한 차량 이동확률
# =========================================================
# P(
#   목적구, 도착요일, 도착시간대
#   | 출발요일, 출발시간대, 출발구
# )

flow = (
    trip.groupby(
        [
            "출발요일",
            "출발시간대",
            "출발구",
            "도착요일",
            "도착시간대",
            DEST_COL
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "운행건수"})
)

flow["출발조합총건수"] = (
    flow.groupby(
        [
            "출발요일",
            "출발시간대",
            "출발구"
        ]
    )["운행건수"]
    .transform("sum")
)

flow["이동확률"] = (
    flow["운행건수"]
    / flow["출발조합총건수"]
)

print(
    f"차량 이동 조합: {len(flow):,}개"
)


# =========================================================
# 5. UD 동적배치 결과
# =========================================================

allocation = pd.read_csv(
    ALLOCATION_PATH,
    encoding="utf-8-sig"
)

allocation = allocation[
    allocation["UD배치대수"] > 0
].copy()

print("\n========== UD 동적배치 ==========")
print(
    f"UD 투입 조합: {len(allocation):,}개"
)


# =========================================================
# 6. 배치된 UD의 실제 하차시점 예상
# =========================================================

vehicle_flow = allocation[
    [
        "요일",
        "승차시간대",
        "출발구",
        "UD배치대수"
    ]
].merge(
    flow[
        [
            "출발요일",
            "출발시간대",
            "출발구",
            "도착요일",
            "도착시간대",
            DEST_COL,
            "이동확률"
        ]
    ],
    left_on=[
        "요일",
        "승차시간대",
        "출발구"
    ],
    right_on=[
        "출발요일",
        "출발시간대",
        "출발구"
    ],
    how="left"
)

vehicle_flow["예상도착UD"] = (
    vehicle_flow["UD배치대수"]
    * vehicle_flow["이동확률"]
)

vehicle_flow = vehicle_flow.dropna(
    subset=["예상도착UD"]
).copy()


# =========================================================
# 7. 실제 하차 시간대별 예상 차량 위치
# =========================================================

arrivals = (
    vehicle_flow.groupby(
        [
            "도착요일",
            "도착시간대",
            DEST_COL
        ],
        as_index=False
    )["예상도착UD"]
    .sum()
    .rename(
        columns={
            "도착요일": "요일",
            "도착시간대": "승차시간대",
            DEST_COL: "출발구",
            "예상도착UD": "도착예상UD"
        }
    )
)


# =========================================================
# 8. 해당 시간대 목표 배치와 비교
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

compare["도착예상UD"] = (
    compare["도착예상UD"]
    .fillna(0)
)


# =========================================================
# 9. 자연 공급 / 부족 / 잉여 계산
# =========================================================

# 목표 배치량보다 많이 도착했다고 해서
# 목표 충족량이 증가하지 않도록 제한
compare["자연공급UD"] = np.minimum(
    compare["UD배치대수"],
    compare["도착예상UD"]
)

compare["재배치필요UD"] = (
    compare["UD배치대수"]
    - compare["자연공급UD"]
).clip(lower=0)

compare["잉여UD"] = (
    compare["도착예상UD"]
    - compare["UD배치대수"]
).clip(lower=0)


# =========================================================
# 10. 시간대별 집계
# =========================================================

hourly = (
    compare.groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        목표UD배치=("UD배치대수", "sum"),
        도착예상UD=("도착예상UD", "sum"),
        자연공급UD=("자연공급UD", "sum"),
        재배치필요UD=("재배치필요UD", "sum"),
        잉여UD=("잉여UD", "sum")
    )
)

hourly["위치일치율_%"] = np.where(
    hourly["목표UD배치"] > 0,
    hourly["자연공급UD"]
    / hourly["목표UD배치"]
    * 100,
    0
)

hourly["재배치필요율_%"] = np.where(
    hourly["목표UD배치"] > 0,
    hourly["재배치필요UD"]
    / hourly["목표UD배치"]
    * 100,
    0
)


# =========================================================
# 11. 결과 검증
# =========================================================

print("\n========== 하차시간 기반 위치 검증 ==========")

print(
    hourly["목표UD배치"]
    .describe()
    .round(2)
)

print(
    "\n12대 초과:",
    (hourly["목표UD배치"] > TOTAL_UD).sum(),
    "개 시간대"
)


# =========================================================
# 12. 재배치 필요 시간대 TOP 30
# =========================================================

cols = [
    "요일",
    "승차시간대",
    "목표UD배치",
    "도착예상UD",
    "자연공급UD",
    "재배치필요UD",
    "잉여UD",
    "위치일치율_%",
    "재배치필요율_%"
]

print(
    "\n========== 재배치 필요 시간대 TOP 30 =========="
)

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
# 13. 요일별 분석
# =========================================================

daily = (
    hourly.groupby(
        "요일",
        as_index=False
    )
    .agg(
        평균목표UD=("목표UD배치", "mean"),
        평균도착예상UD=("도착예상UD", "mean"),
        평균자연공급UD=("자연공급UD", "mean"),
        평균재배치필요UD=("재배치필요UD", "mean"),
        평균잉여UD=("잉여UD", "mean"),
        평균위치일치율=("위치일치율_%", "mean"),
        평균재배치필요율=("재배치필요율_%", "mean")
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
# 14. 전체 결과
# =========================================================

active = hourly[
    hourly["목표UD배치"] > 0
].copy()

print(
    "\n========== 전체 하차시간 기반 재배치 분석 =========="
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

print(
    f"시간당 평균 잉여 차량: "
    f"{active['잉여UD'].mean():.2f}대"
)


# =========================================================
# 15. v1과 비교
# =========================================================

V1_PATH = (
    RESULT_DIR
    / "od_rebalancing_hourly_2025.csv"
)

if V1_PATH.exists():

    v1 = pd.read_csv(
        V1_PATH,
        encoding="utf-8-sig"
    )

    v1_active = v1[
        v1["목표UD배치"] > 0
    ]

    v1_match = (
        v1_active["위치일치율_%"].mean()
    )

    v1_rebalance = (
        v1_active["재배치필요율_%"].mean()
    )

    v2_match = (
        active["위치일치율_%"].mean()
    )

    v2_rebalance = (
        active["재배치필요율_%"].mean()
    )

    print(
        "\n========== v1 vs v2 =========="
    )

    print(
        f"v1 위치 일치율: {v1_match:.2f}%"
    )

    print(
        f"v2 위치 일치율: {v2_match:.2f}%"
    )

    print(
        f"변화: "
        f"{v2_match - v1_match:+.2f}%p"
    )

    print()

    print(
        f"v1 재배치 필요율: "
        f"{v1_rebalance:.2f}%"
    )

    print(
        f"v2 재배치 필요율: "
        f"{v2_rebalance:.2f}%"
    )

    print(
        f"변화: "
        f"{v2_rebalance - v1_rebalance:+.2f}%p"
    )


# =========================================================
# 16. 저장
# =========================================================

vehicle_flow.to_csv(
    RESULT_DIR / "od_vehicle_flow_v2_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

compare.to_csv(
    RESULT_DIR / "od_rebalancing_detail_v2_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

hourly.to_csv(
    RESULT_DIR / "od_rebalancing_hourly_v2_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    RESULT_DIR / "od_rebalancing_daily_v2_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n========== 저장 완료 =========="
)

print(
    RESULT_DIR
    / "od_rebalancing_hourly_v2_2025.csv"
)