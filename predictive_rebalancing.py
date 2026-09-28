from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

STOCK_PATH = RESULT_DIR / "vehicle_stock_detail_2025.csv"
ROUTE_PATH = RESULT_DIR / "rebalancing_routes_v2_2025.csv"
HOURLY_V2_PATH = RESULT_DIR / "rebalancing_hourly_v2_2025.csv"

OUTPUT_HOURLY = RESULT_DIR / "predictive_rebalancing_hourly_2025.csv"
OUTPUT_DAILY = RESULT_DIR / "predictive_rebalancing_daily_2025.csv"
OUTPUT_SUMMARY = RESULT_DIR / "predictive_rebalancing_summary_2025.csv"

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
# 2. 차량 재고 데이터 로드
# =========================================================

stock = pd.read_csv(
    STOCK_PATH,
    encoding="utf-8-sig"
)

print("========== 차량 재고 데이터 ==========")
print(f"분석 조합: {len(stock):,}개")

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

missing_cols = [
    col for col in required_cols
    if col not in stock.columns
]

if missing_cols:
    raise ValueError(
        f"vehicle_stock_hourly_2025.csv에 "
        f"다음 컬럼이 없습니다: {missing_cols}"
    )


# =========================================================
# 3. 기존 v2 재배치 결과 로드
# =========================================================

baseline = pd.read_csv(
    HOURLY_V2_PATH,
    encoding="utf-8-sig"
)

print("\n========== 기존 최소비용 재배치 ==========")
print(f"분석 시간대: {len(baseline):,}개")


# =========================================================
# 4. 다음 시간대 정보 생성
# =========================================================
# 현재 t 시점에서 t+1의 부족 지역을 미리 파악하기 위한 키 생성

stock["다음요일"] = stock["요일"]
stock["다음시간대"] = stock["승차시간대"] + 1

next_day_mask = stock["다음시간대"] >= 24

stock.loc[
    next_day_mask,
    "다음시간대"
] = 0

stock.loc[
    next_day_mask,
    "다음요일"
] = (
    stock.loc[
        next_day_mask,
        "요일"
    ].map(NEXT_DAY)
)


# =========================================================
# 5. 다음 시간대 부족량 가져오기
# =========================================================
# t 시간대 행에 t+1 시간대의 부족량을 결합
#
# 예:
# 월요일 16시 차량 상태
# -> 월요일 17시 예상 부족량을 확인
#
# 여기서는 2025년 반복 패턴을 기반으로 한
# 시간대별 평균 부족량을 '예측값'으로 사용

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
        "부족UD": "다음시간대예상부족UD"
    }
)

predictive = stock.merge(
    future_need,
    on=[
        "다음요일",
        "다음시간대",
        "출발구"
    ],
    how="left"
)

predictive["다음시간대예상부족UD"] = (
    predictive["다음시간대예상부족UD"]
    .fillna(0)
)


# =========================================================
# 6. 현재 시간대 잉여 차량 산정
# =========================================================
# 현재 지역에서 목표 배치보다 많이 남는 차량을
# 다음 시간대 선제 재배치 후보로 사용

predictive["선제재배치가용UD"] = (
    predictive["잉여UD"]
    .clip(lower=0)
)


# =========================================================
# 7. 선제 재배치 수요 산정
# =========================================================
# 다음 시간대 예상 부족량이 있는 지역만 선제 배치 대상

predictive["선제재배치필요UD"] = (
    predictive["다음시간대예상부족UD"]
    .clip(lower=0)
)


# =========================================================
# 8. 시간대별 총 잉여 / 다음 시간대 부족 계산
# =========================================================

current_supply = (
    predictive.groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        현재가용잉여UD=(
            "선제재배치가용UD",
            "sum"
        )
    )
)

future_demand = (
    predictive.groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        다음시간대예상부족UD=(
            "선제재배치필요UD",
            "sum"
        )
    )
)

hourly_predictive = current_supply.merge(
    future_demand,
    on=["요일", "승차시간대"],
    how="outer"
)

hourly_predictive = hourly_predictive.fillna(0)


# =========================================================
# 9. 선제 재배치 가능량 계산
# =========================================================
# 현재 시간대의 잉여 차량과
# 다음 시간대 예상 부족량 중 작은 값만큼
# 선제적으로 이동 가능하다고 가정
#
# 이후 이동시간 현실성을 반영하기 위해
# 기존 v2의 평균 재배치 이동시간을 활용

hourly_predictive["이론적선제재배치UD"] = np.minimum(
    hourly_predictive["현재가용잉여UD"],
    hourly_predictive["다음시간대예상부족UD"]
)


# =========================================================
# 10. 기존 재배치 이동시간 정보 결합
# =========================================================

baseline_cols = [
    "요일",
    "승차시간대",
    "필요재배치UD",
    "최적재배치UD",
    "재배치후미충족UD",
    "평균재배치시간_분",
    "재배치충족률_%"
]

available_baseline_cols = [
    col for col in baseline_cols
    if col in baseline.columns
]

baseline_small = baseline[
    available_baseline_cols
].copy()

hourly_predictive = hourly_predictive.merge(
    baseline_small,
    on=["요일", "승차시간대"],
    how="left"
)

numeric_cols = [
    "필요재배치UD",
    "최적재배치UD",
    "재배치후미충족UD",
    "평균재배치시간_분",
    "재배치충족률_%"
]

for col in numeric_cols:
    if col in hourly_predictive.columns:
        hourly_predictive[col] = (
            hourly_predictive[col]
            .fillna(0)
        )


# =========================================================
# 11. 시간 제약을 고려한 선제 재배치
# =========================================================
# 한 시간 전에 미리 이동한다고 가정하므로
# 평균 이동시간이 60분 이내인 경우
# 이론적 선제 재배치량을 활용할 수 있다고 본다.
#
# 60분 이상이면 다음 시간대까지 도착하지 못할 가능성이
# 있으므로 시간 가중치를 적용한다.
#
# 이동시간 <= 30분 : 100%
# 30~60분          : 선형 감소
# >= 60분          : 0%

def time_weight(minutes):

    if pd.isna(minutes):
        return 0.0

    if minutes <= 30:
        return 1.0

    if minutes >= 60:
        return 0.0

    return (60 - minutes) / 30


hourly_predictive["시간가중치"] = (
    hourly_predictive[
        "평균재배치시간_분"
    ]
    .apply(time_weight)
)

hourly_predictive["시간제약반영_선제재배치UD"] = (
    hourly_predictive["이론적선제재배치UD"]
    * hourly_predictive["시간가중치"]
)


# =========================================================
# 12. 다음 시간대 기준으로 선제 재배치 효과 이동
# =========================================================
# 현재 t 시간대에 실시한 선제 재배치는
# t+1 시간대의 미충족을 감소시키므로
# 결과를 다음 시간대로 이동시킨다.

advance = hourly_predictive[
    [
        "요일",
        "승차시간대",
        "시간제약반영_선제재배치UD"
    ]
].copy()

advance["적용요일"] = advance["요일"]
advance["적용시간대"] = advance["승차시간대"] + 1

mask = advance["적용시간대"] >= 24

advance.loc[
    mask,
    "적용시간대"
] = 0

advance.loc[
    mask,
    "적용요일"
] = (
    advance.loc[
        mask,
        "요일"
    ].map(NEXT_DAY)
)

advance = advance.rename(
    columns={
        "시간제약반영_선제재배치UD":
        "선제공급UD"
    }
)

advance = advance[
    [
        "적용요일",
        "적용시간대",
        "선제공급UD"
    ]
]


# =========================================================
# 13. 기존 v2 결과에 선제 공급 적용
# =========================================================

result = baseline.copy()

result = result.merge(
    advance,
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

result["선제공급UD"] = (
    result["선제공급UD"]
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
# 14. 선제 재배치 후 필요량 계산
# =========================================================

result["선제재배치후_필요UD"] = (
    result["필요재배치UD"]
    - result["선제공급UD"]
).clip(lower=0)

result["실제선제효과UD"] = (
    result["필요재배치UD"]
    - result["선제재배치후_필요UD"]
)


# =========================================================
# 15. 기존 최적화와 결합
# =========================================================
# 선제 배치 후에도 남는 부족량에 대해
# 기존 최소비용 재배치가 수행된다고 가정
#
# 기존 최적 재배치 능력을 그대로 유지하되,
# 선제배치 이후 필요한 양까지만 사용

result["선제후_사후재배치UD"] = np.minimum(
    result["최적재배치UD"],
    result["선제재배치후_필요UD"]
)

result["최종미충족UD"] = (
    result["선제재배치후_필요UD"]
    - result["선제후_사후재배치UD"]
).clip(lower=0)


# =========================================================
# 16. 최종 충족률 계산
# =========================================================

result["선제재배치_총충족UD"] = (
    result["실제선제효과UD"]
    + result["선제후_사후재배치UD"]
)

result["선제재배치충족률_%"] = np.where(
    result["필요재배치UD"] > 0,
    (
        result["선제재배치_총충족UD"]
        / result["필요재배치UD"]
        * 100
    ),
    100
)

result["기존미충족UD"] = (
    result["재배치후미충족UD"]
)

result["미충족감소UD"] = (
    result["기존미충족UD"]
    - result["최종미충족UD"]
)

result["미충족감소율_%"] = np.where(
    result["기존미충족UD"] > 0,
    (
        result["미충족감소UD"]
        / result["기존미충족UD"]
        * 100
    ),
    0
)


# =========================================================
# 17. 사후 재배치 부담 감소
# =========================================================

result["사후재배치감소UD"] = (
    result["최적재배치UD"]
    - result["선제후_사후재배치UD"]
).clip(lower=0)

result["사후재배치감소율_%"] = np.where(
    result["최적재배치UD"] > 0,
    (
        result["사후재배치감소UD"]
        / result["최적재배치UD"]
        * 100
    ),
    0
)


# =========================================================
# 18. TOP 30 출력
# =========================================================

print("\n========== 선제 재배치 효과 TOP 30 ==========")

display_cols = [
    "요일",
    "승차시간대",
    "필요재배치UD",
    "실제선제효과UD",
    "선제후_사후재배치UD",
    "선제재배치_총충족UD",
    "최종미충족UD",
    "재배치충족률_%",
    "선제재배치충족률_%",
    "미충족감소UD"
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
# 19. 반복 취약 시간대 비교
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
        발생요일수=("요일", "nunique"),
        평균필요재배치=("필요재배치UD", "mean"),
        기존평균미충족=("기존미충족UD", "mean"),
        선제후평균미충족=("최종미충족UD", "mean"),
        평균선제효과=("실제선제효과UD", "mean"),
        기존평균충족률=("재배치충족률_%", "mean"),
        선제평균충족률=("선제재배치충족률_%", "mean")
    )
)

weak_hour["충족률개선_%p"] = (
    weak_hour["선제평균충족률"]
    - weak_hour["기존평균충족률"]
)

print("\n========== 시간대별 선제 재배치 효과 ==========")

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
# 20. 요일별 요약
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
        분석시간대=("승차시간대", "count"),
        평균필요재배치=("필요재배치UD", "mean"),
        평균선제공급=("실제선제효과UD", "mean"),
        평균사후재배치=("선제후_사후재배치UD", "mean"),
        기존평균미충족=("기존미충족UD", "mean"),
        선제후평균미충족=("최종미충족UD", "mean"),
        기존평균충족률=("재배치충족률_%", "mean"),
        선제평균충족률=("선제재배치충족률_%", "mean")
    )
)

daily["충족률개선_%p"] = (
    daily["선제평균충족률"]
    - daily["기존평균충족률"]
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

print("\n========== 요일별 선제 재배치 효과 ==========")

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 21. 전체 성능 계산
# =========================================================

active = result[
    result["필요재배치UD"] > 0
].copy()

total_need = (
    active["필요재배치UD"]
    .sum()
)

baseline_fulfilled = (
    active["최적재배치UD"]
    .sum()
)

predictive_fulfilled = (
    active["선제재배치_총충족UD"]
    .sum()
)

baseline_unmet = (
    active["기존미충족UD"]
    .sum()
)

predictive_unmet = (
    active["최종미충족UD"]
    .sum()
)

total_predictive = (
    active["실제선제효과UD"]
    .sum()
)

total_post = (
    active["선제후_사후재배치UD"]
    .sum()
)


# =========================================================
# 22. 전체 가중 충족률
# =========================================================

baseline_weighted_rate = (
    baseline_fulfilled
    / total_need
    * 100
    if total_need > 0
    else 0
)

predictive_weighted_rate = (
    predictive_fulfilled
    / total_need
    * 100
    if total_need > 0
    else 0
)

improvement = (
    predictive_weighted_rate
    - baseline_weighted_rate
)

unmet_reduction = (
    (
        baseline_unmet
        - predictive_unmet
    )
    / baseline_unmet
    * 100
    if baseline_unmet > 0
    else 0
)


# =========================================================
# 23. 결과 출력
# =========================================================

print("\n========== 전체 선제 재배치 결과 ==========")

print(
    f"총 필요 재배치량: "
    f"{total_need:.2f}대·회"
)

print(
    f"기존 최적 재배치량: "
    f"{baseline_fulfilled:.2f}대·회"
)

print(
    f"선제 재배치량: "
    f"{total_predictive:.2f}대·회"
)

print(
    f"선제 후 사후 재배치량: "
    f"{total_post:.2f}대·회"
)

print(
    f"기존 총 미충족량: "
    f"{baseline_unmet:.2f}대·회"
)

print(
    f"선제 적용 후 총 미충족량: "
    f"{predictive_unmet:.2f}대·회"
)

print(
    f"기존 전체 재배치량 기준 충족률: "
    f"{baseline_weighted_rate:.2f}%"
)

print(
    f"선제 재배치 적용 충족률: "
    f"{predictive_weighted_rate:.2f}%"
)

print(
    f"충족률 개선: "
    f"+{improvement:.2f}%p"
)

print(
    f"미충족량 감소율: "
    f"{unmet_reduction:.2f}%"
)


# =========================================================
# 24. 기존 시간대 평균 충족률과 비교
# =========================================================

baseline_mean_rate = (
    active["재배치충족률_%"]
    .mean()
)

predictive_mean_rate = (
    active["선제재배치충족률_%"]
    .mean()
)

print("\n========== 시간대 평균 기준 비교 ==========")

print(
    f"기존 시간대 평균 충족률: "
    f"{baseline_mean_rate:.2f}%"
)

print(
    f"선제 적용 시간대 평균 충족률: "
    f"{predictive_mean_rate:.2f}%"
)

print(
    f"시간대 평균 개선: "
    f"+{predictive_mean_rate - baseline_mean_rate:.2f}%p"
)


# =========================================================
# 25. 12대 제약 검증
# =========================================================
# 선제 재배치는 새로운 차량을 추가하는 것이 아니라
# 현재 잉여 차량의 위치를 이동시키는 것이므로
# 총 차량 수는 12대를 넘지 않아야 함

print("\n========== UD 12대 제약 ==========")

if "시간대시작UD" in active.columns:

    print(
        f"시간대 시작 UD 최대: "
        f"{active['시간대시작UD'].max():.2f}대"
    )

    print(
        "12대 초과:",
        (
            active["시간대시작UD"]
            > TOTAL_UD
        ).sum(),
        "개 시간대"
    )


# =========================================================
# 26. 요약 데이터 생성
# =========================================================

summary = pd.DataFrame(
    {
        "지표": [
            "총 필요 재배치량",
            "기존 최적 재배치량",
            "선제 재배치량",
            "선제 후 사후 재배치량",
            "기존 총 미충족량",
            "선제 후 총 미충족량",
            "기존 전체 충족률",
            "선제 적용 전체 충족률",
            "충족률 개선",
            "미충족량 감소율"
        ],
        "값": [
            total_need,
            baseline_fulfilled,
            total_predictive,
            total_post,
            baseline_unmet,
            predictive_unmet,
            baseline_weighted_rate,
            predictive_weighted_rate,
            improvement,
            unmet_reduction
        ]
    }
)


# =========================================================
# 27. 저장
# =========================================================

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

print("\n========== 저장 완료 ==========")

print(OUTPUT_HOURLY)
print(OUTPUT_DAILY)
print(OUTPUT_SUMMARY)