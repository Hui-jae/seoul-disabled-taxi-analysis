from pathlib import Path
import pandas as pd
import numpy as np


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

HOURLY_PATH = (
    RESULT_DIR / "rebalancing_hourly_v2_2025.csv"
)

ROUTES_PATH = (
    RESULT_DIR / "rebalancing_routes_v2_2025.csv"
)

OUTPUT_PATH = (
    RESULT_DIR / "rebalancing_failure_analysis_2025.csv"
)

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
# 2. 데이터 로드
# =========================================================

hourly = pd.read_csv(
    HOURLY_PATH,
    encoding="utf-8-sig"
)

routes = pd.read_csv(
    ROUTES_PATH,
    encoding="utf-8-sig"
)

print("========== 데이터 로드 ==========")
print(f"시간대 분석 조합: {len(hourly):,}개")
print(f"최적 재배치 경로: {len(routes):,}개")


# =========================================================
# 3. 필요한 컬럼 확인
# =========================================================

required_cols = [
    "요일",
    "승차시간대",
    "필요재배치UD",
    "가용잉여UD",
    "최적재배치UD",
    "재배치후미충족UD",
    "평균재배치시간_분",
    "재배치충족률_%"
]

missing = [
    col for col in required_cols
    if col not in hourly.columns
]

if missing:
    raise KeyError(
        f"hourly 파일에 필요한 컬럼이 없습니다: {missing}"
    )


# =========================================================
# 4. 실제 재배치가 필요한 시간대만 분석
# =========================================================

failure = hourly[
    hourly["필요재배치UD"] > 0
].copy()

print("\n========== 미충족 분석 대상 ==========")
print(f"재배치 필요 시간대: {len(failure):,}개")


# =========================================================
# 5. 기본 부족 지표
# =========================================================

failure["잉여차량부족량"] = (
    failure["필요재배치UD"]
    - failure["가용잉여UD"]
).clip(lower=0)

failure["공간시간미충족량"] = (
    failure["재배치후미충족UD"]
    - failure["잉여차량부족량"]
).clip(lower=0)


# =========================================================
# 6. 미충족 원인 분류
# =========================================================
#
# 1) 공급부족
#    필요한 재배치 차량보다 가용 잉여차량 자체가 적음
#
# 2) 공간·시간 제약
#    잉여차량은 충분하지만 적절한 위치에서
#    필요한 지역까지 효율적으로 이동시키기 어려움
#
# 3) 복합제약
#    공급부족과 공간·시간 제약이 동시에 존재
#
# 4) 완전충족
#    미충족 없음
# =========================================================

conditions = [
    failure["재배치후미충족UD"] <= 1e-6,

    (
        (failure["잉여차량부족량"] > 1e-6)
        & (failure["공간시간미충족량"] <= 1e-6)
    ),

    (
        (failure["잉여차량부족량"] <= 1e-6)
        & (failure["공간시간미충족량"] > 1e-6)
    ),

    (
        (failure["잉여차량부족량"] > 1e-6)
        & (failure["공간시간미충족량"] > 1e-6)
    )
]

choices = [
    "완전충족",
    "공급부족",
    "공간시간제약",
    "복합제약"
]

failure["미충족원인"] = np.select(
    conditions,
    choices,
    default="기타"
)


# =========================================================
# 7. 원인별 통계
# =========================================================

cause_summary = (
    failure
    .groupby(
        "미충족원인",
        as_index=False
    )
    .agg(
        시간대수=("미충족원인", "size"),
        평균필요재배치=("필요재배치UD", "mean"),
        평균가용잉여=("가용잉여UD", "mean"),
        평균최적재배치=("최적재배치UD", "mean"),
        평균미충족=("재배치후미충족UD", "mean"),
        총미충족=("재배치후미충족UD", "sum"),
        평균충족률=("재배치충족률_%", "mean"),
        평균이동시간=("평균재배치시간_분", "mean")
    )
)

cause_summary["시간대비율_%"] = (
    cause_summary["시간대수"]
    / cause_summary["시간대수"].sum()
    * 100
)

total_unmet = (
    cause_summary["총미충족"].sum()
)

cause_summary["미충족기여율_%"] = np.where(
    total_unmet > 0,
    cause_summary["총미충족"]
    / total_unmet
    * 100,
    0
)

cause_summary = cause_summary.round(2)

print("\n========== 미충족 원인별 분석 ==========")
print(
    cause_summary
    .sort_values(
        "총미충족",
        ascending=False
    )
    .to_string(index=False)
)


# =========================================================
# 8. 미충족이 심한 시간대
# =========================================================

failure["미충족률_%"] = np.where(
    failure["필요재배치UD"] > 0,
    failure["재배치후미충족UD"]
    / failure["필요재배치UD"]
    * 100,
    0
)

print("\n========== 미충족 심각 시간대 TOP 30 ==========")

cols = [
    "요일",
    "승차시간대",
    "필요재배치UD",
    "가용잉여UD",
    "최적재배치UD",
    "재배치후미충족UD",
    "미충족률_%",
    "평균재배치시간_분",
    "미충족원인"
]

print(
    failure
    .sort_values(
        [
            "재배치후미충족UD",
            "미충족률_%"
        ],
        ascending=False
    )
    .head(30)[cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 9. 시간대별 반복 취약성
# =========================================================

hour_summary = (
    failure
    .groupby(
        "승차시간대",
        as_index=False
    )
    .agg(
        발생요일수=("요일", "nunique"),
        평균필요재배치=("필요재배치UD", "mean"),
        평균최적재배치=("최적재배치UD", "mean"),
        평균미충족=("재배치후미충족UD", "mean"),
        총미충족=("재배치후미충족UD", "sum"),
        평균충족률=("재배치충족률_%", "mean")
    )
    .sort_values(
        "총미충족",
        ascending=False
    )
)

print("\n========== 반복 취약 시간대 TOP 15 ==========")

print(
    hour_summary
    .head(15)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 10. 요일별 미충족
# =========================================================

daily = (
    failure
    .groupby(
        "요일",
        as_index=False
    )
    .agg(
        분석시간대=("요일", "size"),
        평균필요재배치=("필요재배치UD", "mean"),
        평균가용잉여=("가용잉여UD", "mean"),
        평균최적재배치=("최적재배치UD", "mean"),
        평균미충족=("재배치후미충족UD", "mean"),
        총미충족=("재배치후미충족UD", "sum"),
        평균충족률=("재배치충족률_%", "mean")
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

print("\n========== 요일별 미충족 ==========")

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 11. 경로 데이터로 공간적 제약 분석
# =========================================================

if {
    "재배치출발구",
    "재배치도착구",
    "재배치대수",
    "평균재배치시간_분"
}.issubset(routes.columns):

    route_summary = (
        routes
        .groupby(
            [
                "재배치출발구",
                "재배치도착구"
            ],
            as_index=False
        )
        .agg(
            총재배치UD=("재배치대수", "sum"),
            평균이동시간_분=(
                "평균재배치시간_분",
                "mean"
            )
        )
    )

    route_summary["총차량시간_분"] = (
        route_summary["총재배치UD"]
        * route_summary["평균이동시간_분"]
    )

    print(
        "\n========== 재배치 부담이 큰 경로 TOP 20 =========="
    )

    print(
        route_summary
        .sort_values(
            "총차량시간_분",
            ascending=False
        )
        .head(20)
        .round(2)
        .to_string(index=False)
    )

else:
    route_summary = pd.DataFrame()


# =========================================================
# 12. 전체 원인 분해
# =========================================================

total_need = (
    failure["필요재배치UD"].sum()
)

total_allocated = (
    failure["최적재배치UD"].sum()
)

total_unmet = (
    failure["재배치후미충족UD"].sum()
)

total_supply_shortage = (
    failure["잉여차량부족량"].sum()
)

total_spatial_shortage = (
    failure["공간시간미충족량"].sum()
)


print("\n========== 전체 미충족 원인 분해 ==========")

print(
    f"총 필요 재배치량: "
    f"{total_need:.2f}대·회"
)

print(
    f"총 최적 재배치량: "
    f"{total_allocated:.2f}대·회"
)

print(
    f"총 미충족량: "
    f"{total_unmet:.2f}대·회"
)

if total_unmet > 0:

    print(
        f"공급 부족 기여: "
        f"{total_supply_shortage:.2f}대·회 "
        f"({total_supply_shortage / total_unmet * 100:.2f}%)"
    )

    print(
        f"공간·시간 제약 기여: "
        f"{total_spatial_shortage:.2f}대·회 "
        f"({total_spatial_shortage / total_unmet * 100:.2f}%)"
    )


# =========================================================
# 13. 정책 해석용 지표
# =========================================================

failure_hours = (
    failure["재배치후미충족UD"] > 1e-6
).sum()

complete_hours = (
    failure["재배치후미충족UD"] <= 1e-6
).sum()

print("\n========== 정책 해석 지표 ==========")

print(
    f"재배치 필요 시간대: "
    f"{len(failure):,}개"
)

print(
    f"100% 충족 시간대: "
    f"{complete_hours:,}개 "
    f"({complete_hours / len(failure) * 100:.2f}%)"
)

print(
    f"미충족 발생 시간대: "
    f"{failure_hours:,}개 "
    f"({failure_hours / len(failure) * 100:.2f}%)"
)

print(
    f"전체 차량 기준 재배치 충족률: "
    f"{total_allocated / total_need * 100:.2f}%"
)


# =========================================================
# 14. 저장
# =========================================================

failure.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

cause_summary.to_csv(
    RESULT_DIR
    / "rebalancing_failure_causes_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

hour_summary.to_csv(
    RESULT_DIR
    / "rebalancing_failure_hour_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    RESULT_DIR
    / "rebalancing_failure_daily_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

if not route_summary.empty:

    route_summary.to_csv(
        RESULT_DIR
        / "rebalancing_failure_routes_2025.csv",
        index=False,
        encoding="utf-8-sig"
    )


print("\n========== 분석 완료 ==========")

print(
    "저장 위치:",
    OUTPUT_PATH
)