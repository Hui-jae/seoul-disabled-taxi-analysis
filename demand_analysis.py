from pathlib import Path
import pandas as pd


# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

print("데이터 위치:", DATA_PATH)
print("파일 존재 여부:", DATA_PATH.exists())


# =========================================================
# 2. 데이터 불러오기
# =========================================================

usecols = [
    "승차일자",
    "승차시간대",
    "출발구",
    "출발동",
    "목적구"
]

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=usecols
)

df["승차일자"] = pd.to_datetime(
    df["승차일자"],
    errors="coerce"
)


# 2025년 승차 완료운행만 분석
df = df[
    df["승차일자"].dt.year == 2025
].copy()


print("\n========== 데이터 로드 완료 ==========")
print("전체 완료운행:", f"{len(df):,}건")
print(
    "분석 기간:",
    df["승차일자"].min().date(),
    "~",
    df["승차일자"].max().date()
)
print("출발구:", df["출발구"].nunique(), "개")


# =========================================================
# 3. 날짜 × 출발구 × 시간대별 완료운행
# =========================================================

daily_hourly = (
    df.groupby(
        ["승차일자", "출발구", "승차시간대"]
    )
    .size()
    .reset_index(name="완료운행건수")
)


# ---------------------------------------------------------
# 완료운행 0건인 조합도 포함
# ---------------------------------------------------------

all_dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31",
    freq="D"
)

all_gu = sorted(
    df["출발구"]
    .dropna()
    .unique()
)

all_hours = range(24)

full_index = pd.MultiIndex.from_product(
    [
        all_dates,
        all_gu,
        all_hours
    ],
    names=[
        "승차일자",
        "출발구",
        "승차시간대"
    ]
)

daily_hourly = (
    daily_hourly
    .set_index(
        ["승차일자", "출발구", "승차시간대"]
    )
    .reindex(
        full_index,
        fill_value=0
    )
    .reset_index()
)


# =========================================================
# 4. 요일 정보 추가
# =========================================================

day_map = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일"
}

day_order = [
    "월", "화", "수", "목", "금", "토", "일"
]

daily_hourly["요일"] = (
    daily_hourly["승차일자"]
    .dt.dayofweek
    .map(day_map)
)


# =========================================================
# 5. 요일 × 출발구 × 시간대 통계
# =========================================================

weekday_hour_gu = (
    daily_hourly
    .groupby(
        [
            "요일",
            "출발구",
            "승차시간대"
        ],
        as_index=False
    )
    .agg(
        평균완료운행건수=(
            "완료운행건수",
            "mean"
        ),
        중앙값완료운행건수=(
            "완료운행건수",
            "median"
        ),
        최대완료운행건수=(
            "완료운행건수",
            "max"
        )
    )
)

weekday_hour_gu["평균완료운행건수"] = (
    weekday_hour_gu["평균완료운행건수"]
    .round(2)
)

weekday_hour_gu["요일"] = pd.Categorical(
    weekday_hour_gu["요일"],
    categories=day_order,
    ordered=True
)

weekday_hour_gu = (
    weekday_hour_gu
    .sort_values(
        [
            "요일",
            "출발구",
            "승차시간대"
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# 6. 자치구별 피크 요일·시간대
# =========================================================

gu_peak = (
    weekday_hour_gu
    .sort_values(
        "평균완료운행건수",
        ascending=False
    )
    .groupby(
        "출발구",
        observed=True
    )
    .first()
    .reset_index()
)

gu_peak = gu_peak[
    [
        "출발구",
        "요일",
        "승차시간대",
        "평균완료운행건수",
        "중앙값완료운행건수",
        "최대완료운행건수"
    ]
]

gu_peak = (
    gu_peak
    .sort_values(
        "평균완료운행건수",
        ascending=False
    )
    .reset_index(drop=True)
)


# =========================================================
# 7. 출발동별 완료운행 분석
# =========================================================

dong_total = (
    df.groupby(
        ["출발구", "출발동"]
    )
    .size()
    .reset_index(name="완료운행건수")
)


# 자치구별 전체 완료운행 건수
gu_total = (
    dong_total
    .groupby(
        "출발구",
        as_index=False
    )
    .agg(
        구전체운행건수=(
            "완료운행건수",
            "sum"
        )
    )
)


dong_total = dong_total.merge(
    gu_total,
    on="출발구",
    how="left"
)


# 각 동이 해당 자치구에서 차지하는 비율
dong_total["구내비율_%"] = (
    dong_total["완료운행건수"]
    / dong_total["구전체운행건수"]
    * 100
).round(2)


dong_total = (
    dong_total
    .sort_values(
        [
            "출발구",
            "완료운행건수"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)


# =========================================================
# 8. 출발구 → 목적구 OD 분석
# =========================================================

od_gu = (
    df.groupby(
        ["출발구", "목적구"]
    )
    .size()
    .reset_index(name="완료운행건수")
)


# ---------------------------------------------------------
# 25 × 25 전체 OD 조합 생성
# ---------------------------------------------------------

gu_list = sorted(
    df["출발구"]
    .dropna()
    .unique()
)

od_index = pd.MultiIndex.from_product(
    [
        gu_list,
        gu_list
    ],
    names=[
        "출발구",
        "목적구"
    ]
)

od_gu = (
    od_gu
    .set_index(
        ["출발구", "목적구"]
    )
    .reindex(
        od_index,
        fill_value=0
    )
    .reset_index()
)


# =========================================================
# 9. 출발구별 OD 비율 계산
# =========================================================

origin_total = (
    od_gu
    .groupby(
        "출발구"
    )["완료운행건수"]
    .transform("sum")
)

od_gu["출발구내비율_%"] = (
    od_gu["완료운행건수"]
    / origin_total
    * 100
).round(2)


# =========================================================
# 10. 동일구 / 타구 이동 계산
# =========================================================

same_gu = od_gu.loc[
    od_gu["출발구"] == od_gu["목적구"],
    "완료운행건수"
].sum()

different_gu = od_gu.loc[
    od_gu["출발구"] != od_gu["목적구"],
    "완료운행건수"
].sum()

total_od = same_gu + different_gu


# =========================================================
# 11. 25 × 25 OD 행렬 생성
# =========================================================

od_matrix = od_gu.pivot(
    index="출발구",
    columns="목적구",
    values="완료운행건수"
)


# =========================================================
# 12. 분석 결과 저장
# =========================================================

daily_hourly.to_csv(
    RESULT_DIR / "daily_hourly_gu_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

weekday_hour_gu.to_csv(
    RESULT_DIR / "weekday_hour_gu_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

gu_peak.to_csv(
    RESULT_DIR / "gu_peak_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

dong_total.to_csv(
    RESULT_DIR / "dong_total_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

od_gu.to_csv(
    RESULT_DIR / "od_gu_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

od_matrix.to_csv(
    RESULT_DIR / "od_matrix_2025.csv",
    encoding="utf-8-sig"
)


# =========================================================
# 13. 최종 결과 확인
# =========================================================

print("\n========== 분석 완료 ==========")

print(
    "날짜 × 구 × 시간대:",
    f"{len(daily_hourly):,}행"
)

print(
    "요일 × 구 × 시간대:",
    f"{len(weekday_hour_gu):,}행"
)

print(
    "자치구별 피크:",
    f"{len(gu_peak):,}행"
)

print(
    "출발동:",
    f"{len(dong_total):,}행"
)

print(
    "OD:",
    f"{len(od_gu):,}행"
)


print("\n========== 동일구 / 타구 이동 ==========")

print(
    "동일구 이동:",
    f"{same_gu:,}건",
    f"({same_gu / total_od * 100:.2f}%)"
)

print(
    "타구 이동:",
    f"{different_gu:,}건",
    f"({different_gu / total_od * 100:.2f}%)"
)


print("\n========== 자치구별 피크 ==========")

print(
    gu_peak.to_string(index=False)
)


print("\n========== OD 상위 20개 ==========")

print(
    od_gu
    .sort_values(
        "완료운행건수",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


print("\n========== 결과 저장 완료 ==========")

print("저장 위치:", RESULT_DIR)

print(
    """
생성 파일:
- daily_hourly_gu_2025.csv
- weekday_hour_gu_2025.csv
- gu_peak_2025.csv
- dong_total_2025.csv
- od_gu_2025.csv
- od_matrix_2025.csv
"""
)