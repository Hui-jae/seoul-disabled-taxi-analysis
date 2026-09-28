from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

# =========================================================
# 1. 데이터 로드
# =========================================================

demand = pd.read_csv(
    RESULT_DIR / "weekday_hour_gu_2025.csv",
    encoding="utf-8-sig"
)

wait = pd.read_csv(
    RESULT_DIR / "wait_risk_2025.csv",
    encoding="utf-8-sig"
)

print("========== 데이터 로드 ==========")
print(f"수요 데이터: {len(demand):,}행")
print(f"대기 데이터: {len(wait):,}행")


# =========================================================
# 2. 수요 + 대기 데이터 결합
# =========================================================

df = demand.merge(
    wait,
    on=["요일", "출발구", "승차시간대"],
    how="left"
)

# 대기 데이터가 없는 조합 제외
df = df.dropna(
    subset=[
        "평균대기시간_분",
        "대기90분초과비율_%",
        "평균접수배차_분"
    ]
).copy()

print(f"\n분석 가능 조합: {len(df):,}개")

# =========================================================
# 3. 최소 표본 수 필터링
# =========================================================

MIN_CALLS = 30

before = len(df)

df = df[
    df["일반콜건수"] >= MIN_CALLS
].copy()

print("\n========== 표본 수 필터 ==========")
print(f"필터 전: {before:,}개")
print(f"일반콜 {MIN_CALLS}건 이상: {len(df):,}개")
print(f"제외: {before - len(df):,}개")


# =========================================================
# 4. Min-Max 정규화
# =========================================================

def normalize(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(0, index=series.index)

    return (series - min_value) / (max_value - min_value)


df["수요점수"] = normalize(
    df["평균완료운행건수"]
)

df["대기점수"] = normalize(
    df["평균대기시간_분"]
)

df["장기대기점수"] = normalize(
    df["대기90분초과비율_%"]
)

df["배차병목점수"] = normalize(
    df["평균접수배차_분"]
)


# =========================================================
# 5. UD택시 투입 우선순위 점수
# =========================================================

df["UD우선순위점수"] = (
    df["수요점수"] * 0.30
    + df["대기점수"] * 0.25
    + df["장기대기점수"] * 0.25
    + df["배차병목점수"] * 0.20
) * 100

df["UD우선순위점수"] = df["UD우선순위점수"].round(2)


# =========================================================
# 6. 순위
# =========================================================

df = df.sort_values(
    "UD우선순위점수",
    ascending=False
).reset_index(drop=True)

df["우선순위"] = range(1, len(df) + 1)


# =========================================================
# 7. 상위 30개 출력
# =========================================================

columns = [
    "우선순위",
    "요일",
    "출발구",
    "승차시간대",
    "평균완료운행건수",
    "평균대기시간_분",
    "대기90분초과비율_%",
    "평균접수배차_분",
    "UD우선순위점수"
]

print("\n========== UD택시 투입 우선순위 TOP 30 ==========")

print(
    df[columns]
    .head(30)
    .to_string(index=False)
)


# =========================================================
# 8. 결과 저장
# =========================================================

df.to_csv(
    RESULT_DIR / "ud_priority_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n저장 완료:")
print(RESULT_DIR / "ud_priority_2025.csv")

# =========================================================
# 대량 수요 대응형 UD택시 우선순위
# =========================================================

mass = pd.read_csv(
    RESULT_DIR / "ud_priority_2025.csv",
    encoding="utf-8-sig"
)

# 표본 30건 이상
mass = mass[
    mass["일반콜건수"] >= 30
].copy()


def minmax(series):
    min_v = series.min()
    max_v = series.max()

    if max_v == min_v:
        return pd.Series(0, index=series.index)

    return (series - min_v) / (max_v - min_v)


# 정규화
mass["수요점수"] = minmax(mass["평균완료운행건수"])
mass["대기점수"] = minmax(mass["평균대기시간_분"])
mass["장기대기점수"] = minmax(mass["대기90분초과비율_%"])


# 대량 수요 대응 점수
mass["대량수요대응점수"] = (
    mass["수요점수"] * 0.60
    + mass["대기점수"] * 0.25
    + mass["장기대기점수"] * 0.15
) * 100


# 순위
mass = (
    mass
    .sort_values("대량수요대응점수", ascending=False)
    .reset_index(drop=True)
)

mass["대량수요순위"] = range(1, len(mass) + 1)


# =========================================================
# 결과 출력
# =========================================================

cols = [
    "대량수요순위",
    "요일",
    "출발구",
    "승차시간대",
    "평균완료운행건수",
    "일반콜건수",
    "평균대기시간_분",
    "대기90분초과비율_%",
    "대량수요대응점수"
]

print("\n========== 대량 수요 대응형 UD택시 TOP 30 ==========")

print(
    mass[cols]
    .head(30)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 저장
# =========================================================

mass.to_csv(
    RESULT_DIR / "ud_mass_demand_priority_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n대량 수요 대응형 저장 완료:",
    RESULT_DIR / "ud_mass_demand_priority_2025.csv"
)

# =========================================================
# 최종 UD택시 동적배치 4분면 분석
# =========================================================

# 표본 수가 충분한 조합만 사용
dynamic = pd.read_csv(
    RESULT_DIR / "ud_priority_2025.csv",
    encoding="utf-8-sig"
)

dynamic = dynamic[
    dynamic["일반콜건수"] >= 30
].copy()


# =========================================================
# 1. 고수요 / 고대기 기준 설정
# =========================================================

# 분석 대상의 중앙값 사용
demand_cut = dynamic["평균완료운행건수"].median()
wait_cut = dynamic["평균대기시간_분"].median()

print("\n========== 4분면 분류 기준 ==========")
print(f"고수요 기준: {demand_cut:.2f}건 이상")
print(f"고대기 기준: {wait_cut:.2f}분 이상")


# =========================================================
# 2. 4분면 분류
# =========================================================

def classify(row):

    high_demand = row["평균완료운행건수"] >= demand_cut
    high_wait = row["평균대기시간_분"] >= wait_cut

    if high_demand and high_wait:
        return "A_최우선개입"

    elif not high_demand and high_wait:
        return "B_서비스취약"

    elif high_demand and not high_wait:
        return "C_대량수요대응"

    else:
        return "D_낮은우선순위"


dynamic["배치유형"] = dynamic.apply(
    classify,
    axis=1
)


# =========================================================
# 3. 유형별 개수
# =========================================================

type_count = (
    dynamic["배치유형"]
    .value_counts()
    .sort_index()
)

print("\n========== 동적배치 유형별 조합 수 ==========")
print(type_count)


# =========================================================
# 4. A유형 : 고수요 + 고대기
# =========================================================

type_a = dynamic[
    dynamic["배치유형"] == "A_최우선개입"
].copy()

# A유형 안에서는 수요 × 대기시간으로 우선순위 결정
type_a["최우선점수"] = (
    type_a["평균완료운행건수"]
    * type_a["평균대기시간_분"]
)

type_a = (
    type_a
    .sort_values(
        "최우선점수",
        ascending=False
    )
    .reset_index(drop=True)
)

type_a["A유형순위"] = range(
    1,
    len(type_a) + 1
)


# =========================================================
# 5. A유형 TOP 30
# =========================================================

cols = [
    "A유형순위",
    "요일",
    "출발구",
    "승차시간대",
    "평균완료운행건수",
    "일반콜건수",
    "평균대기시간_분",
    "대기90분초과비율_%",
    "평균접수배차_분",
    "최우선점수"
]

print(
    "\n========== A유형: 고수요 + 고대기 TOP 30 =========="
)

print(
    type_a[cols]
    .head(30)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 6. 유형별 평균 특성
# =========================================================

type_summary = (
    dynamic
    .groupby("배치유형")
    .agg(
        조합수=("배치유형", "size"),
        평균수요=("평균완료운행건수", "mean"),
        평균대기시간=("평균대기시간_분", "mean"),
        평균90분초과비율=("대기90분초과비율_%", "mean"),
        평균접수배차시간=("평균접수배차_분", "mean")
    )
    .round(2)
    .reset_index()
)

print(
    "\n========== 유형별 특성 =========="
)

print(
    type_summary.to_string(index=False)
)


# =========================================================
# 7. 최종 정책명 부여
# =========================================================

policy_map = {
    "A_최우선개입": "집중투입",
    "B_서비스취약": "최소공급보장",
    "C_대량수요대응": "수요대응배치",
    "D_낮은우선순위": "기본운영"
}

dynamic["UD운영전략"] = (
    dynamic["배치유형"]
    .map(policy_map)
)


# =========================================================
# 8. 결과 저장
# =========================================================

dynamic.to_csv(
    RESULT_DIR / "ud_dynamic_policy_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

type_a.to_csv(
    RESULT_DIR / "ud_priority_type_A_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

type_summary.to_csv(
    RESULT_DIR / "ud_policy_type_summary_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 최종 동적배치 분석 저장 완료 ==========")
print(RESULT_DIR / "ud_dynamic_policy_2025.csv")