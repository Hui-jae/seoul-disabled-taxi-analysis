from pathlib import Path
import pandas as pd


# =========================================================
# 1. 데이터 로드
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

usecols = [
    "승차일자", "접수일시", "배차일시", "승차일시",
    "출발구", "출발동", "목적구", "이용목적", "예약여부"
]

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=usecols
)

datetime_cols = [
    "승차일자", "접수일시", "배차일시", "승차일시"
]

df[datetime_cols] = df[datetime_cols].apply(
    pd.to_datetime,
    errors="coerce"
)

df = df[df["승차일자"].dt.year == 2025].copy()

# 예약여부 문자열 → bool
if df["예약여부"].dtype == "object":
    df["예약여부"] = (
        df["예약여부"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

normal = df[df["예약여부"] == False].copy()
reserved = df[df["예약여부"] == True].copy()

print(f"전체 완료운행: {len(df):,}건")
print(f"일반콜: {len(normal):,}건")
print(f"예약콜: {len(reserved):,}건")


# =========================================================
# 2. 일반콜 대기시간 계산
# =========================================================

normal["접수_배차시간_분"] = (
    normal["배차일시"] - normal["접수일시"]
).dt.total_seconds() / 60

normal["배차_승차시간_분"] = (
    normal["승차일시"] - normal["배차일시"]
).dt.total_seconds() / 60

normal["접수_승차대기시간_분"] = (
    normal["승차일시"] - normal["접수일시"]
).dt.total_seconds() / 60

wait_cols = [
    "접수_배차시간_분",
    "배차_승차시간_분",
    "접수_승차대기시간_분"
]

# 시간 순서가 잘못된 데이터 제거
normal = normal[
    (normal[wait_cols] >= 0).all(axis=1)
].copy()

# 대기시간 180분 이하만 분석
analysis = normal[
    normal["접수_승차대기시간_분"] <= 180
].copy()

print(f"\n분석 대상: {len(analysis):,}건")
print(f"180분 초과 제외: {len(normal) - len(analysis):,}건")


# =========================================================
# 3. 시간 변수 + 장기대기 변수
# =========================================================

day_map = {
    0: "월", 1: "화", 2: "수", 3: "목",
    4: "금", 5: "토", 6: "일"
}

analysis["요일"] = (
    analysis["승차일시"].dt.dayofweek.map(day_map)
)

analysis["승차시간대"] = (
    analysis["승차일시"].dt.hour
)

for minute in [60, 90, 120]:
    analysis[f"{minute}분초과"] = (
        analysis["접수_승차대기시간_분"] > minute
    )


# =========================================================
# 4. 기본 대기시간 집계 함수
# =========================================================

def wait_summary(group):
    return (
        analysis
        .groupby(group)
        .agg(
            일반콜건수=("접수_승차대기시간_분", "size"),
            평균대기시간_분=("접수_승차대기시간_분", "mean"),
            중앙값대기시간_분=("접수_승차대기시간_분", "median"),
            평균접수배차_분=("접수_배차시간_분", "mean"),
            평균배차승차_분=("배차_승차시간_분", "mean")
        )
        .reset_index()
        .round(2)
    )


# =========================================================
# 5. 구 / 시간대 / 요일 분석
# =========================================================

gu_wait = wait_summary("출발구")
hour_wait = wait_summary("승차시간대")
weekday_wait = wait_summary("요일")

day_order = ["월", "화", "수", "목", "금", "토", "일"]

weekday_wait["요일"] = pd.Categorical(
    weekday_wait["요일"],
    categories=day_order,
    ordered=True
)

weekday_wait = weekday_wait.sort_values("요일")


# =========================================================
# 6. 요일 × 출발구 × 시간대 장기대기 분석
# =========================================================

wait_risk = (
    analysis
    .groupby(
        ["요일", "출발구", "승차시간대"],
        as_index=False
    )
    .agg(
        일반콜건수=("접수_승차대기시간_분", "size"),

        평균대기시간_분=(
            "접수_승차대기시간_분", "mean"
        ),

        중앙값대기시간_분=(
            "접수_승차대기시간_분", "median"
        ),

        평균접수배차_분=(
            "접수_배차시간_분", "mean"
        ),

        평균배차승차_분=(
            "배차_승차시간_분", "mean"
        ),

        대기60분초과건수=("60분초과", "sum"),
        대기90분초과건수=("90분초과", "sum"),
        대기120분초과건수=("120분초과", "sum")
    )
)

# 장기대기 비율 계산
for minute in [60, 90, 120]:

    wait_risk[f"대기{minute}분초과비율_%"] = (
        wait_risk[f"대기{minute}분초과건수"]
        / wait_risk["일반콜건수"]
        * 100
    ).round(2)

# 평균값 반올림
round_cols = [
    "평균대기시간_분",
    "중앙값대기시간_분",
    "평균접수배차_분",
    "평균배차승차_분"
]

wait_risk[round_cols] = (
    wait_risk[round_cols].round(2)
)


# =========================================================
# 7. 결과 출력
# =========================================================

print("\n========== 출발구별 대기시간 ==========")

print(
    gu_wait
    .sort_values(
        "평균대기시간_분",
        ascending=False
    )
    .to_string(index=False)
)


print("\n========== 시간대별 대기시간 ==========")

print(
    hour_wait.to_string(index=False)
)


print("\n========== 요일별 대기시간 ==========")

print(
    weekday_wait.to_string(index=False)
)


# 표본 30건 이상만 비교
valid_risk = wait_risk[
    wait_risk["일반콜건수"] >= 30
].copy()


print(
    "\n========== 평균 대기시간 상위 30개 =========="
)

print(
    valid_risk
    .sort_values(
        "평균대기시간_분",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


print(
    "\n========== 90분 초과 비율 상위 30개 =========="
)

print(
    valid_risk
    .sort_values(
        "대기90분초과비율_%",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


print(
    "\n========== 120분 초과 비율 상위 30개 =========="
)

print(
    valid_risk
    .sort_values(
        "대기120분초과비율_%",
        ascending=False
    )
    .head(30)
    .to_string(index=False)
)


# =========================================================
# 8. 결과 저장
# =========================================================

results = {
    "gu_wait_2025.csv": gu_wait,
    "hour_wait_2025.csv": hour_wait,
    "weekday_wait_2025.csv": weekday_wait,
    "weekday_hour_gu_wait_2025.csv": wait_risk,
    "wait_risk_2025.csv": wait_risk
}

for filename, data in results.items():

    data.to_csv(
        RESULT_DIR / filename,
        index=False,
        encoding="utf-8-sig"
    )

print("\n========== 저장 완료 ==========")
print("저장 위치:", RESULT_DIR)