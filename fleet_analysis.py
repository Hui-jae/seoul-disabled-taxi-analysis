from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 경로 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "processed" / "trip_demand_2025.csv"
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

KEY = ["요일", "출발구", "승차시간대"]


# =========================================================
# 2. 원본 데이터 로드 + 운행시간 계산
# =========================================================

df = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=["승차일자", "승차일시", "하차일시", "출발구"]
)

for col in ["승차일자", "승차일시", "하차일시"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df = df[df["승차일자"].dt.year == 2025].copy()

df["운행시간_분"] = (
    df["하차일시"] - df["승차일시"]
).dt.total_seconds() / 60

valid = df[df["운행시간_분"].between(0, 180, inclusive="right")].copy()

day_map = {
    0: "월", 1: "화", 2: "수", 3: "목",
    4: "금", 5: "토", 6: "일"
}

valid["요일"] = valid["승차일시"].dt.dayofweek.map(day_map)
valid["승차시간대"] = valid["승차일시"].dt.hour

print("========== 운행시간 품질 ==========")
print(f"전체: {len(df):,}건")
print(f"분석 대상: {len(valid):,}건")
print(f"제외: {len(df) - len(valid):,}건")


# =========================================================
# 3. 요일 × 구 × 시간대별 운행시간
# =========================================================

fleet_time = (
    valid.groupby(KEY, as_index=False)
    .agg(
        운행건수=("운행시간_분", "size"),
        평균운행시간_분=("운행시간_분", "mean"),
        중앙값운행시간_분=("운행시간_분", "median"),
        총운행시간_분=("운행시간_분", "sum")
    )
)

time_cols = [
    "평균운행시간_분",
    "중앙값운행시간_분",
    "총운행시간_분"
]

fleet_time[time_cols] = fleet_time[time_cols].round(2)

fleet_time.to_csv(
    RESULT_DIR / "fleet_time_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"\n운행시간 분석 조합: {len(fleet_time):,}개")


# =========================================================
# 4. 수요 + 운행시간 + 배차시간 결합
# =========================================================

demand = pd.read_csv(
    RESULT_DIR / "weekday_hour_gu_2025.csv",
    encoding="utf-8-sig"
)

wait = pd.read_csv(
    RESULT_DIR / "weekday_hour_gu_wait_2025.csv",
    encoding="utf-8-sig"
)

fleet = (
    demand[KEY + ["평균완료운행건수"]]
    .merge(
        fleet_time[KEY + ["평균운행시간_분"]],
        on=KEY,
        how="left"
    )
    .merge(
        wait[KEY + ["일반콜건수", "평균배차승차_분"]],
        on=KEY,
        how="left"
    )
    .dropna(
        subset=[
            "평균완료운행건수",
            "평균운행시간_분",
            "평균배차승차_분"
        ]
    )
    .copy()
)


# =========================================================
# 5. 기본 필요 차량대수
# =========================================================

fleet["평균차량점유시간_분"] = (
    fleet["평균운행시간_분"]
    + fleet["평균배차승차_분"]
)

fleet["기본필요차량대수"] = (
    fleet["평균완료운행건수"]
    * fleet["평균차량점유시간_분"]
    / 60
)

fleet["기본필요차량대수_올림"] = np.ceil(
    fleet["기본필요차량대수"]
).astype(int)

fleet.to_csv(
    RESULT_DIR / "fleet_requirement_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 기본 필요 차량대수 TOP 30 ==========")

fleet_cols = KEY + [
    "평균완료운행건수",
    "평균운행시간_분",
    "평균배차승차_분",
    "평균차량점유시간_분",
    "기본필요차량대수",
    "기본필요차량대수_올림"
]

print(
    fleet.sort_values("기본필요차량대수", ascending=False)
    .head(30)[fleet_cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 6. 동적배치 정책 데이터 결합
# =========================================================

policy = pd.read_csv(
    RESULT_DIR / "ud_dynamic_policy_2025.csv",
    encoding="utf-8-sig"
)

# 평균대기시간_분 포함 ← 기존 오류 수정
policy_cols = KEY + [
    "배치유형",
    "평균대기시간_분",
    "대기90분초과비율_%"
]

fleet_policy = fleet.merge(
    policy[policy_cols],
    on=KEY,
    how="inner"
)

print("\n========== 차량수요 + 정책유형 결합 ==========")
print(f"분석 조합: {len(fleet_policy):,}개")


# =========================================================
# 7. 차량 부족 압력 계산
# =========================================================

def normalize(series):
    min_v, max_v = series.min(), series.max()

    if min_v == max_v:
        return pd.Series(0.0, index=series.index)

    return (series - min_v) / (max_v - min_v)


fleet_policy["차량수요점수"] = normalize(
    fleet_policy["기본필요차량대수"]
)

fleet_policy["장기대기점수"] = normalize(
    fleet_policy["대기90분초과비율_%"]
)

fleet_policy["차량부족압력점수"] = (
    0.5 * fleet_policy["차량수요점수"]
    + 0.5 * fleet_policy["장기대기점수"]
) * 100

fleet_policy = (
    fleet_policy
    .sort_values("차량부족압력점수", ascending=False)
    .reset_index(drop=True)
)

fleet_policy["차량부족순위"] = fleet_policy.index + 1


# =========================================================
# 8. 결과 출력
# =========================================================

print("\n========== 차량 부족 압력 TOP 30 ==========")

result_cols = [
    "차량부족순위",
    "요일",
    "출발구",
    "승차시간대",
    "배치유형",
    "평균완료운행건수",
    "평균대기시간_분",
    "기본필요차량대수",
    "대기90분초과비율_%",
    "차량부족압력점수"
]

print(
    fleet_policy[result_cols]
    .head(30)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 9. 배치유형별 요약
# =========================================================

type_summary = (
    fleet_policy.groupby("배치유형", as_index=False)
    .agg(
        조합수=("배치유형", "size"),
        평균필요차량=("기본필요차량대수", "mean"),
        평균대기시간=("평균대기시간_분", "mean"),
        평균90분초과비율=("대기90분초과비율_%", "mean"),
        평균차량부족압력=("차량부족압력점수", "mean")
    )
    .round(2)
)

print("\n========== 배치유형별 차량 부족 압력 ==========")
print(type_summary.to_string(index=False))


# =========================================================
# 10. 최종 저장
# =========================================================

OUTPUT_PATH = RESULT_DIR / "fleet_dynamic_pressure_2025.csv"

fleet_policy.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 분석 완료 ==========")
print(f"저장 위치: {OUTPUT_PATH}")