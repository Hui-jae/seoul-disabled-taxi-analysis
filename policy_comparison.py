from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# 1. 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

ALLOCATION_PATH = RESULT_DIR / "ud_allocation_2025.csv"
OUTPUT_PATH = RESULT_DIR / "policy_comparison_2025.csv"

TOTAL_UD = 12
CURRENT_DAILY_LIMIT = 100


# =========================================================
# 2. 데이터 로드
# =========================================================

df = pd.read_csv(
    ALLOCATION_PATH,
    encoding="utf-8-sig"
)

print("========== 데이터 로드 ==========")
print(f"분석 조합: {len(df):,}개")


# =========================================================
# 3. UD 1대당 시간당 처리능력
# =========================================================
# 평균 차량점유시간 =
# 배차 후 승차까지 시간 + 실제 운행시간
#
# 60 / 평균점유시간 = 차량 1대가 1시간 동안
# 이론적으로 처리 가능한 운행건수

df["UD1대_시간당처리능력"] = np.where(
    df["평균차량점유시간_분"] > 0,
    60 / df["평균차량점유시간_분"],
    0
)

# 실제 수요 이상을 처리했다고 계산하지 않음
df["동적배치_처리가능건수"] = np.minimum(
    df["평균완료운행건수"],
    df["UD배치대수"] * df["UD1대_시간당처리능력"]
)

df["동적배치_기존차량부담건수"] = (
    df["평균완료운행건수"]
    - df["동적배치_처리가능건수"]
).clip(lower=0)

df["기존차량부담감소율_%"] = np.where(
    df["평균완료운행건수"] > 0,
    (
        df["동적배치_처리가능건수"]
        / df["평균완료운행건수"]
        * 100
    ),
    0
)


# =========================================================
# 4. UD 활용률 계산
# =========================================================
# 해당 지역에 배치된 UD가 가진 이론적 처리능력 중
# 실제 수요로 얼마나 활용될 수 있는지를 계산

df["UD총처리용량"] = (
    df["UD배치대수"]
    * df["UD1대_시간당처리능력"]
)

df["UD활용률_%"] = np.where(
    df["UD총처리용량"] > 0,
    (
        df["동적배치_처리가능건수"]
        / df["UD총처리용량"]
        * 100
    ),
    0
)

df["UD유휴처리용량"] = (
    df["UD총처리용량"]
    - df["동적배치_처리가능건수"]
).clip(lower=0)


# =========================================================
# 5. 동적배치 시간대별 집계
# =========================================================

dynamic = (
    df.groupby(
        ["요일", "승차시간대"],
        as_index=False
    )
    .agg(
        총수요=("평균완료운행건수", "sum"),
        UD배치대수=("UD배치대수", "sum"),
        UD처리가능건수=("동적배치_처리가능건수", "sum"),
        기존차량부담건수=("동적배치_기존차량부담건수", "sum"),
        UD총처리용량=("UD총처리용량", "sum"),
        UD유휴처리용량=("UD유휴처리용량", "sum")
    )
)

dynamic["UD활용률_%"] = np.where(
    dynamic["UD총처리용량"] > 0,
    dynamic["UD처리가능건수"]
    / dynamic["UD총처리용량"]
    * 100,
    0
)

dynamic["기존차량부담감소율_%"] = np.where(
    dynamic["총수요"] > 0,
    dynamic["UD처리가능건수"]
    / dynamic["총수요"]
    * 100,
    0
)


# =========================================================
# 6. 요일별 동적배치 성과
# =========================================================

daily_dynamic = (
    dynamic.groupby("요일", as_index=False)
    .agg(
        총수요=("총수요", "sum"),
        동적배치_UD처리건수=("UD처리가능건수", "sum"),
        기존차량부담건수=("기존차량부담건수", "sum"),
        UD총처리용량=("UD총처리용량", "sum"),
        UD유휴처리용량=("UD유휴처리용량", "sum")
    )
)

daily_dynamic["동적배치_UD활용률_%"] = np.where(
    daily_dynamic["UD총처리용량"] > 0,
    daily_dynamic["동적배치_UD처리건수"]
    / daily_dynamic["UD총처리용량"]
    * 100,
    0
)

daily_dynamic["동적배치_기존부담감소율_%"] = np.where(
    daily_dynamic["총수요"] > 0,
    daily_dynamic["동적배치_UD처리건수"]
    / daily_dynamic["총수요"]
    * 100,
    0
)


# =========================================================
# 7. 현행 100건 제한 정책과 비교
# =========================================================
# 주의:
# 실제 UD택시 운행자료가 없기 때문에
# 현행 정책의 '실제 처리건수'가 아니라
# 100건이라는 일일 상한과 비교함

daily_dynamic["현행정책_최대처리건수"] = np.minimum(
    daily_dynamic["총수요"],
    CURRENT_DAILY_LIMIT
)

daily_dynamic["동적배치_100건초과분"] = (
    daily_dynamic["동적배치_UD처리건수"]
    - CURRENT_DAILY_LIMIT
)

daily_dynamic["100건대비_처리능력비율_%"] = (
    daily_dynamic["동적배치_UD처리건수"]
    / CURRENT_DAILY_LIMIT
    * 100
)


# =========================================================
# 8. 서비스 취약지역 커버
# =========================================================

vulnerable = df[
    df["배치유형"].isin(
        ["A_최우선개입", "B_서비스취약"]
    )
].copy()

coverage = (
    vulnerable.groupby("요일")
    .agg(
        취약조합수=("배치유형", "size"),
        UD배치취약조합수=(
            "UD배치대수",
            lambda x: (x > 0).sum()
        )
    )
    .reset_index()
)

coverage["취약지역커버율_%"] = (
    coverage["UD배치취약조합수"]
    / coverage["취약조합수"]
    * 100
)

daily_dynamic = daily_dynamic.merge(
    coverage,
    on="요일",
    how="left"
)


# =========================================================
# 9. 요일 정렬
# =========================================================

day_order = {
    "월": 0,
    "화": 1,
    "수": 2,
    "목": 3,
    "금": 4,
    "토": 5,
    "일": 6
}

daily_dynamic["요일순서"] = (
    daily_dynamic["요일"].map(day_order)
)

daily_dynamic = (
    daily_dynamic
    .sort_values("요일순서")
    .drop(columns="요일순서")
)


# =========================================================
# 10. 결과 출력
# =========================================================

print("\n========== 현행 100건 제한 vs 동적배치 ==========")

cols = [
    "요일",
    "총수요",
    "현행정책_최대처리건수",
    "동적배치_UD처리건수",
    "동적배치_100건초과분",
    "100건대비_처리능력비율_%",
    "동적배치_UD활용률_%",
    "동적배치_기존부담감소율_%",
    "취약지역커버율_%"
]

print(
    daily_dynamic[cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 11. 전체 평균 성과
# =========================================================

print("\n========== 정책 비교 요약 ==========")

avg_dynamic = daily_dynamic[
    "동적배치_UD처리건수"
].mean()

avg_util = daily_dynamic[
    "동적배치_UD활용률_%"
].mean()

avg_reduction = daily_dynamic[
    "동적배치_기존부담감소율_%"
].mean()

avg_coverage = daily_dynamic[
    "취약지역커버율_%"
].mean()

print(
    f"현행 일일 운행 상한: "
    f"{CURRENT_DAILY_LIMIT}건"
)

print(
    f"동적배치 평균 처리 가능량: "
    f"{avg_dynamic:.2f}건/일"
)

print(
    f"100건 대비: "
    f"{avg_dynamic / CURRENT_DAILY_LIMIT * 100:.2f}%"
)

print(
    f"평균 UD 활용률: "
    f"{avg_util:.2f}%"
)

print(
    f"기존 장애인콜택시 평균 부담 감소율: "
    f"{avg_reduction:.2f}%"
)

print(
    f"A/B 서비스 취약조합 평균 커버율: "
    f"{avg_coverage:.2f}%"
)


# =========================================================
# 12. 시간대별 배치 확인
# =========================================================

print("\n========== 동적배치 시간대별 TOP 30 ==========")

time_cols = [
    "요일",
    "승차시간대",
    "총수요",
    "UD배치대수",
    "UD처리가능건수",
    "UD활용률_%",
    "기존차량부담감소율_%"
]

print(
    dynamic
    .sort_values(
        "UD처리가능건수",
        ascending=False
    )
    .head(30)[time_cols]
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 13. 저장
# =========================================================

daily_dynamic.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

dynamic.to_csv(
    RESULT_DIR / "policy_comparison_hourly_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print(OUTPUT_PATH)