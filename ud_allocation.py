from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 1. 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

INPUT_PATH = RESULT_DIR / "fleet_dynamic_pressure_2025.csv"
OUTPUT_PATH = RESULT_DIR / "ud_allocation_2025.csv"

TOTAL_UD = 12

KEY = ["요일", "승차시간대"]


# =========================================================
# 2. 데이터 로드
# =========================================================

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

print("========== 데이터 로드 ==========")
print(f"전체 조합: {len(df):,}개")


# =========================================================
# 3. 배치 우선순위 설정
# =========================================================

# 정책 유형별 우선순위
type_weight = {
    "A_최우선개입": 1.00,
    "B_서비스취약": 0.80,
    "C_대량수요대응": 0.70,
    "D_낮은우선순위": 0.10
}

df["유형가중치"] = (
    df["배치유형"]
    .map(type_weight)
    .fillna(0)
)

# 기존 차량부족압력점수에 정책 중요도 반영
df["최종배치점수"] = (
    df["차량부족압력점수"]
    * df["유형가중치"]
)


# =========================================================
# 4. 한 시간대의 UD 12대 배치 함수
# =========================================================

def allocate_ud(group):

    group = group.copy()

    # 실제 배치 대상
    target = group[
        group["배치유형"].isin([
            "A_최우선개입",
            "B_서비스취약",
            "C_대량수요대응"
        ])
    ].copy()

    group["UD배치대수"] = 0

    if len(target) == 0:
        group["미배치UD"] = TOTAL_UD
        return group

    # -----------------------------------------------------
    # 1. 지역별 최대 배치 가능 대수
    # -----------------------------------------------------

    target["배치상한"] = np.ceil(
        target["기본필요차량대수"]
    ).astype(int)

    target["배치상한"] = target["배치상한"].clip(lower=0)

    # -----------------------------------------------------
    # 2. 실제 배치 필요도 계산
    # -----------------------------------------------------

    # 정책 유형 가중치 × 차량 부족 압력
    target["배치필요도"] = (
        target["차량부족압력점수"]
        * target["유형가중치"]
    )

    # 필요도가 없는 경우 종료
    total_need = target["배치필요도"].sum()

    if total_need <= 0:
        group["미배치UD"] = TOTAL_UD
        return group

    # -----------------------------------------------------
    # 3. 12대를 필요도 비율로 배분
    # -----------------------------------------------------

    target["배치비율"] = (
        target["배치필요도"]
        / total_need
    )

    target["이론배치대수"] = (
        target["배치비율"]
        * TOTAL_UD
    )

    # 우선 정수 부분 배치
    target["UD배치대수"] = np.floor(
        target["이론배치대수"]
    ).astype(int)

    # 지역 필요차량보다 많이 배치하지 않음
    target["UD배치대수"] = np.minimum(
        target["UD배치대수"],
        target["배치상한"]
    )

    remaining = (
        TOTAL_UD
        - target["UD배치대수"].sum()
    )

    # -----------------------------------------------------
    # 4. 남은 차량 배분
    # Largest Remainder 방식
    # -----------------------------------------------------

    target["배치잔여값"] = (
        target["이론배치대수"]
        - np.floor(target["이론배치대수"])
    )

    while remaining > 0:

        candidates = target[
            target["UD배치대수"]
            < target["배치상한"]
        ].copy()

        if len(candidates) == 0:
            break

        candidates = candidates.sort_values(
            [
                "배치잔여값",
                "배치필요도"
            ],
            ascending=False
        )

        allocated = False

        for idx in candidates.index:

            if remaining <= 0:
                break

            target.loc[idx, "UD배치대수"] += 1
            remaining -= 1
            allocated = True

        if not allocated:
            break

    # -----------------------------------------------------
    # 5. 원본 데이터에 반영
    # -----------------------------------------------------

    group.loc[
        target.index,
        "UD배치대수"
    ] = target["UD배치대수"]

    group["미배치UD"] = remaining

    return group

# =========================================================
# 5. 요일 × 시간대별 12대 배치
# =========================================================

allocation = (
    df.groupby(KEY, group_keys=False)
    .apply(allocate_ud)
    .reset_index(drop=True)
)


# =========================================================
# 6. 배치 결과 검증
# =========================================================

hour_summary = (
    allocation.groupby(KEY, as_index=False)
    .agg(
        총UD배치대수=("UD배치대수", "sum"),
        배치지역수=("UD배치대수", lambda x: (x > 0).sum()),
        미배치UD=("미배치UD", "max")
    )
)

print("\n========== UD 12대 배치 검증 ==========")

print(
    hour_summary["총UD배치대수"]
    .describe()
    .round(2)
)

print(
    "\n12대 초과 배치:",
    (hour_summary["총UD배치대수"] > TOTAL_UD).sum(),
    "개 시간대"
)


# =========================================================
# 7. 실제 배치된 지역만 확인
# =========================================================

allocated = allocation[
    allocation["UD배치대수"] > 0
].copy()

day_order = {
    "월": 0, "화": 1, "수": 2, "목": 3,
    "금": 4, "토": 5, "일": 6
}

allocated["요일순서"] = allocated["요일"].map(day_order)

allocated = allocated.sort_values(
    ["요일순서", "승차시간대", "UD배치대수"],
    ascending=[True, True, False]
)

print("\n========== UD택시 동적배치 예시 ==========")

cols = [
    "요일",
    "승차시간대",
    "출발구",
    "배치유형",
    "평균완료운행건수",
    "평균대기시간_분",
    "기본필요차량대수",
    "차량부족압력점수",
    "최종배치점수",
    "UD배치대수"
]

print(
    allocated[cols]
    .head(50)
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 8. 시간대별 유형 배분 확인
# =========================================================

type_allocation = (
    allocated.groupby(
        ["요일", "승차시간대", "배치유형"],
        as_index=False
    )
    .agg(
        UD배치대수=("UD배치대수", "sum")
    )
)

print("\n========== 배치유형별 UD 배분 ==========")

print(
    allocated.groupby("배치유형")["UD배치대수"]
    .sum()
    .sort_values(ascending=False)
)


# =========================================================
# 9. 저장
# =========================================================

allocation.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

hour_summary.to_csv(
    RESULT_DIR / "ud_allocation_hour_summary_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

type_allocation.to_csv(
    RESULT_DIR / "ud_allocation_type_summary_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 저장 완료 ==========")
print(OUTPUT_PATH)