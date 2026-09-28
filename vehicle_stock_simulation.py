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

for col in ["승차일시", "하차일시"]:
    trip[col] = pd.to_datetime(
        trip[col],
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

trip["운행시간_분"] = (
    trip["하차일시"] - trip["승차일시"]
).dt.total_seconds() / 60

trip = trip[
    (trip["운행시간_분"] > 0)
    & (trip["운행시간_분"] <= 180)
].copy()

trip["요일"] = (
    trip["승차일시"]
    .dt.dayofweek
    .map(DAY_MAP)
)

trip["승차시간대"] = trip["승차일시"].dt.hour

print("========== 운행 데이터 ==========")
print(f"분석 운행: {len(trip):,}건")


# =========================================================
# 3. 서울 25개 구 목록 생성
# =========================================================

gu_list = sorted(
    set(trip["출발구"].unique())
    | set(trip[DEST_COL].unique())
)

print("\n========== 지역 ==========")
print(f"구 개수: {len(gu_list)}개")
print(gu_list)

if len(gu_list) != 25:
    print(
        f"[경고] 서울 25개 구가 아닌 "
        f"{len(gu_list)}개 구가 확인되었습니다."
    )


# =========================================================
# 4. OD 확률 생성
# =========================================================
# P(목적구 | 요일, 시간대, 출발구)
# =========================================================

od = (
    trip.groupby(
        [
            "요일",
            "승차시간대",
            "출발구",
            DEST_COL
        ],
        as_index=False
    )
    .size()
    .rename(columns={"size": "OD건수"})
)

od["출발총건수"] = (
    od.groupby(
        [
            "요일",
            "승차시간대",
            "출발구"
        ]
    )["OD건수"]
    .transform("sum")
)

od["OD확률"] = (
    od["OD건수"] / od["출발총건수"]
)

print(f"\nOD 조합: {len(od):,}개")


# =========================================================
# 5. 목표 UD 배치 로드
# =========================================================

allocation = pd.read_csv(
    ALLOCATION_PATH,
    encoding="utf-8-sig"
)

# 기존처럼 UD배치대수 > 0만 남기지 않는다.
# 0대인 지역도 완전 패널에 포함한다.

allocation["요일순서"] = (
    allocation["요일"].map(DAY_ORDER)
)

allocation = allocation.sort_values(
    ["요일순서", "승차시간대", "출발구"]
)

print("\n========== 목표 UD 배치 ==========")
print(f"원본 배치 행: {len(allocation):,}개")


# =========================================================
# 6. 7일 × 24시간 × 25개 구 완전 패널 생성
# =========================================================

full_index = pd.MultiIndex.from_product(
    [
        list(DAY_ORDER.keys()),
        range(24),
        gu_list
    ],
    names=[
        "요일",
        "승차시간대",
        "출발구"
    ]
)

full_allocation = (
    allocation[
        [
            "요일",
            "승차시간대",
            "출발구",
            "UD배치대수"
        ]
    ]
    .set_index(
        [
            "요일",
            "승차시간대",
            "출발구"
        ]
    )
    .reindex(
        full_index,
        fill_value=0
    )
    .reset_index()
)

print(
    f"완전 패널 행: "
    f"{len(full_allocation):,}개"
)

expected_rows = (
    len(DAY_ORDER)
    * 24
    * len(gu_list)
)

print(
    f"예상 행 수: "
    f"{expected_rows:,}개"
)

if len(full_allocation) != expected_rows:
    raise ValueError(
        "완전 패널 생성에 실패했습니다."
    )


# =========================================================
# 7. 시간대별 목표 배치 사전
# =========================================================

targets = {}

for (day, hour), group in full_allocation.groupby(
    ["요일", "승차시간대"]
):

    targets[(day, hour)] = dict(
        zip(
            group["출발구"],
            group["UD배치대수"].astype(float)
        )
    )


# =========================================================
# 8. OD 확률 사전
# =========================================================

od_dict = {}

for (day, hour, origin), group in od.groupby(
    ["요일", "승차시간대", "출발구"]
):

    od_dict[(day, hour, origin)] = dict(
        zip(
            group[DEST_COL],
            group["OD확률"]
        )
    )


# =========================================================
# 9. 초기 차량 위치
# =========================================================
# 실제 시작 위치 자료가 없으므로
# 월요일 0시 목표배치를 초기 위치로 사용
#
# 목표배치 합이 12대가 아닐 경우
# 비율을 유지하면서 총량을 12대로 보정
# =========================================================

initial_target = targets[("월", 0)]

initial_total = sum(initial_target.values())

if initial_total > 0:

    stock = {
        gu: (
            initial_target.get(gu, 0)
            * TOTAL_UD
            / initial_total
        )
        for gu in gu_list
    }

else:

    # 월요일 0시 목표가 모두 0일 경우
    # 전체 배치자료의 구별 비중으로 초기화

    overall = (
        full_allocation
        .groupby("출발구")["UD배치대수"]
        .sum()
    )

    overall_total = overall.sum()

    if overall_total > 0:

        stock = {
            gu: (
                overall.get(gu, 0)
                * TOTAL_UD
                / overall_total
            )
            for gu in gu_list
        }

    else:

        # 최후 fallback
        # 모든 구에 균등 배치

        stock = {
            gu: TOTAL_UD / len(gu_list)
            for gu in gu_list
        }


# =========================================================
# 10. 차량 재고 시뮬레이션
# =========================================================

detail_records = []
hourly_records = []


for day in DAY_ORDER.keys():

    # 반드시 0~23시 전부 순회
    for hour in range(24):

        target = targets[(day, hour)]

        # ---------------------------------------------
        # 현재 시간 시작 차량 위치
        # ---------------------------------------------

        before = {
            gu: float(stock.get(gu, 0))
            for gu in gu_list
        }

        before_total = sum(before.values())

        # 차량 총량 검증
        if not np.isclose(
            before_total,
            TOTAL_UD,
            atol=1e-6
        ):
            raise ValueError(
                f"{day}요일 {hour}시 "
                f"시작 차량 총량 오류: "
                f"{before_total}"
            )

        # ---------------------------------------------
        # 자연 충족 / 부족 / 잉여
        # ---------------------------------------------

        natural = {
            gu: min(
                before[gu],
                target.get(gu, 0)
            )
            for gu in gu_list
        }

        shortage = {
            gu: max(
                target.get(gu, 0)
                - before[gu],
                0
            )
            for gu in gu_list
        }

        surplus = {
            gu: max(
                before[gu]
                - target.get(gu, 0),
                0
            )
            for gu in gu_list
        }

        natural_total = sum(natural.values())

        total_shortage = sum(
            shortage.values()
        )

        total_surplus = sum(
            surplus.values()
        )

        # ---------------------------------------------
        # 사후 재배치 가능량
        # ---------------------------------------------

        rebalance = min(
            total_shortage,
            total_surplus
        )

        target_total = sum(
            target.values()
        )

        satisfied = min(
            natural_total + rebalance,
            target_total
        )

        remaining_shortage = max(
            target_total - satisfied,
            0
        )

        # ---------------------------------------------
        # 재배치 후 실제 차량 위치 생성
        # ---------------------------------------------
        #
        # 먼저 기존 위치에서 목표량만큼 확보
        # 부족지역은 잉여지역 차량을 이동시켜 채움
        # ---------------------------------------------

        positioned = {
            gu: natural[gu]
            for gu in gu_list
        }

        shortage_left = shortage.copy()
        surplus_left = surplus.copy()

        for dest in gu_list:

            need = shortage_left[dest]

            if need <= 0:
                continue

            for origin in gu_list:

                available = surplus_left[origin]

                if available <= 0:
                    continue

                moved = min(
                    need,
                    available
                )

                positioned[dest] += moved
                surplus_left[origin] -= moved

                need -= moved

                if need <= 1e-12:
                    break

            shortage_left[dest] = need

        # ---------------------------------------------
        # 운행하지 않고 남은 차량
        # ---------------------------------------------

        idle = {
            gu: max(
                surplus_left[gu],
                0
            )
            for gu in gu_list
        }

        # ---------------------------------------------
        # 실제 운행 차량
        # ---------------------------------------------
        #
        # 목표량보다 많은 차량은 운행시키지 않으며,
        # 실제 배치 가능한 차량만 운행
        # ---------------------------------------------

        dispatched = {
            gu: min(
                positioned[gu],
                target.get(gu, 0)
            )
            for gu in gu_list
        }

        # ---------------------------------------------
        # 운행 후 차량 위치
        # ---------------------------------------------

        next_stock = {
            gu: 0.0
            for gu in gu_list
        }

        # 운행 차량은 OD 확률에 따라 목적지 이동
        for origin in gu_list:

            vehicle_count = dispatched[origin]

            if vehicle_count <= 0:
                continue

            probs = od_dict.get(
                (day, hour, origin)
            )

            # OD 정보가 없으면 출발구에 유지
            if not probs:

                next_stock[origin] += (
                    vehicle_count
                )

                continue

            for dest, prob in probs.items():

                if dest not in next_stock:
                    continue

                next_stock[dest] += (
                    vehicle_count * prob
                )

        # 미운행 차량은 현재 위치에 유지
        for gu in gu_list:

            next_stock[gu] += idle[gu]

        # ---------------------------------------------
        # 수치 오차 검증
        # ---------------------------------------------

        stock_total = sum(
            next_stock.values()
        )

        if not np.isclose(
            stock_total,
            TOTAL_UD,
            atol=1e-6
        ):

            # 부동소수점 수준의 오차만 보정
            diff = TOTAL_UD - stock_total

            max_gu = max(
                next_stock,
                key=next_stock.get
            )

            next_stock[max_gu] += diff

        # 다음 시간 시작 위치
        stock = next_stock

        # ---------------------------------------------
        # 상세 결과
        # 반드시 25개 구 모두 기록
        # ---------------------------------------------

        for gu in gu_list:

            detail_records.append({

                "요일": day,

                "승차시간대": hour,

                "출발구": gu,

                "목표UD":
                    target.get(gu, 0),

                "시간대시작UD":
                    before[gu],

                "자연충족UD":
                    natural[gu],

                "부족UD":
                    shortage[gu],

                "잉여UD":
                    surplus[gu],

                "실제운행UD":
                    dispatched[gu],

                "시간대종료UD":
                    next_stock[gu]
            })

        # ---------------------------------------------
        # 시간대 요약
        # ---------------------------------------------

        hourly_records.append({

            "요일": day,

            "승차시간대": hour,

            "목표UD":
                target_total,

            "시간대시작UD":
                before_total,

            "자연충족UD":
                natural_total,

            "재배치UD":
                rebalance,

            "재배치후충족UD":
                satisfied,

            "미충족UD":
                remaining_shortage,

            "실제운행UD":
                sum(dispatched.values()),

            "시간대종료UD":
                sum(next_stock.values())
        })


# =========================================================
# 11. 결과 DataFrame
# =========================================================

detail = pd.DataFrame(
    detail_records
)

hourly = pd.DataFrame(
    hourly_records
)


# =========================================================
# 12. 지표 계산
# =========================================================

hourly["자연위치일치율_%"] = np.where(
    hourly["목표UD"] > 0,
    hourly["자연충족UD"]
    / hourly["목표UD"]
    * 100,
    0
)

hourly["재배치후충족률_%"] = np.where(
    hourly["목표UD"] > 0,
    hourly["재배치후충족UD"]
    / hourly["목표UD"]
    * 100,
    0
)

hourly["재배치비율_%"] = np.where(
    hourly["목표UD"] > 0,
    hourly["재배치UD"]
    / hourly["목표UD"]
    * 100,
    0
)


# =========================================================
# 13. 핵심 검증
# =========================================================

print(
    "\n========== 핵심 검증 =========="
)

expected_detail_rows = (
    7 * 24 * len(gu_list)
)

print(
    f"상세 행 수: "
    f"{len(detail):,}"
)

print(
    f"예상 상세 행 수: "
    f"{expected_detail_rows:,}"
)

print(
    f"시간대 행 수: "
    f"{len(hourly):,}"
)

print(
    f"예상 시간대 행 수: "
    f"{7 * 24:,}"
)

print(
    "\n시간대 시작 차량 총량:"
)

print(
    hourly["시간대시작UD"]
    .describe()
    .round(6)
)

print(
    "\n시간대 종료 차량 총량:"
)

print(
    hourly["시간대종료UD"]
    .describe()
    .round(6)
)

bad_start = (
    ~np.isclose(
        hourly["시간대시작UD"],
        TOTAL_UD,
        atol=1e-6
    )
).sum()

bad_end = (
    ~np.isclose(
        hourly["시간대종료UD"],
        TOTAL_UD,
        atol=1e-6
    )
).sum()

print(
    f"\n시작 재고 12대 불일치: "
    f"{bad_start}개 시간대"
)

print(
    f"종료 재고 12대 불일치: "
    f"{bad_end}개 시간대"
)

if len(detail) != expected_detail_rows:
    raise ValueError(
        "vehicle_stock_detail 행 수가 "
        "완전 패널과 일치하지 않습니다."
    )

if len(hourly) != 168:
    raise ValueError(
        "시간대 결과가 168시간과 "
        "일치하지 않습니다."
    )

if bad_start > 0 or bad_end > 0:
    raise ValueError(
        "차량 총량 12대 보존에 "
        "실패했습니다."
    )


# =========================================================
# 14. 요일별 요약
# =========================================================

daily = (
    hourly.groupby(
        "요일",
        as_index=False
    )
    .agg(
        분석시간대=(
            "승차시간대",
            "size"
        ),

        평균목표UD=(
            "목표UD",
            "mean"
        ),

        평균자연충족UD=(
            "자연충족UD",
            "mean"
        ),

        평균재배치UD=(
            "재배치UD",
            "mean"
        ),

        총재배치UD=(
            "재배치UD",
            "sum"
        ),

        평균자연위치일치율=(
            "자연위치일치율_%",
            "mean"
        ),

        평균재배치후충족률=(
            "재배치후충족률_%",
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
    "\n========== 요일별 차량 재고 분석 =========="
)

print(
    daily
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 15. 전체 요약
# =========================================================

active = hourly[
    hourly["목표UD"] > 0
]

print(
    "\n========== 전체 차량 재고 시뮬레이션 =========="
)

print(
    f"평균 자연 위치 일치율: "
    f"{active['자연위치일치율_%'].mean():.2f}%"
)

print(
    f"평균 재배치 차량: "
    f"{active['재배치UD'].mean():.2f}대/시간"
)

print(
    f"평균 재배치 비율: "
    f"{active['재배치비율_%'].mean():.2f}%"
)

print(
    f"재배치 후 목표 충족률: "
    f"{active['재배치후충족률_%'].mean():.2f}%"
)

print(
    f"전체 재배치량: "
    f"{active['재배치UD'].sum():.2f}대·회"
)


# =========================================================
# 16. 저장
# =========================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

detail.to_csv(
    RESULT_DIR
    / "vehicle_stock_detail_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

hourly.to_csv(
    RESULT_DIR
    / "vehicle_stock_hourly_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

daily.to_csv(
    RESULT_DIR
    / "vehicle_stock_daily_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\n========== 저장 완료 =========="
)

print(
    RESULT_DIR
    / "vehicle_stock_detail_2025.csv"
)

print(
    RESULT_DIR
    / "vehicle_stock_hourly_2025.csv"
)

print(
    RESULT_DIR
    / "vehicle_stock_daily_2025.csv"
)