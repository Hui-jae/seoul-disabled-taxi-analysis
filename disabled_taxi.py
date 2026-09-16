from pathlib import Path
import pandas as pd


# =========================================================
# 1. 경로 및 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
FILE_PATH = BASE_DIR / "disabled_taxi_2025.csv"
OUTPUT_DIR = BASE_DIR / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)

SEOUL_GU = {
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구"
}

DATE_COLUMNS = [
    "접수일시",
    "예정일시",
    "배차일시",
    "승차일시",
    "하차일시"
]

DAY_MAP = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일"
}

print("CSV 위치:", FILE_PATH)
print("파일 존재 여부:", FILE_PATH.exists())


# =========================================================
# 2. 원본 데이터 전처리
# =========================================================

cleaned_chunks = []

for df in pd.read_csv(
    FILE_PATH,
    encoding="utf-8-sig",
    chunksize=100000
):

    # 자치구 이름 정리
    df["출발구"] = df["출발구"].astype("string").str.strip()
    df["목적구"] = df["목적구"].astype("string").str.strip()

    # 날짜형 변환
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    # 승차·하차 기록이 있는 운행만 사용
    df = df[
        df["승차일시"].notna()
        & df["하차일시"].notna()
    ].copy()

    # 서울 → 서울 운행만 사용
    df = df[
        df["출발구"].isin(SEOUL_GU)
        & df["목적구"].isin(SEOUL_GU)
    ].copy()

    # 하차시간이 승차시간보다 빠른 오류 제거
    df = df[
        df["하차일시"] >= df["승차일시"]
    ].copy()

    # -----------------------------------------------------
    # 시간 관련 파생변수
    # -----------------------------------------------------

    df["승차일자"] = df["승차일시"].dt.date
    df["승차시간대"] = df["승차일시"].dt.hour
    df["요일"] = df["승차일시"].dt.dayofweek.map(DAY_MAP)
    df["주말여부"] = df["승차일시"].dt.dayofweek >= 5
    df["월"] = df["승차일시"].dt.month

    # 실제 운행시간
    df["운행시간_분"] = (
        df["하차일시"] - df["승차일시"]
    ).dt.total_seconds() / 60

    # 접수 → 승차
    df["접수_승차대기시간_분"] = (
        df["승차일시"] - df["접수일시"]
    ).dt.total_seconds() / 60

    # 배차 → 승차
    df["배차_승차시간_분"] = (
        df["승차일시"] - df["배차일시"]
    ).dt.total_seconds() / 60

    # 접수 → 예정
    df["접수_예정차이_분"] = (
        df["예정일시"] - df["접수일시"]
    ).dt.total_seconds() / 60

    # 예정 → 실제 승차
    df["예정_승차차이_분"] = (
        df["승차일시"] - df["예정일시"]
    ).dt.total_seconds() / 60

    # -----------------------------------------------------
    # 거리 관련 변수
    # -----------------------------------------------------

    df["승차거리_km"] = (
        pd.to_numeric(
            df["승차거리"],
            errors="coerce"
        ) / 1000
    )

    df["평균속도_kmh"] = (
        df["승차거리_km"]
        / (df["운행시간_분"] / 60)
    )

    # -----------------------------------------------------
    # OD
    # -----------------------------------------------------

    df["OD_구"] = (
        df["출발구"]
        + "→"
        + df["목적구"]
    )

    df["OD_동"] = (
        df["출발동"].astype("string")
        + "→"
        + df["목적동"].astype("string")
    )

    # -----------------------------------------------------
    # 이용목적 기준 예약 여부
    # -----------------------------------------------------

    df["예약여부"] = (
        df["이용목적"]
        .astype("string")
        .str.contains("예약", na=False)
    )

    cleaned_chunks.append(df)


# =========================================================
# 3. 전체 데이터 결합
# =========================================================

clean_df = pd.concat(
    cleaned_chunks,
    ignore_index=True
)

print("\n전체 전처리 완료")
print("서울→서울 완료운행:", f"{len(clean_df):,}건")
print("출발구:", clean_df["출발구"].nunique(), "개")
print("목적구:", clean_df["목적구"].nunique(), "개")


# =========================================================
# 4. 분석 목적별 데이터 생성
# =========================================================

# A. 완료운행 / OD 분석용
# 거리나 운행시간에 문제가 있어도 실제 완료운행 기록이면 유지
trip_demand = clean_df.copy()


# B. 운행시간 분석용
# 품질검사 결과를 바탕으로 1~180분 사용
trip_time = clean_df[
    (clean_df["운행시간_분"] >= 1)
    & (clean_df["운행시간_분"] <= 180)
].copy()


# C. 거리 / 탄소 / 에너지 분석용
# 운행시간 정상 + 거리 > 0 + 평균속도 <= 100km/h
trip_distance = trip_time[
    (trip_time["승차거리_km"] > 0)
    & (trip_time["평균속도_kmh"] <= 100)
].copy()


# =========================================================
# 5. 저장
# =========================================================

trip_demand.to_csv(
    OUTPUT_DIR / "trip_demand_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

trip_time.to_csv(
    OUTPUT_DIR / "trip_time_2025.csv",
    index=False,
    encoding="utf-8-sig"
)

trip_distance.to_csv(
    OUTPUT_DIR / "trip_distance_2025.csv",
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 6. 결과 확인
# =========================================================

print("\n========== 저장 완료 ==========")

print(
    "완료운행/OD :",
    f"{len(trip_demand):,}건"
)

print(
    "운행시간 분석 :",
    f"{len(trip_time):,}건"
)

print(
    "거리/탄소 분석 :",
    f"{len(trip_distance):,}건"
)

print("\n[예약유형]")
print(
    clean_df["예약여부"].value_counts()
)

print("\n저장 위치:")
print(OUTPUT_DIR)