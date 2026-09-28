from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# =========================================================
# 1. 경로 / 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "processed"
    / "trip_demand_2025.csv"
)

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_START_DATE = pd.Timestamp("2025-11-01")

LAG_DAYS = [
    1,
    2,
    3,
    7,
    14,
    28
]

ROLLING_WINDOWS = [
    3,
    7,
    14,
    28
]

SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구"
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


# =========================================================
# 2. 2025년 공휴일 설정
# =========================================================
# 공휴일명은 결과 분석용
# 모델에는 공휴일 여부 / 전날 / 다음날 / 연휴 여부를 사용
# =========================================================

HOLIDAYS_2025 = {
    "2025-01-01": "신정",

    "2025-01-28": "설날연휴",
    "2025-01-29": "설날",
    "2025-01-30": "설날연휴",

    "2025-03-01": "삼일절",
    "2025-03-03": "삼일절대체공휴일",

    "2025-05-05": "어린이날·부처님오신날",
    "2025-05-06": "대체공휴일",

    "2025-06-06": "현충일",

    "2025-08-15": "광복절",

    "2025-10-03": "개천절",

    "2025-10-05": "추석연휴",
    "2025-10-06": "추석",
    "2025-10-07": "추석연휴",
    "2025-10-08": "추석대체공휴일",

    "2025-10-09": "한글날",

    "2025-12-25": "성탄절"
}

HOLIDAY_DATES = {
    pd.Timestamp(date)
    for date in HOLIDAYS_2025.keys()
}


# =========================================================
# 3. 원자료 로드
# =========================================================

print("========== 데이터 로드 ==========")

usecols = [
    "승차일시",
    "출발구"
]

trip = pd.read_csv(
    DATA_PATH,
    encoding="utf-8-sig",
    usecols=usecols
)

trip["승차일시"] = pd.to_datetime(
    trip["승차일시"],
    errors="coerce"
)

trip = (
    trip
    .dropna(
        subset=[
            "승차일시",
            "출발구"
        ]
    )
    .copy()
)

trip = trip[
    trip["승차일시"].dt.year == 2025
].copy()

trip = trip[
    trip["출발구"].isin(SEOUL_GU)
].copy()

trip["날짜"] = (
    trip["승차일시"]
    .dt
    .normalize()
)

trip["시간대"] = (
    trip["승차일시"]
    .dt
    .hour
)

print(
    f"분석 운행: {len(trip):,}건"
)


# =========================================================
# 4. 날짜 × 시간 × 구 수요 생성
# =========================================================

print(
    "\n========== 시간대별 수요 생성 =========="
)

demand = (
    trip
    .groupby(
        [
            "날짜",
            "시간대",
            "출발구"
        ],
        as_index=False
    )
    .size()
    .rename(
        columns={
            "size": "수요"
        }
    )
)

all_dates = pd.date_range(
    trip["날짜"].min(),
    trip["날짜"].max(),
    freq="D"
)

full_index = pd.MultiIndex.from_product(
    [
        all_dates,
        range(24),
        SEOUL_GU
    ],
    names=[
        "날짜",
        "시간대",
        "출발구"
    ]
)

demand = (
    demand
    .set_index(
        [
            "날짜",
            "시간대",
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
    f"수요 조합: {len(demand):,}개"
)

print(
    f"평균 시간당 구별 수요: "
    f"{demand['수요'].mean():.2f}건"
)


# =========================================================
# 5. 달력 변수 생성
# =========================================================

demand["요일번호"] = (
    demand["날짜"]
    .dt
    .dayofweek
)

demand["요일"] = (
    demand["요일번호"]
    .map(DAY_MAP)
)

demand["주말여부"] = (
    demand["요일번호"]
    .isin([5, 6])
    .astype(int)
)

demand["월"] = (
    demand["날짜"]
    .dt
    .month
)

demand["일"] = (
    demand["날짜"]
    .dt
    .day
)

demand["주차"] = (
    demand["날짜"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)


# =========================================================
# 6. 공휴일 / 연휴 Feature 생성
# =========================================================

print(
    "\n========== 공휴일 Feature 생성 =========="
)

demand["공휴일여부"] = (
    demand["날짜"]
    .isin(HOLIDAY_DATES)
    .astype(int)
)

demand["공휴일명"] = (
    demand["날짜"]
    .dt
    .strftime("%Y-%m-%d")
    .map(HOLIDAYS_2025)
    .fillna("비공휴일")
)

# 다음 날이 공휴일이면 현재 날짜는 공휴일 전날
demand["공휴일전날"] = (
    (demand["날짜"] + pd.Timedelta(days=1))
    .isin(HOLIDAY_DATES)
    .astype(int)
)

# 전날이 공휴일이면 현재 날짜는 공휴일 다음날
demand["공휴일다음날"] = (
    (demand["날짜"] - pd.Timedelta(days=1))
    .isin(HOLIDAY_DATES)
    .astype(int)
)

# 주말 또는 공휴일
demand["휴일여부"] = (
    (
        (demand["주말여부"] == 1)
        | (demand["공휴일여부"] == 1)
    )
    .astype(int)
)

# 연휴 여부:
# 현재가 공휴일이면서 전날 또는 다음날도
# 주말/공휴일인 경우를 연휴로 처리
date_info = (
    demand[
        [
            "날짜",
            "주말여부",
            "공휴일여부"
        ]
    ]
    .drop_duplicates("날짜")
    .sort_values("날짜")
    .reset_index(drop=True)
)

date_info["휴일여부"] = (
    (
        (date_info["주말여부"] == 1)
        | (date_info["공휴일여부"] == 1)
    )
    .astype(int)
)

date_info["전날휴일"] = (
    date_info["휴일여부"]
    .shift(1)
    .fillna(0)
    .astype(int)
)

date_info["다음날휴일"] = (
    date_info["휴일여부"]
    .shift(-1)
    .fillna(0)
    .astype(int)
)

date_info["연휴여부"] = (
    (
        (date_info["공휴일여부"] == 1)
        & (
            (date_info["전날휴일"] == 1)
            | (date_info["다음날휴일"] == 1)
        )
    )
    .astype(int)
)

demand = demand.merge(
    date_info[
        [
            "날짜",
            "연휴여부"
        ]
    ],
    on="날짜",
    how="left"
)

print(
    f"공휴일 조합: "
    f"{demand['공휴일여부'].sum():,}개"
)

print(
    f"연휴 조합: "
    f"{demand['연휴여부'].sum():,}개"
)


# =========================================================
# 7. 시간 주기성 변수
# =========================================================

demand["시간_sin"] = np.sin(
    2
    * np.pi
    * demand["시간대"]
    / 24
)

demand["시간_cos"] = np.cos(
    2
    * np.pi
    * demand["시간대"]
    / 24
)

demand["요일_sin"] = np.sin(
    2
    * np.pi
    * demand["요일번호"]
    / 7
)

demand["요일_cos"] = np.cos(
    2
    * np.pi
    * demand["요일번호"]
    / 7
)


# =========================================================
# 8. 출발구 범주형 변수
# =========================================================

demand["출발구"] = (
    demand["출발구"]
    .astype("category")
)


# =========================================================
# 9. Lag Feature 생성
# =========================================================

print(
    "\n========== Lag Feature 생성 =========="
)

demand = demand.sort_values(
    [
        "출발구",
        "시간대",
        "날짜"
    ]
).reset_index(drop=True)

grouped = demand.groupby(
    [
        "출발구",
        "시간대"
    ],
    observed=True
)["수요"]

for lag in LAG_DAYS:

    demand[
        f"lag_{lag}일"
    ] = grouped.shift(lag)


# =========================================================
# 10. Rolling Feature 생성
# =========================================================

for window in ROLLING_WINDOWS:

    demand[
        f"rolling_mean_{window}일"
    ] = (
        grouped
        .transform(
            lambda x:
            x
            .shift(1)
            .rolling(
                window,
                min_periods=1
            )
            .mean()
        )
    )

    demand[
        f"rolling_std_{window}일"
    ] = (
        grouped
        .transform(
            lambda x:
            x
            .shift(1)
            .rolling(
                window,
                min_periods=2
            )
            .std()
        )
    )


# =========================================================
# 11. 동일 요일 기반 변수
# =========================================================

demand["최근4주동요일평균"] = (
    demand[
        [
            "lag_7일",
            "lag_14일",
            "lag_28일"
        ]
    ]
    .mean(axis=1)
)


# =========================================================
# 12. 결측치 처리
# =========================================================

feature_fill_cols = [
    c
    for c in demand.columns
    if (
        c.startswith("lag_")
        or c.startswith("rolling_")
        or c == "최근4주동요일평균"
    )
]

demand[feature_fill_cols] = (
    demand[
        feature_fill_cols
    ]
    .fillna(0)
)


# =========================================================
# 13. 학습 / 테스트 분리
# =========================================================

print(
    "\n========== 학습 / 테스트 분리 =========="
)

train = demand[
    demand["날짜"] < TEST_START_DATE
].copy()

test = demand[
    demand["날짜"] >= TEST_START_DATE
].copy()

print(
    f"학습 기간: "
    f"{train['날짜'].min().date()} "
    f"~ {train['날짜'].max().date()}"
)

print(
    f"테스트 기간: "
    f"{test['날짜'].min().date()} "
    f"~ {test['날짜'].max().date()}"
)

print(
    f"학습 데이터: {len(train):,}개"
)

print(
    f"테스트 데이터: {len(test):,}개"
)


# =========================================================
# 14. Feature 정의
# =========================================================

FEATURES = [
    "출발구",

    "시간대",
    "요일번호",
    "주말여부",
    "월",
    "일",
    "주차",

    # v2 추가 변수
    "공휴일여부",
    "공휴일전날",
    "공휴일다음날",
    "휴일여부",
    "연휴여부",

    "시간_sin",
    "시간_cos",
    "요일_sin",
    "요일_cos",

    "lag_1일",
    "lag_2일",
    "lag_3일",
    "lag_7일",
    "lag_14일",
    "lag_28일",

    "rolling_mean_3일",
    "rolling_std_3일",

    "rolling_mean_7일",
    "rolling_std_7일",

    "rolling_mean_14일",
    "rolling_std_14일",

    "rolling_mean_28일",
    "rolling_std_28일",

    "최근4주동요일평균"
]

TARGET = "수요"

X_train = train[
    FEATURES
]

y_train = train[
    TARGET
]

X_test = test[
    FEATURES
]

y_test = test[
    TARGET
]


# =========================================================
# 15. LightGBM 모델
# =========================================================

print(
    "\n========== LightGBM v2 학습 =========="
)

model = LGBMRegressor(
    objective="poisson",

    n_estimators=800,

    learning_rate=0.03,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.9,

    colsample_bytree=0.9,

    reg_alpha=0.1,

    reg_lambda=0.2,

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)

model.fit(
    X_train,
    y_train,
    categorical_feature=[
        "출발구"
    ]
)


# =========================================================
# 16. 예측
# =========================================================

prediction = model.predict(
    X_test
)

prediction = np.clip(
    prediction,
    0,
    None
)

test["예측수요"] = prediction

test["예측오차"] = (
    test["수요"]
    - test["예측수요"]
)

test["절대오차"] = (
    test["예측오차"]
    .abs()
)


# =========================================================
# 17. 전체 모델 평가
# =========================================================

print(
    "\n========== v2 전체 예측 성능 =========="
)

mae = mean_absolute_error(
    y_test,
    prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediction
    )
)

r2 = r2_score(
    y_test,
    prediction
)

mean_demand = y_test.mean()

mae_ratio = (
    mae
    / mean_demand
    * 100
    if mean_demand > 0
    else 0
)

print(
    f"MAE: {mae:.3f}건"
)

print(
    f"RMSE: {rmse:.3f}건"
)

print(
    f"R²: {r2:.4f}"
)

print(
    f"테스트 평균 수요: "
    f"{mean_demand:.3f}건"
)

print(
    f"평균 수요 대비 MAE: "
    f"{mae_ratio:.2f}%"
)


# =========================================================
# 18. v1 성능과 직접 비교
# =========================================================

V1_MAE = 1.808
V1_RMSE = 3.267
V1_R2 = 0.8738

mae_improvement_v1 = (
    (V1_MAE - mae)
    / V1_MAE
    * 100
)

rmse_improvement_v1 = (
    (V1_RMSE - rmse)
    / V1_RMSE
    * 100
)

r2_change_v1 = (
    r2 - V1_R2
)

print(
    "\n========== v1 vs v2 =========="
)

print(
    f"v1 MAE: {V1_MAE:.3f}"
)

print(
    f"v2 MAE: {mae:.3f}"
)

print(
    f"MAE 개선율: "
    f"{mae_improvement_v1:+.2f}%"
)

print(
    f"v1 RMSE: {V1_RMSE:.3f}"
)

print(
    f"v2 RMSE: {rmse:.3f}"
)

print(
    f"RMSE 개선율: "
    f"{rmse_improvement_v1:+.2f}%"
)

print(
    f"v1 R²: {V1_R2:.4f}"
)

print(
    f"v2 R²: {r2:.4f}"
)

print(
    f"R² 변화: "
    f"{r2_change_v1:+.4f}"
)


# =========================================================
# 19. Naive Baseline
# =========================================================

baseline_pred = (
    test["lag_7일"]
    .clip(lower=0)
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_pred
    )
)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)

if baseline_mae > 0:

    baseline_mae_improvement = (
        (
            baseline_mae
            - mae
        )
        / baseline_mae
        * 100
    )

else:

    baseline_mae_improvement = 0.0


print(
    "\n========== Naive Baseline 비교 =========="
)

print(
    f"Baseline MAE: "
    f"{baseline_mae:.3f}"
)

print(
    f"LightGBM v2 MAE: "
    f"{mae:.3f}"
)

print(
    f"MAE 개선율: "
    f"{baseline_mae_improvement:.2f}%"
)

print(
    f"Baseline RMSE: "
    f"{baseline_rmse:.3f}"
)

print(
    f"LightGBM v2 RMSE: "
    f"{rmse:.3f}"
)

print(
    f"Baseline R²: "
    f"{baseline_r2:.4f}"
)

print(
    f"LightGBM v2 R²: "
    f"{r2:.4f}"
)


# =========================================================
# 20. 공휴일 / 비공휴일 성능 비교
# =========================================================

holiday_test = test[
    test["공휴일여부"] == 1
].copy()

normal_test = test[
    test["공휴일여부"] == 0
].copy()


def calculate_group_metrics(df):

    if len(df) == 0:

        return {
            "데이터수": 0,
            "실제수요": 0,
            "예측수요": 0,
            "MAE": np.nan,
            "RMSE": np.nan
        }

    group_mae = mean_absolute_error(
        df["수요"],
        df["예측수요"]
    )

    group_rmse = np.sqrt(
        mean_squared_error(
            df["수요"],
            df["예측수요"]
        )
    )

    return {
        "데이터수": len(df),
        "실제수요": df["수요"].sum(),
        "예측수요": df["예측수요"].sum(),
        "MAE": group_mae,
        "RMSE": group_rmse
    }


holiday_metrics = calculate_group_metrics(
    holiday_test
)

normal_metrics = calculate_group_metrics(
    normal_test
)

holiday_eval = pd.DataFrame(
    [
        {
            "구분": "공휴일",
            **holiday_metrics
        },
        {
            "구분": "비공휴일",
            **normal_metrics
        }
    ]
)

holiday_eval["총수요오차율_%"] = np.where(
    holiday_eval["실제수요"] > 0,
    (
        np.abs(
            holiday_eval["실제수요"]
            - holiday_eval["예측수요"]
        )
        / holiday_eval["실제수요"]
        * 100
    ),
    0
)

print(
    "\n========== 공휴일 vs 비공휴일 예측 성능 =========="
)

print(
    holiday_eval
    .round(3)
    .to_string(
        index=False
    )
)


# =========================================================
# 21. 개별 공휴일 성능
# =========================================================

holiday_detail = (
    holiday_test
    .groupby(
        [
            "날짜",
            "공휴일명"
        ],
        as_index=False
    )
    .agg(
        실제수요=(
            "수요",
            "sum"
        ),
        예측수요=(
            "예측수요",
            "sum"
        ),
        MAE=(
            "절대오차",
            "mean"
        )
    )
)

holiday_detail["총수요오차"] = (
    holiday_detail["실제수요"]
    - holiday_detail["예측수요"]
)

holiday_detail["총수요오차율_%"] = np.where(
    holiday_detail["실제수요"] > 0,
    (
        np.abs(
            holiday_detail["총수요오차"]
        )
        / holiday_detail["실제수요"]
        * 100
    ),
    0
)

print(
    "\n========== 테스트 기간 공휴일별 성능 =========="
)

if len(holiday_detail) > 0:

    print(
        holiday_detail
        .round(2)
        .to_string(
            index=False
        )
    )

else:

    print(
        "테스트 기간에 공휴일 데이터가 없습니다."
    )


# =========================================================
# 22. 크리스마스 시간대별 분석
# =========================================================

christmas = test[
    test["날짜"]
    == pd.Timestamp("2025-12-25")
].copy()

christmas_hourly = (
    christmas
    .groupby(
        "시간대",
        as_index=False
    )
    .agg(
        실제수요=(
            "수요",
            "sum"
        ),
        예측수요=(
            "예측수요",
            "sum"
        ),
        MAE=(
            "절대오차",
            "mean"
        )
    )
)

christmas_hourly["오차"] = (
    christmas_hourly["실제수요"]
    - christmas_hourly["예측수요"]
)

christmas_hourly["오차율_%"] = np.where(
    christmas_hourly["실제수요"] > 0,
    (
        np.abs(
            christmas_hourly["오차"]
        )
        / christmas_hourly["실제수요"]
        * 100
    ),
    0
)

print(
    "\n========== 크리스마스 시간대별 예측 =========="
)

print(
    christmas_hourly
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 23. 시간대별 성능
# =========================================================

hourly_eval = (
    test.groupby(
        "시간대",
        as_index=False
    )
    .agg(
        실제수요=("수요", "sum"),
        예측수요=("예측수요", "sum"),
        MAE=("절대오차", "mean")
    )
)

hourly_eval["총수요오차율_%"] = np.where(
    hourly_eval["실제수요"] > 0,

    (
        np.abs(
            hourly_eval["실제수요"]
            - hourly_eval["예측수요"]
        )
        / hourly_eval["실제수요"]
        * 100
    ),

    0
)

print(
    "\n========== 시간대별 예측 성능 =========="
)

print(
    hourly_eval
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 24. 출발구별 성능
# =========================================================

gu_eval = (
    test.groupby(
        "출발구",
        observed=True,
        as_index=False
    )
    .agg(
        실제수요=("수요", "sum"),
        예측수요=("예측수요", "sum"),
        MAE=("절대오차", "mean")
    )
)

gu_eval["총수요오차율_%"] = np.where(
    gu_eval["실제수요"] > 0,

    (
        np.abs(
            gu_eval["실제수요"]
            - gu_eval["예측수요"]
        )
        / gu_eval["실제수요"]
        * 100
    ),

    0
)

print(
    "\n========== 출발구별 예측 성능 =========="
)

print(
    gu_eval
    .sort_values(
        "MAE",
        ascending=False
    )
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 25. Feature Importance
# =========================================================

importance = pd.DataFrame(
    {
        "변수": FEATURES,

        "중요도": (
            model.feature_importances_
        )
    }
)

importance = (
    importance
    .sort_values(
        "중요도",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

importance["중요도_%"] = (
    importance["중요도"]
    / importance["중요도"].sum()
    * 100
)

print(
    "\n========== 변수 중요도 TOP 25 =========="
)

print(
    importance
    .head(25)
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 26. 공휴일 변수 중요도
# =========================================================

holiday_features = [
    "공휴일여부",
    "공휴일전날",
    "공휴일다음날",
    "휴일여부",
    "연휴여부"
]

holiday_importance = importance[
    importance["변수"].isin(
        holiday_features
    )
].copy()

print(
    "\n========== 공휴일 관련 변수 중요도 =========="
)

print(
    holiday_importance
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 27. 예측 오차 TOP 30
# =========================================================

print(
    "\n========== 예측 오차 TOP 30 =========="
)

error_cols = [
    "날짜",
    "요일",
    "공휴일명",
    "시간대",
    "출발구",
    "수요",
    "예측수요",
    "예측오차",
    "절대오차"
]

print(
    test
    .sort_values(
        "절대오차",
        ascending=False
    )
    .head(30)[
        error_cols
    ]
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 28. 정책용 집계
# =========================================================

policy_prediction = (
    test.groupby(
        [
            "요일",
            "시간대",
            "출발구"
        ],
        observed=True,
        as_index=False
    )
    .agg(
        실제평균수요=(
            "수요",
            "mean"
        ),

        예측평균수요=(
            "예측수요",
            "mean"
        ),

        평균절대오차=(
            "절대오차",
            "mean"
        )
    )
)

policy_prediction = (
    policy_prediction
    .rename(
        columns={
            "시간대":
            "승차시간대"
        }
    )
)


# =========================================================
# 29. 성능 요약
# =========================================================

summary = pd.DataFrame(
    {
        "지표": [
            "v2_MAE",
            "v2_RMSE",
            "v2_R2",
            "평균수요",
            "평균수요대비_MAE_%",
            "Baseline_MAE",
            "Baseline_RMSE",
            "Baseline_R2",
            "Baseline대비_MAE개선율_%",
            "v1_MAE",
            "v1_RMSE",
            "v1_R2",
            "v1대비_MAE개선율_%",
            "v1대비_RMSE개선율_%",
            "v1대비_R2변화",
            "공휴일_MAE",
            "비공휴일_MAE"
        ],

        "값": [
            mae,
            rmse,
            r2,
            mean_demand,
            mae_ratio,
            baseline_mae,
            baseline_rmse,
            baseline_r2,
            baseline_mae_improvement,
            V1_MAE,
            V1_RMSE,
            V1_R2,
            mae_improvement_v1,
            rmse_improvement_v1,
            r2_change_v1,
            holiday_metrics["MAE"],
            normal_metrics["MAE"]
        ]
    }
)


# =========================================================
# 30. 저장 경로
# =========================================================

prediction_path = (
    RESULT_DIR
    / "demand_forecast_test_v2_2025.csv"
)

policy_path = (
    RESULT_DIR
    / "demand_forecast_policy_v2_2025.csv"
)

hourly_path = (
    RESULT_DIR
    / "demand_forecast_hourly_v2_2025.csv"
)

gu_path = (
    RESULT_DIR
    / "demand_forecast_gu_v2_2025.csv"
)

importance_path = (
    RESULT_DIR
    / "demand_forecast_importance_v2_2025.csv"
)

summary_path = (
    RESULT_DIR
    / "demand_forecast_summary_v2_2025.csv"
)

holiday_path = (
    RESULT_DIR
    / "demand_forecast_holiday_v2_2025.csv"
)

christmas_path = (
    RESULT_DIR
    / "demand_forecast_christmas_v2_2025.csv"
)


# =========================================================
# 31. CSV 저장
# =========================================================

test.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)

policy_prediction.to_csv(
    policy_path,
    index=False,
    encoding="utf-8-sig"
)

hourly_eval.to_csv(
    hourly_path,
    index=False,
    encoding="utf-8-sig"
)

gu_eval.to_csv(
    gu_path,
    index=False,
    encoding="utf-8-sig"
)

importance.to_csv(
    importance_path,
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig"
)

holiday_detail.to_csv(
    holiday_path,
    index=False,
    encoding="utf-8-sig"
)

christmas_hourly.to_csv(
    christmas_path,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 32. 최종 출력
# =========================================================

print(
    "\n========== v2 최종 요약 =========="
)

print(
    f"v1 MAE → v2 MAE: "
    f"{V1_MAE:.3f} → {mae:.3f}"
)

print(
    f"v1 RMSE → v2 RMSE: "
    f"{V1_RMSE:.3f} → {rmse:.3f}"
)

print(
    f"v1 R² → v2 R²: "
    f"{V1_R2:.4f} → {r2:.4f}"
)

print(
    f"공휴일 MAE: "
    f"{holiday_metrics['MAE']:.3f}"
)

print(
    f"비공휴일 MAE: "
    f"{normal_metrics['MAE']:.3f}"
)

print(
    "\n========== 저장 완료 =========="
)

print(prediction_path)
print(policy_path)
print(hourly_path)
print(gu_path)
print(importance_path)
print(summary_path)
print(holiday_path)
print(christmas_path)