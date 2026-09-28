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

# 마지막 2개월을 테스트 기간으로 사용
TEST_START_DATE = pd.Timestamp("2025-11-01")

# 이전 며칠의 동일 시간대 수요
LAG_DAYS = [
    1,
    2,
    3,
    7,
    14,
    28
]

# 이동평균 기간
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
# 2. 원자료 로드
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
# 3. 날짜 × 시간 × 구 수요 생성
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


# ---------------------------------------------------------
# 운행이 0건인 조합도 포함
# ---------------------------------------------------------

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
# 4. 달력 변수 생성
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
# 5. 시간 주기성 변수
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
# 6. 출발구 범주형 변수
# =========================================================

demand["출발구"] = (
    demand["출발구"]
    .astype("category")
)


# =========================================================
# 7. Lag Feature 생성
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
# 8. Rolling Feature 생성
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
# 9. 동일 요일 기반 변수
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
# 10. 결측치 처리
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
# 11. 학습 / 테스트 분리
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
# 12. Feature 정의
# =========================================================

FEATURES = [
    "출발구",
    "시간대",
    "요일번호",
    "주말여부",
    "월",
    "일",
    "주차",

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
# 13. LightGBM 모델
# =========================================================

print(
    "\n========== LightGBM 학습 =========="
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
# 14. 예측
# =========================================================

prediction = model.predict(
    X_test
)

# 수요는 음수가 될 수 없음
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
# 15. 모델 평가
# =========================================================

print(
    "\n========== 전체 예측 성능 =========="
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
# 16. Naive Baseline
# =========================================================
# 지난주 동일 요일/시간/지역 수요를 그대로 예측

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

# ---------------------------------------------------------
# Baseline 대비 MAE 개선율
# ---------------------------------------------------------

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
    f"LightGBM MAE: "
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
    f"LightGBM RMSE: "
    f"{rmse:.3f}"
)

print(
    f"Baseline R²: "
    f"{baseline_r2:.4f}"
)

print(
    f"LightGBM R²: "
    f"{r2:.4f}"
)


# =========================================================
# 17. 시간대별 성능
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
# 18. 출발구별 성능
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
# 19. Feature Importance
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
    "\n========== 변수 중요도 TOP 20 =========="
)

print(
    importance
    .head(20)
    .round(2)
    .to_string(
        index=False
    )
)


# =========================================================
# 20. 오차가 큰 조합
# =========================================================

print(
    "\n========== 예측 오차 TOP 30 =========="
)

error_cols = [
    "날짜",
    "요일",
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
# 21. 정책용 집계
# =========================================================
# 날짜별 실제/예측 수요를
# 요일 × 시간 × 구 수준으로 평균
#
# 이후 predicted_dynamic_allocation.py에서 사용

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
# 22. 성능 요약 저장
# =========================================================

summary = pd.DataFrame(
    {
        "지표": [
            "MAE",
            "RMSE",
            "R2",
            "평균수요",
            "평균수요대비_MAE_%",
            "Baseline_MAE",
            "Baseline_RMSE",
            "Baseline_R2",
            "Baseline대비_MAE개선율_%"
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
            baseline_mae_improvement
        ]
    }
)


# =========================================================
# 23. 저장
# =========================================================

prediction_path = (
    RESULT_DIR
    / "demand_forecast_test_2025.csv"
)

policy_path = (
    RESULT_DIR
    / "demand_forecast_policy_2025.csv"
)

hourly_path = (
    RESULT_DIR
    / "demand_forecast_hourly_2025.csv"
)

gu_path = (
    RESULT_DIR
    / "demand_forecast_gu_2025.csv"
)

importance_path = (
    RESULT_DIR
    / "demand_forecast_importance_2025.csv"
)

summary_path = (
    RESULT_DIR
    / "demand_forecast_summary_2025.csv"
)


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


print(
    "\n========== 저장 완료 =========="
)

print(
    prediction_path
)

print(
    policy_path
)

print(
    hourly_path
)

print(
    gu_path
)

print(
    importance_path
)

print(
    summary_path
)