# ============================================================
# map_visualization.py
#
# 서울시 UD택시 수요예측 및 동적 재배치 지도 시각화
#
# 생성 그래프
# 11_actual_demand_map.png
# 12_predicted_demand_map.png
# 13_prediction_error_map.png
# 14_dynamic_allocation_map.png
# 15_rebalancing_flow_map.png
# 16_rebalancing_flow_45min.png
# 17_map_dashboard.png
#
# 기준:
# - LightGBM v2 수요예측
# - forecast_based_rebalancing_v5 결과
# - 정책 권장안: 45분
# ============================================================

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings("ignore")


# ============================================================
# 0. 경로 설정
# ============================================================

BASE_DIR = Path("/Users/sinhuijae/Desktop/공모전/데이터 분석")

RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULT_DIR / "figures"

GEO_FILE = (
    BASE_DIR
    / "data"
    / "seoul-maps"
    / "juso"
    / "2015"
    / "json"
    / "seoul_municipalities_geo_simple.json"
)

FORECAST_FILE = RESULT_DIR / "demand_forecast_test_v2_2025.csv"

ALLOCATION_DETAIL_FILE = (
    RESULT_DIR / "allocation_detail_v5_2025.csv"
)

ALLOCATION_SUMMARY_FILE = (
    RESULT_DIR / "allocation_summary_v5_2025.csv"
)

ROUTE_FILE = (
    RESULT_DIR / "allocation_routes_v5_2025.csv"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. 한글 폰트
# ============================================================

mpl.rcParams["axes.unicode_minus"] = False

font_candidates = [
    "AppleGothic",
    "Arial Unicode MS",
    "NanumGothic"
]

available_fonts = {
    f.name
    for f in mpl.font_manager.fontManager.ttflist
}

selected_font = None

for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

if selected_font is not None:
    mpl.rcParams["font.family"] = selected_font
    print(f"사용 한글 폰트: {selected_font}")
else:
    print("※ 적절한 한글 폰트를 자동으로 찾지 못했습니다.")


# ============================================================
# 2. 기본 함수
# ============================================================

def normalize_gu(value):
    """
    구 이름 정리
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    value = value.replace(
        "서울특별시",
        ""
    ).strip()

    value = value.replace(
        "서울시",
        ""
    ).strip()

    return value


def load_geojson():
    """
    서울 25개 구 GeoJSON 로드
    """

    with open(
        GEO_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        geo = json.load(f)

    return geo


def get_polygons(feature):
    """
    Polygon / MultiPolygon을
    matplotlib Polygon 좌표로 변환
    """

    geometry = feature["geometry"]

    geom_type = geometry["type"]
    coords = geometry["coordinates"]

    polygons = []

    if geom_type == "Polygon":

        for ring in coords[:1]:
            polygons.append(
                np.asarray(ring)
            )

    elif geom_type == "MultiPolygon":

        for polygon in coords:
            for ring in polygon[:1]:
                polygons.append(
                    np.asarray(ring)
                )

    return polygons


def polygon_centroid(feature):
    """
    단순 polygon 평균 중심점 계산
    """

    polygons = get_polygons(feature)

    if not polygons:
        return np.nan, np.nan

    all_points = np.vstack(polygons)

    return (
        float(all_points[:, 0].mean()),
        float(all_points[:, 1].mean())
    )


def build_centroids(geo):
    """
    각 구 중심 좌표 생성
    """

    result = {}

    for feature in geo["features"]:

        gu = normalize_gu(
            feature["properties"]["SIG_KOR_NM"]
        )

        x, y = polygon_centroid(feature)

        result[gu] = (x, y)

    return result


def draw_base_map(
    ax,
    geo,
    facecolor="#f7f7f7",
    edgecolor="white",
    linewidth=0.8
):
    """
    서울 기본 지도
    """

    for feature in geo["features"]:

        for poly in get_polygons(feature):

            patch = Polygon(
                poly,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth
            )

            ax.add_patch(patch)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")


def draw_choropleth(
    ax,
    geo,
    value_dict,
    cmap="YlOrRd",
    title="",
    label_format="{:.0f}",
    center_zero=False,
    show_labels=True
):
    """
    구별 Choropleth 지도
    """

    values = np.array(
        [
            value_dict.get(
                normalize_gu(
                    f["properties"]["SIG_KOR_NM"]
                ),
                np.nan
            )
            for f in geo["features"]
        ],
        dtype=float
    )

    valid_values = values[
        np.isfinite(values)
    ]

    if len(valid_values) == 0:
        vmin = 0
        vmax = 1
    else:
        vmin = np.nanmin(valid_values)
        vmax = np.nanmax(valid_values)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1

    if center_zero:

        max_abs = max(
            abs(vmin),
            abs(vmax)
        )

        norm = TwoSlopeNorm(
            vmin=-max_abs,
            vcenter=0,
            vmax=max_abs
        )

    else:

        norm = mpl.colors.Normalize(
            vmin=vmin,
            vmax=vmax
        )

    colormap = plt.get_cmap(cmap)

    patches = []
    patch_values = []

    for feature in geo["features"]:

        gu = normalize_gu(
            feature["properties"]["SIG_KOR_NM"]
        )

        value = value_dict.get(
            gu,
            np.nan
        )

        for poly in get_polygons(feature):

            patches.append(
                Polygon(
                    poly,
                    closed=True
                )
            )

            patch_values.append(
                value
            )

    collection = PatchCollection(
        patches,
        cmap=colormap,
        norm=norm,
        edgecolor="white",
        linewidth=0.8
    )

    collection.set_array(
        np.asarray(
            patch_values,
            dtype=float
        )
    )

    ax.add_collection(collection)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    cbar = plt.colorbar(
        collection,
        ax=ax,
        fraction=0.035,
        pad=0.01
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    if show_labels:

        centroids = build_centroids(geo)

        for gu, (x, y) in centroids.items():

            value = value_dict.get(
                gu,
                np.nan
            )

            if np.isfinite(value):

                label = (
                    f"{gu}\n"
                    + label_format.format(value)
                )

            else:

                label = gu

            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                fontweight="bold"
            )

    return collection


def save_figure(
    fig,
    filename
):
    """
    그림 저장
    """

    output = FIGURE_DIR / filename

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"저장 완료: {output}"
    )


# ============================================================
# 3. 데이터 로드
# ============================================================

print(
    "\n========== 지도 시각화 데이터 로드 =========="
)

geo = load_geojson()

print(
    f"서울 자치구: {len(geo['features'])}개"
)

forecast = pd.read_csv(
    FORECAST_FILE,
    encoding="utf-8-sig"
)

summary = pd.read_csv(
    ALLOCATION_SUMMARY_FILE,
    encoding="utf-8-sig"
)

routes = pd.read_csv(
    ROUTE_FILE,
    encoding="utf-8-sig"
)

print(
    f"수요예측 데이터: {len(forecast):,}개"
)

print(
    f"정책 비교 데이터: {len(summary):,}개"
)

print(
    f"재배치 경로 데이터: {len(routes):,}개"
)


# ============================================================
# 4. 구 이름 정리
# ============================================================

forecast["출발구"] = (
    forecast["출발구"]
    .apply(normalize_gu)
)

routes["재배치출발구"] = (
    routes["재배치출발구"]
    .apply(normalize_gu)
)

routes["재배치도착구"] = (
    routes["재배치도착구"]
    .apply(normalize_gu)
)


# ============================================================
# 5. 실제 / 예측 구별 수요
# ============================================================

gu_demand = (
    forecast
    .groupby(
        "출발구",
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
        )
    )
)

gu_demand["예측오차"] = (
    gu_demand["예측수요"]
    - gu_demand["실제수요"]
)

gu_demand["예측오차율_%"] = np.where(
    gu_demand["실제수요"] > 0,
    gu_demand["예측오차"]
    / gu_demand["실제수요"]
    * 100,
    np.nan
)

actual_dict = dict(
    zip(
        gu_demand["출발구"],
        gu_demand["실제수요"]
    )
)

prediction_dict = dict(
    zip(
        gu_demand["출발구"],
        gu_demand["예측수요"]
    )
)

error_dict = dict(
    zip(
        gu_demand["출발구"],
        gu_demand["예측오차율_%"]
    )
)


# ============================================================
# 6. 정책 결과 추출
# ============================================================

def get_policy_row(
    policy,
    time_limit
):

    temp = summary[
        (
            summary["정책"]
            == policy
        )
        &
        (
            summary["시간제한_분"]
            == time_limit
        )
    ]

    if len(temp) == 0:
        return None

    return temp.iloc[0]


current_rows = summary[
    summary["정책"]
    == "현재배치"
]

fixed_rows = summary[
    summary["정책"]
    == "고정배치"
]

lightgbm_45 = get_policy_row(
    "LightGBM선제",
    45
)

lightgbm_60 = get_policy_row(
    "LightGBM선제",
    60
)


if len(current_rows) > 0:

    current_coverage = float(
        current_rows.iloc[0][
            "수요가중커버리지_%"
        ]
    )

else:

    current_coverage = np.nan


if len(fixed_rows) > 0:

    fixed_coverage = float(
        fixed_rows.iloc[0][
            "수요가중커버리지_%"
        ]
    )

else:

    fixed_coverage = np.nan


if lightgbm_45 is not None:

    coverage_45 = float(
        lightgbm_45[
            "수요가중커버리지_%"
        ]
    )

    rebalancing_45 = float(
        lightgbm_45[
            "총재배치UD"
        ]
    )

    empty_45 = float(
        lightgbm_45[
            "차량없는수요비율_%"
        ]
    )

else:

    coverage_45 = np.nan
    rebalancing_45 = np.nan
    empty_45 = np.nan


if lightgbm_60 is not None:

    coverage_60 = float(
        lightgbm_60[
            "수요가중커버리지_%"
        ]
    )

else:

    coverage_60 = np.nan


print(
    "\n========== 정책 핵심값 =========="
)

print(
    f"고정배치 커버리지: "
    f"{fixed_coverage:.2f}%"
)

print(
    f"현재배치 커버리지: "
    f"{current_coverage:.2f}%"
)

print(
    f"45분 LightGBM: "
    f"{coverage_45:.2f}%"
)

print(
    f"60분 LightGBM: "
    f"{coverage_60:.2f}%"
)


# ============================================================
# GRAPH 11
# 실제 수요 지도
# ============================================================

print(
    "\n========== MAP 11 =========="
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

draw_choropleth(
    ax=ax,
    geo=geo,
    value_dict=actual_dict,
    cmap="YlOrRd",
    title=(
        "서울시 장애인콜택시 실제 수요 분포\n"
        "2025년 11~12월"
    ),
    label_format="{:,.0f}"
)

fig.text(
    0.5,
    0.03,
    "색이 진할수록 실제 장애인콜택시 수요가 많은 지역",
    ha="center",
    fontsize=10
)

save_figure(
    fig,
    "11_actual_demand_map.png"
)


# ============================================================
# GRAPH 12
# LightGBM 예측 수요 지도
# ============================================================

print(
    "\n========== MAP 12 =========="
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

draw_choropleth(
    ax=ax,
    geo=geo,
    value_dict=prediction_dict,
    cmap="YlGnBu",
    title=(
        "LightGBM 예측 수요 분포\n"
        "2025년 11~12월"
    ),
    label_format="{:,.0f}"
)

fig.text(
    0.5,
    0.03,
    "LightGBM이 예측한 시간대·지역별 수요를 구 단위로 집계",
    ha="center",
    fontsize=10
)

save_figure(
    fig,
    "12_predicted_demand_map.png"
)


# ============================================================
# GRAPH 13
# 예측 오차 지도
# ============================================================

print(
    "\n========== MAP 13 =========="
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

draw_choropleth(
    ax=ax,
    geo=geo,
    value_dict=error_dict,
    cmap="RdBu_r",
    title=(
        "LightGBM 구별 수요 예측 오차율\n"
        "양수 = 과대예측 / 음수 = 과소예측"
    ),
    label_format="{:+.1f}%",
    center_zero=True
)

fig.text(
    0.5,
    0.03,
    "구별 총 예측수요와 실제수요의 차이를 실제수요 대비 비율로 계산",
    ha="center",
    fontsize=10
)

save_figure(
    fig,
    "13_prediction_error_map.png"
)


# ============================================================
# 7. 45분 재배치 경로 준비
# ============================================================

route_45 = routes[
    (
        routes["정책"]
        == "LightGBM선제"
    )
    &
    (
        routes["시간제한_분"]
        == 45
    )
].copy()

route_45 = route_45.sort_values(
    "총재배치UD",
    ascending=False
)

centroids = build_centroids(
    geo
)


# ============================================================
# 8. 구별 재배치 유입 / 유출 계산
# ============================================================

outflow = (
    route_45
    .groupby("재배치출발구")[
        "총재배치UD"
    ]
    .sum()
)

inflow = (
    route_45
    .groupby("재배치도착구")[
        "총재배치UD"
    ]
    .sum()
)

all_gu = [
    normalize_gu(
        f["properties"]["SIG_KOR_NM"]
    )
    for f in geo["features"]
]

net_flow_dict = {}

for gu in all_gu:

    incoming = float(
        inflow.get(
            gu,
            0
        )
    )

    outgoing = float(
        outflow.get(
            gu,
            0
        )
    )

    net_flow_dict[gu] = (
        incoming - outgoing
    )


# ============================================================
# GRAPH 14
# 45분 동적 배치 순유입 지도
# ============================================================

print(
    "\n========== MAP 14 =========="
)

fig, ax = plt.subplots(
    figsize=(10, 9)
)

draw_choropleth(
    ax=ax,
    geo=geo,
    value_dict=net_flow_dict,
    cmap="RdBu",
    title=(
        "LightGBM 45분 선제 재배치의 지역별 차량 순이동\n"
        "양수 = 순유입 / 음수 = 순유출"
    ),
    label_format="{:+.0f}",
    center_zero=True
)

fig.text(
    0.5,
    0.03,
    (
        f"45분 정책 총 재배치량 "
        f"{rebalancing_45:,.0f}대·회"
    ),
    ha="center",
    fontsize=10
)

save_figure(
    fig,
    "14_dynamic_allocation_map.png"
)


# ============================================================
# 재배치 경로 지도 함수
# ============================================================

def draw_route_map(
    route_df,
    title,
    top_n=20,
    filename=None,
    subtitle=None
):

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    draw_base_map(
        ax,
        geo,
        facecolor="#eeeeee",
        edgecolor="white",
        linewidth=1
    )

    route_plot = (
        route_df
        .sort_values(
            "총재배치UD",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    if len(route_plot) == 0:

        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold"
        )

        if filename:
            save_figure(
                fig,
                filename
            )

        return

    max_flow = route_plot[
        "총재배치UD"
    ].max()

    for _, row in route_plot.iterrows():

        origin = row[
            "재배치출발구"
        ]

        destination = row[
            "재배치도착구"
        ]

        flow = float(
            row[
                "총재배치UD"
            ]
        )

        if (
            origin not in centroids
            or destination not in centroids
        ):
            continue

        x1, y1 = centroids[origin]
        x2, y2 = centroids[destination]

        width = (
            0.8
            + 5.0
            * flow
            / max_flow
        )

        alpha = (
            0.35
            + 0.55
            * flow
            / max_flow
        )

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10 + width * 2,
            linewidth=width,
            alpha=alpha,
            connectionstyle="arc3,rad=0.08"
        )

        ax.add_patch(
            arrow
        )

    # 구 이름
    for gu, (x, y) in centroids.items():

        ax.scatter(
            x,
            y,
            s=16,
            zorder=5
        )

        ax.text(
            x,
            y,
            gu,
            fontsize=7,
            ha="center",
            va="bottom",
            fontweight="bold",
            zorder=6
        )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=15
    )

    if subtitle:

        fig.text(
            0.5,
            0.035,
            subtitle,
            ha="center",
            fontsize=10
        )

    if filename:

        save_figure(
            fig,
            filename
        )


# ============================================================
# GRAPH 15
# 60분 재배치 경로
# ============================================================

print(
    "\n========== MAP 15 =========="
)

route_60 = routes[
    (
        routes["정책"]
        == "LightGBM선제"
    )
    &
    (
        routes["시간제한_분"]
        == 60
    )
].copy()

draw_route_map(
    route_df=route_60,
    title=(
        "LightGBM 60분 선제 재배치 주요 경로"
    ),
    top_n=20,
    filename="15_rebalancing_flow_map.png",
    subtitle=(
        "화살표가 굵을수록 해당 구간의 누적 재배치 차량이 많음"
    )
)


# ============================================================
# GRAPH 16
# ★ 최종 정책안 45분
# ============================================================

print(
    "\n========== MAP 16 =========="
)

improvement_45 = (
    coverage_45
    - current_coverage
)

draw_route_map(
    route_df=route_45,
    title=(
        "권장 정책안: 45분 이내 수요예측 기반 선제 재배치"
    ),
    top_n=20,
    filename="16_rebalancing_flow_45min.png",
    subtitle=(
        f"현재배치 {current_coverage:.2f}%"
        f" → 45분 동적배치 {coverage_45:.2f}%"
        f"  |  +{improvement_45:.2f}%p"
    )
)


# ============================================================
# GRAPH 17
# ★ 공모전용 지도 Dashboard
# ============================================================

print(
    "\n========== MAP 17 =========="
)

fig = plt.figure(
    figsize=(18, 13)
)

gs = fig.add_gridspec(
    2,
    2,
    hspace=0.15,
    wspace=0.08
)


# ------------------------------------------------------------
# Dashboard A
# 실제 수요
# ------------------------------------------------------------

ax1 = fig.add_subplot(
    gs[0, 0]
)

draw_choropleth(
    ax=ax1,
    geo=geo,
    value_dict=actual_dict,
    cmap="YlOrRd",
    title="① 실제 장애인콜택시 수요",
    label_format="{:,.0f}",
    show_labels=False
)


# ------------------------------------------------------------
# Dashboard B
# 예측 수요
# ------------------------------------------------------------

ax2 = fig.add_subplot(
    gs[0, 1]
)

draw_choropleth(
    ax=ax2,
    geo=geo,
    value_dict=prediction_dict,
    cmap="YlGnBu",
    title="② LightGBM 예측 수요",
    label_format="{:,.0f}",
    show_labels=False
)


# ------------------------------------------------------------
# Dashboard C
# 순재배치
# ------------------------------------------------------------

ax3 = fig.add_subplot(
    gs[1, 0]
)

draw_choropleth(
    ax=ax3,
    geo=geo,
    value_dict=net_flow_dict,
    cmap="RdBu",
    title="③ 45분 정책 차량 순이동",
    label_format="{:+.0f}",
    center_zero=True,
    show_labels=False
)


# ------------------------------------------------------------
# Dashboard D
# 재배치 경로
# ------------------------------------------------------------

ax4 = fig.add_subplot(
    gs[1, 1]
)

draw_base_map(
    ax4,
    geo,
    facecolor="#eeeeee",
    edgecolor="white",
    linewidth=0.8
)

route_top = (
    route_45
    .sort_values(
        "총재배치UD",
        ascending=False
    )
    .head(15)
)

if len(route_top) > 0:

    max_flow = route_top[
        "총재배치UD"
    ].max()

    for _, row in route_top.iterrows():

        origin = row[
            "재배치출발구"
        ]

        destination = row[
            "재배치도착구"
        ]

        flow = float(
            row[
                "총재배치UD"
            ]
        )

        if (
            origin not in centroids
            or destination not in centroids
        ):
            continue

        x1, y1 = centroids[
            origin
        ]

        x2, y2 = centroids[
            destination
        ]

        width = (
            0.7
            + 4
            * flow
            / max_flow
        )

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10 + width,
            linewidth=width,
            alpha=0.6,
            connectionstyle="arc3,rad=0.08"
        )

        ax4.add_patch(
            arrow
        )

    for gu, (x, y) in centroids.items():

        ax4.scatter(
            x,
            y,
            s=10,
            zorder=5
        )

        ax4.text(
            x,
            y,
            gu,
            fontsize=6,
            ha="center",
            va="bottom",
            zorder=6
        )

ax4.set_title(
    "④ 주요 선제 재배치 경로",
    fontsize=16,
    fontweight="bold",
    pad=15
)


# ------------------------------------------------------------
# 전체 제목
# ------------------------------------------------------------

fig.suptitle(
    (
        "서울 UD택시 수요예측 기반 동적 배치 정책"
    ),
    fontsize=23,
    fontweight="bold",
    y=0.98
)


# ------------------------------------------------------------
# 핵심 정책 결과
# ------------------------------------------------------------

policy_text = (
    f"현재 차량배치  {current_coverage:.2f}%"
    f"   →   "
    f"45분 동적배치  {coverage_45:.2f}%"
    f"   "
    f"(+{improvement_45:.2f}%p)"
    f"\n"
    f"60분 최대 성능  {coverage_60:.2f}%"
    f"   |   "
    f"45분 정책 차량없는수요비율  {empty_45:.3f}%"
)

fig.text(
    0.5,
    0.015,
    policy_text,
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.7",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
)


save_figure(
    fig,
    "17_map_dashboard.png"
)


# ============================================================
# 9. 구별 결과 CSV 저장
# ============================================================

gu_output = gu_demand.copy()

gu_output["45분_재배치유입UD"] = (
    gu_output["출발구"]
    .map(inflow)
    .fillna(0)
)

gu_output["45분_재배치유출UD"] = (
    gu_output["출발구"]
    .map(outflow)
    .fillna(0)
)

gu_output["45분_순재배치UD"] = (
    gu_output["45분_재배치유입UD"]
    - gu_output["45분_재배치유출UD"]
)

GU_OUTPUT_FILE = (
    RESULT_DIR
    / "map_gu_summary_v5_2025.csv"
)

gu_output.to_csv(
    GU_OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 콘솔 요약
# ============================================================

print(
    "\n========== 지도 분석 핵심 결과 =========="
)

print(
    f"현재배치 커버리지: "
    f"{current_coverage:.2f}%"
)

print(
    f"45분 LightGBM 커버리지: "
    f"{coverage_45:.2f}%"
)

print(
    f"현재 대비 개선: "
    f"+{improvement_45:.2f}%p"
)

print(
    f"60분 최대 커버리지: "
    f"{coverage_60:.2f}%"
)

print(
    f"45분 총 재배치량: "
    f"{rebalancing_45:,.2f}대·회"
)

print(
    f"45분 차량없는수요비율: "
    f"{empty_45:.3f}%"
)


# ------------------------------------------------------------
# 순유입 TOP 5
# ------------------------------------------------------------

net_series = pd.Series(
    net_flow_dict
).sort_values(
    ascending=False
)

print(
    "\n========== 45분 차량 순유입 TOP 5 =========="
)

for gu, value in net_series.head(5).items():

    print(
        f"{gu}: "
        f"+{value:.2f}대·회"
    )


# ------------------------------------------------------------
# 순유출 TOP 5
# ------------------------------------------------------------

print(
    "\n========== 45분 차량 순유출 TOP 5 =========="
)

for gu, value in (
    net_series
    .sort_values()
    .head(5)
    .items()
):

    print(
        f"{gu}: "
        f"{value:.2f}대·회"
    )


# ------------------------------------------------------------
# 주요 경로 TOP 10
# ------------------------------------------------------------

print(
    "\n========== 45분 주요 재배치 경로 TOP 10 =========="
)

route_print = (
    route_45
    .sort_values(
        "총재배치UD",
        ascending=False
    )
    .head(10)
)

print(
    route_print[
        [
            "재배치출발구",
            "재배치도착구",
            "총재배치UD",
            "평균이동시간_분"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 완료
# ============================================================

print(
    "\n============================================"
)

print(
    "모든 지도 시각화 생성 완료"
)

print(
    "============================================"
)

print(
    f"저장 폴더:\n{FIGURE_DIR}"
)

print(
    "\n생성 파일:"
)

for filename in [
    "11_actual_demand_map.png",
    "12_predicted_demand_map.png",
    "13_prediction_error_map.png",
    "14_dynamic_allocation_map.png",
    "15_rebalancing_flow_map.png",
    "16_rebalancing_flow_45min.png",
    "17_map_dashboard.png"
]:

    print(
        f"- {filename}"
    )

print(
    f"\n구별 분석 CSV:\n{GU_OUTPUT_FILE}"
)