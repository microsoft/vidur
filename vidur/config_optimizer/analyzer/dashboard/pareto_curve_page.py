import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from paretoset import paretoset

from vidur.config_optimizer.analyzer.bottleneck_analyzer import BottleneckAnalyzer
from vidur.config_optimizer.analyzer.constants import AXIS_COLS
from vidur.config_optimizer.analyzer.dashboard.utils import (
    add_advanced_filters,
    add_model_trace_selector,
    add_small_divider,
    write_best_config,
)


def plot_pareto_curve(df, metric, percentile, slo):
    metric_col = f"{metric}_{percentile}%"

    # filter out points which are more than 1.5x away from the SLO limit
    df = df[(df[metric_col] <= slo * 1.5)]

    paretoset_mask = paretoset(df[[metric_col, "QPS per Dollar"]], sense=["min", "max"])
    pareto_df = df[paretoset_mask]

    fig, ax = plt.subplots()
    ax.scatter(
        df[metric_col],
        df["QPS per Dollar"],
        color="blue",
        alpha=0.7,
        label="Configs",
    )

    pareto_df = pareto_df.sort_values(metric_col)
    if not pareto_df.empty:
        ax.plot(
            pareto_df[metric_col],
            pareto_df["QPS per Dollar"],
            color="orange",
            marker="o",
            linestyle="-",
            label="Pareto Frontier",
        )

    ax.axvline(slo, color="red", linestyle="--", label="SLO Limit")
    ax.axvspan(0, slo, color="green", alpha=0.05, label="SLO Compliant")

    ax.set_xlim(0, slo * 1.25)
    ax.set_title(f"Pareto Curve for {metric} vs. QPS per Dollar")
    ax.set_xlabel(f"{metric} - P{percentile}")
    ax.set_ylabel("QPS per Dollar")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_metrics_scatter(
    df, metric_1, percentile_1, slo_1, metric_2, percentile_2, slo_2
):
    metric_1_col = f"{metric_1}_{percentile_1}%"
    metric_2_col = f"{metric_2}_{percentile_2}%"

    slo_compliant_df = df[(df[metric_1_col] <= slo_1) & (df[metric_2_col] <= slo_2)]
    best_config = slo_compliant_df.sort_values("QPS per Dollar", ascending=False).iloc[
        0
    ]
    # convert the best config to a dataframe
    best_config = pd.DataFrame([best_config])
    best_x = best_config[metric_1_col].iloc[0]
    best_y = best_config[metric_2_col].iloc[0]

    fig, ax = plt.subplots()

    scatter = ax.scatter(
        df[metric_1_col],
        df[metric_2_col],
        c=df["QPS per Dollar"],
        cmap="viridis",
        alpha=0.8,
        label="Configs",
    )

    ax.scatter(
        best_x,
        best_y,
        color="orange",
        marker="*",
        s=160,
        edgecolor="black",
        linewidth=1,
        label="Best Config",
    )

    ax.axvline(slo_1, color="red", linestyle="--", label=f"{metric_1} SLO")
    ax.axhline(slo_2, color="red", linestyle=":", label=f"{metric_2} SLO")

    ax.axvspan(0, slo_1, color="green", alpha=0.05)
    ax.axhspan(0, slo_2, color="green", alpha=0.05)

    ax.set_xlim(0, slo_1 * 1.25)
    ax.set_ylim(0, slo_2 * 1.25)

    ax.set_title(f"{metric_1} vs. {metric_2} Colored by QPS per Dollar")
    ax.set_xlabel(f"{metric_1} - P{percentile_1}")
    ax.set_ylabel(f"{metric_2} - P{percentile_2}")
    ax.grid(True, linestyle="--", alpha=0.6)
    legend = ax.legend()
    legend.set_title("")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("QPS per Dollar")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_pareto_curve_under_slos(
    bottleneck_analyzer,
    df,
    metric_1,
    percentile_1,
    slo_1,
    metric_2,
    percentile_2,
    slo_2,
):
    write_best_config(
        bottleneck_analyzer,
        df,
        metric_1,
        percentile_1,
        slo_1,
        metric_2,
        percentile_2,
        slo_2,
    )

    add_small_divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        plot_pareto_curve(df, metric_1, percentile_1, slo_1)
    with col2:
        plot_pareto_curve(df, metric_2, percentile_2, slo_2)
    with col3:
        plot_metrics_scatter(
            df, metric_1, percentile_1, slo_1, metric_2, percentile_2, slo_2
        )


def render_pareto_curve_page(subset_dfs):
    # create a subpage for best config selection
    st.markdown("## Pareto Curve Analysis")
    st.markdown(
        "This tool helps you visualize the trade-offs between different performance metrics and helps you identify the best configurations."
    )
    add_small_divider()
    st.markdown("### Input Parameters")

    model_select_box, trace_select_box = add_model_trace_selector(subset_dfs)

    subset_df = subset_dfs[(model_select_box, trace_select_box)]

    col1, col2 = st.columns(2)

    with col1:
        ttft_slo_percentile = st.selectbox(
            "TTFT SLO Percentile:", [50, 75, 90, 95, 99], index=2
        )
        ttft_slo_value = st.slider(
            "TTFT SLO Value (s)", min_value=0.1, max_value=10.0, value=2.0
        )

    with col2:
        tbt_slo_percentile = st.selectbox(
            "TBT SLO Percentile:", [50, 75, 90, 95, 99], index=4
        )
        tbt_slo_value = st.slider(
            "TBT SLO Value (ms)", min_value=10, max_value=2000, value=200
        )

    filtered_df = add_advanced_filters(subset_df)

    bottleneck_analyzer = BottleneckAnalyzer(
        ttft_slo_percentile,
        ttft_slo_value,
        tbt_slo_percentile,
        tbt_slo_value,
    )

    st.markdown("### Results")
    add_small_divider()

    plot_pareto_curve_under_slos(
        bottleneck_analyzer,
        filtered_df,
        "TTFT",
        ttft_slo_percentile,
        ttft_slo_value,
        "TBT",
        tbt_slo_percentile,
        tbt_slo_value,
    )
