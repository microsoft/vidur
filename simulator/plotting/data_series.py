import logging

import numpy as np
import pandas as pd
import plotly_express as px
import wandb

logger = logging.getLogger(__name__)


class DataSeries:
    def __init__(
        self,
        x_name: str,
        y_name: str,
        subsamples: int = 1000,
        save_table_to_wandb: bool = True,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self._data_series = []
        # column names of x, y datatpoints for data collection
        self._x_name = x_name
        self._y_name = y_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data_y = 0

        self._subsamples = subsamples
        self._save_table_to_wandb = save_table_to_wandb

    def __len__(self):
        return len(self._data_series)

    # add a new x, y datapoint
    def put(self, data_x: float, data_y: float) -> None:
        self._last_data_y = data_y
        self._data_series.append((data_x, data_y))

    # get most recently collected y datapoint
    def _peek_y(self):
        return self._last_data_y

    # convert list of x, y datapoints to a pandas dataframe
    def _to_df(self):
        return pd.DataFrame(self._data_series, columns=[self._x_name, self._y_name])

    # add a new x, y datapoint as an incremental (delta) update to
    # recently collected y datapoint
    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        last_data_y = self._peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y)

    def print_series_stats(
        self, df: pd.DataFrame, plot_name: str, x_name: str = None, y_name: str = None
    ) -> None:
        if x_name is None:
            x_name = self._x_name

        if y_name is None:
            y_name = self._y_name

        logger.info(
            f"{plot_name}: {y_name} stats:"
            f" min: {df[y_name].min()},"
            f" max: {df[y_name].max()},"
            f" mean: {df[y_name].mean()},"
        )
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": df[y_name].min(),
                    f"{plot_name}_max": df[y_name].max(),
                    f"{plot_name}_mean": df[y_name].mean(),
                },
                step=0,
            )

    def print_distribution_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str = None
    ) -> None:
        if y_name is None:
            y_name = self._y_name

        logger.info(
            f"{plot_name}: {y_name} stats:"
            f" min: {df[y_name].min()},"
            f" max: {df[y_name].max()},"
            f" mean: {df[y_name].mean()},"
            f" median: {df[y_name].median()},"
            f" 95th percentile: {df[y_name].quantile(0.95)},"
            f" 99th percentile: {df[y_name].quantile(0.99)}"
            f" 99.9th percentile: {df[y_name].quantile(0.999)}"
        )
        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": df[y_name].min(),
                    f"{plot_name}_max": df[y_name].max(),
                    f"{plot_name}_mean": df[y_name].mean(),
                    f"{plot_name}_median": df[y_name].median(),
                    f"{plot_name}_95th_percentile": df[y_name].quantile(0.95),
                    f"{plot_name}_99th_percentile": df[y_name].quantile(0.99),
                    f"{plot_name}_99.9th_percentile": df[y_name].quantile(0.999),
                },
                step=0,
            )

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv")
        if wandb.run and self._save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def plot_step(self, path: str, plot_name: str, y_axis_label: str = None) -> None:
        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self._to_df()

        self.print_series_stats(df, plot_name)

        # subsample
        if len(df) > self._subsamples:
            # pick self._subsamples from the dataframe
            # however, if we make the difference between indices constant
            # we might pick spurious periodic patterns
            # so we pick the indices with a constant difference in the indices with a random offset
            # this is to avoid picking spurious periodic patterns
            indices = np.arange(0, len(df), len(df) // self._subsamples)
            offsets = np.random.randint(0, 5)
            indices = (indices + offsets) % len(df)
            df = df.iloc[indices]

        # change marker color to red
        fig = px.line(
            df, x=self._x_name, y=self._y_name, markers=True, labels={"x": y_axis_label}
        )
        fig.update_traces(marker=dict(color="red", size=2))

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_step": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        self._x_name,
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)

    def plot_cdf(self, path: str, plot_name: str, y_axis_label: str = None) -> None:
        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self._to_df()

        self.print_distribution_stats(df, plot_name)

        df["cdf"] = df[self._y_name].rank(method="first", pct=True)
        # sort by cdf
        df = df.sort_values(by=["cdf"])

        # subsample
        if len(df) > self._subsamples:
            df = df.iloc[:: len(df) // self._subsamples]

        fig = px.line(
            df, x=self._y_name, y="cdf", markers=True, labels={"x": y_axis_label}
        )
        fig.update_traces(marker=dict(color="red", size=2))

        if wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)

    def plot_histogram(self, path: str, plot_name: str) -> None:
        df = self._to_df()

        self.print_distribution_stats(df, plot_name)

        fig = px.histogram(df, x=self._y_name, nbins=25)

        # wandb histogram is highly inaccurate so we need to generate the histogram
        # ourselves and then use wandb bar chart

        histogram_df = df[self._y_name].value_counts(bins=25, sort=False).sort_index()
        histogram_df = histogram_df.reset_index()
        histogram_df.columns = ["Bins", "count"]
        histogram_df["Bins"] = histogram_df["Bins"].apply(lambda x: x.mid)
        histogram_df = histogram_df.sort_values(by=["Bins"])
        # convert to percentage
        histogram_df["Percentage"] = histogram_df["count"] * 100 / len(df)
        # drop bins with less than 0.1% of the total count
        histogram_df = histogram_df[histogram_df["Percentage"] > 0.1]

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_histogram": wandb.plot.bar(
                        wandb.Table(dataframe=histogram_df),
                        "Bins",
                        "Percentage",  # wandb plots are horizontal
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")

    def plot_differential(self, path: str, plot_name: str) -> None:
        df = self._to_df()

        differential_col_name = f"{self._y_name}/{self._x_name}"

        # if there are repeated values of x, then we cannot compute a differential
        # in that, case aggreagate the y values for the same x value
        df = df.groupby(self._x_name).max().reset_index().sort_values(self._x_name)

        # compute differential
        df[differential_col_name] = df[self._y_name].diff() / df[self._x_name].diff()

        df = df.dropna()

        self.print_distribution_stats(df, plot_name, differential_col_name)

        # subsample
        if len(df) > self._subsamples:
            df = df.iloc[:: len(df) // self._subsamples]

        # change marker color to red
        fig = px.line(df, x=self._x_name, y=differential_col_name, markers=True)
        fig.update_traces(marker=dict(color="red", size=2))

        if wandb.run:
            wandb.log(
                {
                    f"{plot_name}_differential": wandb.plot.line(
                        wandb.Table(dataframe=df),
                        self._x_name,
                        differential_col_name,
                        title=plot_name,
                    )
                },
                step=0,
            )

        fig.write_image(f"{path}/{plot_name}.png")
        self._save_df(df, path, plot_name)
