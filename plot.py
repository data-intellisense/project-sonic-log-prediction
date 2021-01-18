import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def plot_logs(
    df,
    well_name="well",
    DTSM_only=True,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
):

    df = df.copy()

    # drop part of logs where no DTSM data
    if DTSM_only:
        df.dropna(subset=["DTSM"], inplace=True)

    fig = go.Figure()

    for col in df.columns:

        # do not plots in the below list:
        if col not in ["DEPT", "TEND", "TENR"]:
            fig.add_trace(go.Scatter(x=df[col], y=df["DEPT"], name=col))

    fig.update_layout(
        showlegend=True,
        title=dict(text=well_name),
        yaxis=dict(autorange="reversed", title="Depth, ft"),
        font=dict(size=18),
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig