import pandas as pd
import numpy as np
import plotly.express as px
import prince

def get_pca_nd(df: pd.DataFrame, predict: np.ndarray, ndim: int=3) -> tuple[prince.PCA, pd.DataFrame]:
    pca_nd_object = prince.PCA(
        n_components=ndim,
        n_iter=3,
        rescale_with_mean=True,
        rescale_with_std=True,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=42
    )

    pca_nd_object.fit(df)

    df_pca_nd = pca_nd_object.transform(df)
    df_pca_nd.columns = [f"comp{i+1}" for i in range(ndim)]
    df_pca_nd["cluster"] = predict 

    return pca_nd_object, df_pca_nd

def get_mca_nd(df: pd.DataFrame, predict: np.ndarray, ndim: int=3) -> tuple[prince.MCA, pd.DataFrame]:
    mca = prince.MCA(
        n_components=ndim,
        n_iter=100,
        random_state=101
    )
    mca_df = mca.fit_transform(df)
    mca_df.columns = [f"comp{i + 1}" for i in range(ndim)]
    mca_df["cluster"] = predict

    return mca, mca_df

def plot_pca_3d(df: pd.DataFrame, title: str = "PCA Space", opacity: float = 0.8, width_line: float = 0.1):

    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")

    fig = px.scatter_3d(
        df, 
        x='comp1', 
        y='comp2', 
        z='comp3',
        color='cluster',
        template="plotly",
        
        # symbol = "cluster",
        
        color_discrete_sequence=px.colors.qualitative.Vivid, title=title).update_traces(
            # mode = 'markers',
            marker={
                "size": 4,
                "opacity": opacity,
                # "symbol" : "diamond",
                "line": {
                    "width": width_line,
                    "color": "black",
                }
            }
        ).update_layout(
            width = 1000, 
            height = 800, 
            autosize = False, 
            showlegend = True,
            legend=dict(
                title_font_family="Times New Roman",
                font=dict(size= 20)
            ),
            scene = dict(
                xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                yaxis=dict(title = 'comp2', titlefont_color = 'black'),
                zaxis=dict(title = 'comp3', titlefont_color = 'black')
            ),
            font = dict(
                family = "Gilroy", color  = 'black', size = 15
            )
        )
    fig.show()


def plot_pca_2d(df: pd.DataFrame, title: str = "PCA Space", opacity: float =0.8, width_line: float = 0.1):

    df = df.astype({"cluster": "object"})
    df = df.sort_values("cluster")

    fig = px.scatter(
        df, 
        x='comp1', 
        y='comp2', 
        color='cluster',
        template="plotly",
        # symbol = "cluster",
        
        color_discrete_sequence=px.colors.qualitative.Vivid, title=title).update_traces(
            # mode = 'markers',
            marker={
                "size": 8,
                "opacity": opacity,
                # "symbol" : "diamond",
                "line": {
                    "width": width_line,
                    "color": "black",
                }
            }
        ).update_layout(
            width = 800, 
            height = 700, 
            autosize = False, 
            showlegend = True,
            legend = dict(
                title_font_family="Times New Roman",
                font=dict(size= 20)
            ),
            scene = dict(
                xaxis=dict(title = 'comp1', titlefont_color = 'black'),
                yaxis=dict(title = 'comp2', titlefont_color = 'black'),
            ),
            font = dict(
                family = "Gilroy", color  = 'black', size = 15
            )
        )
    fig.show()