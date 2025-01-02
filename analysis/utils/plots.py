from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.collections import EllipseCollection
from matplotlib.colors import Normalize
from sklearn.model_selection import learning_curve


def create_grouped_boxplots(
    df,
    x_column: str,
    y_column: str,
    hue_column: str,
    title: str,
    figsize=(10, 6),
    show_means=True,
):
    """
    Create grouped boxplots using pure matplotlib

    Args:
        df: pandas DataFrame
        x_column: column name for x-axis categories
        y_column: column name for y-axis values
        hue_column: column name for grouping
        title: plot title
        figsize: tuple of figure dimensions
        show_means: boolean to show mean markers
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique categories and groups
    categories = df[x_column].unique()
    groups = df[hue_column].unique()

    # Calculate positions for boxes
    n_groups = len(groups)
    box_width = 0.8 / n_groups
    positions = np.arange(len(categories))

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_groups))

    # Create boxplots for each group
    for i, group in enumerate(groups):
        # Get data for this group
        group_data = [
            df[(df[x_column] == cat) & (df[hue_column] == group)][y_column].values
            for cat in categories
        ]

        # Calculate positions for this group's boxes
        group_positions = positions + (i - n_groups / 2 + 0.5) * box_width

        # Create boxplot
        bp = ax.boxplot(
            group_data,
            positions=group_positions,
            widths=box_width,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=colors[i]),
            showmeans=show_means,
            meanprops=dict(
                marker="D",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
            ),
        )

        # Color boxes
        for box in bp["boxes"]:
            box.set(facecolor=colors[i], alpha=0.7)

    # Customize plot
    ax.set_xlabel(x_column, fontsize=20)
    ax.set_ylabel(y_column, fontsize=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=30)
    ax.set_yticklabels(ax.get_yticks(), fontsize=30)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7)
        for i in range(len(groups))
    ]
    ax.legend(
        legend_elements,
        groups,
        title=hue_column,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=12,
    )

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent legend cutoff
    plt.title(title, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_feature_importance(
    df: pandas.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    threshold: float = 0.002,
    pad: float = 5.0,
    title: str = "Feature Importance",
    xlabel: str = "Features",
    ylabel: str = "Importance",
    palette: Optional[List] = None,
):
    """
    Function to plot the feature importance with a distinction of importance based on a threshold.

    Parameters:
    - df: pandas.DataFrame
        DataFrame containing features and their importance scores.
    - x: str
        Name of the column representing feature names.
    - y: str
        Name of the column representing feature importance scores.
    - ax: matplotlib axis object
        Axis on which to draw the plot.
    - threshold: float, optional (default=0.002)
        Value above which bars will be colored differently.
    - pad: float, optional (default=5.0)
        Adjust the layout of the plot.
    - title: str, optional (default='Feature Importance')
        Title of the plot.
    - xlabel: str, optional (default='Features')
        Label for the x-axis.
    - ylabel: str, optional (default='Importance')
        Label for the y-axis.
    - palette: list, optional
        A list of two colors. The first color is for bars below the threshold and the second is for bars above.

    Returns:
    - None (modifies ax in-place)
    """
    if palette is None:
        palette = ["blue", "red"]

    blue, red = palette

    # Get the x-axis positions
    x_pos = range(len(df[x]))

    # Create bars with different colors based on threshold
    for i, (feature, value) in enumerate(zip(df[x], df[y])):
        color = red if value >= threshold else blue
        ax.bar(i, value, color=color, alpha=0.5)

    # Customize the plot
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df[x], rotation=0, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=15)

    # Adjust layout
    plt.tight_layout(pad=pad)


def plot_corr_ellipses(
    data: pandas.DataFrame, figsize: Tuple[int, int], **kwargs: dict
):
    """
    Plots a correlation matrix using ellipses to represent the correlations.

    Parameters:
    - data (pandas.DataFrame): A 2D array or DataFrame containing the correlation matrix.
    - figsize: Tuple specifying the figure size.
    - kwargs: Additional keyword arguments for EllipseCollection.

    Returns:
    - A tuple containing the EllipseCollection object and the Axes object.

    """
    M = np.array(data)
    if not M.ndim == 2:
        raise ValueError("Data must be a 2D array.")

    # Mask the upper triangle of the matrix
    mask = np.triu(np.ones_like(M, dtype=bool), k=1)
    M[mask] = np.nan

    # Initialize the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"aspect": "equal"})
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(-0.5, M.shape[0] - 0.5)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.grid(False)

    # Determine xy locations of each ellipse center
    xy = np.indices(M.shape)[::-1].reshape(2, -1).T

    # Define ellipse properties
    w = np.ones_like(M).ravel() + 0.01  # Widths of ellipses
    h = 1 - np.abs(M).ravel() - 0.01  # Heights of ellipses
    a = 45 * np.sign(M).ravel()  # Rotation angles

    # Create and add the ellipse collection
    ec = EllipseCollection(
        widths=w,
        heights=h,
        angles=a,
        units="x",
        offsets=xy,
        norm=Normalize(vmin=-1, vmax=1),
        transOffset=ax.transData,
        array=M.ravel(),
        **kwargs
    )
    ax.add_collection(ec)

    # Add a color bar for correlation values
    cb = fig.colorbar(ec, ax=ax, orientation="horizontal", fraction=0.047, pad=0.00)
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(top=False, labeltop=False)

    # Feature names on the diagonal
    if isinstance(data, pandas.DataFrame):
        diagonal_positions = np.arange(M.shape[1])
        for i, label in enumerate(data.columns):
            ax.annotate(
                " -  " + label, (i - 0.4, i - 1), ha="left", va="bottom", rotation=0
            )
        ax.set_yticks(diagonal_positions)
        ax.set_yticklabels(data.index)

    # Hide the plot spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ec, ax


def plot_catboost_learning_curve(model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plot learning curve for CatBoost model
    
    Parameters:
    -----------
    model : CatBoostRegressor or CatBoostClassifier
        Fitted CatBoost model
    X : array-like
        Training data
    y : array-like
        Target values
    cv : int
        Number of cross-validation folds
    train_sizes : array-like
        Points at which to evaluate training size effect
    """    
    # Calculate learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,
        scoring='r2'
    )
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, label='Cross Validation Score', marker='o')
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1)
    
    plt.xlabel('Training Instances')
    plt.ylabel('Score')
    plt.title('Learning Curve for CatBoost')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    
    return train_sizes, train_scores, val_scores