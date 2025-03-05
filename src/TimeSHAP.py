import numpy as np
import pandas as pd
from typing import Callable, Union, Optional
from timeshap.explainer.kernel import TimeShapKernel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from typing import Optional


def avg_rul(X_train: np.ndarray) -> np.ndarray:
    """
    Computes the average remaining useful life (RUL) from non-zero entries in the training set.
    
    Returns a 2D numpy array of shape (1, 1).
    """
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    non_zero_mask = np.any(X_train_2d != 0, axis=1)
    avg = np.mean(X_train_2d[non_zero_mask])
    return np.array([[avg]])




def get_sequence(X_test: np.ndarray, index: Optional[int] = None) -> np.ndarray:
    """
    Selects a single sequence from the test set. If no index is provided,
    a random one is chosen.

    Parameters
    ----------
    X_test : np.ndarray
        Test set array of shape [n_samples, seq_len, n_features].
    index : int, optional
        Index of the sequence to return. If None, a random index is chosen.

    Returns
    -------
    np.ndarray
        The chosen sequence with shape [1, seq_len, n_features], so it's ready
        to be fed directly into many model APIs expecting a batch dimension.
    """
    if index is None:
        index = np.random.randint(0, len(X_test))
    
    sequence = X_test[index][np.newaxis, :, :]
    print("Selected index:", index)
    print("Sequence shape:", sequence.shape)
    
    return sequence



def local_event_explainer(
    f: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    baseline: Union[pd.DataFrame, np.ndarray],
    pruned_idx: int,
    random_seed: int = 42,
    nsamples: int = 1000
) -> pd.DataFrame:
    """
    Computes event-level Shapley explanations for a given sequence.
    
    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The model function that accepts a 3-D numpy array and returns predictions.
    data : np.ndarray
        Input data with shape [1, seq_len, n_features].
    baseline : Union[pd.DataFrame, np.ndarray]
        Baseline data (e.g., an average event or sequence).
    pruned_idx : int
        The index from which to start the explanation (for pruning).
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    nsamples : int, optional
        Number of samples for the kernel estimation (default is 1000).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["Shapley Value", "Sequence Value"] for the explained events.
    """
    explainer = TimeShapKernel(f, baseline, random_seed, mode="event")
    shap_values_arr = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)
    shap_values_arr_reversed = shap_values_arr[::-1]
    sequence_values = data[0, pruned_idx:, 0]
    return pd.DataFrame({"Shapley Value": shap_values_arr_reversed, "Sequence Value": sequence_values})


def plot_local_event_explanation(event_explanation: pd.DataFrame, height: int = 600, width: int = 1000) -> go.Figure:
    """
    Generates a dual-axis Plotly plot showing event-level explanations.
    
    Parameters
    ----------
    event_explanation : pd.DataFrame
        DataFrame with columns ['Shapley Value', 'Sequence Value'].
    height : int, optional
        Height of the plot (default 600).
    width : int, optional
        Width of the plot (default 1000).
        
    Returns
    -------
    go.Figure
        A Plotly Figure with a dual-axis plot.
    """
    # If "Event Number" is not present, create it from the index.
    if "Event Number" not in event_explanation.columns:
        event_explanation = event_explanation.reset_index().rename(columns={'index': 'Event Number'})
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=event_explanation['Event Number'],
            y=event_explanation["Shapley Value"],
            mode='lines',
            name='Shapley Value'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=event_explanation['Event Number'],
            y=event_explanation["Sequence Value"],
            mode='lines',
            name='Sequence Value'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Event Explanation",
        xaxis_title="Event Number",
        height=height,
        width=width
    )
    fig.update_yaxes(title_text="Shapley Value", secondary_y=False)
    fig.update_yaxes(title_text="Sequence Value", secondary_y=True)
    
    return fig


import numpy as np
import pandas as pd
from typing import Callable, Union, Optional
from timeshap.explainer.kernel import TimeShapKernel

def compute_global_explanation(
    f: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    baseline: Union[pd.DataFrame, np.ndarray],
    random_seed: int = 42,
    nsamples: int = 1000,
    verbose: bool = False,
    absolute: bool = False
) -> pd.DataFrame:
    """
    Computes a global event-level explanation by aggregating Shapley values 
    for each event index (timestep) across all sequences in `data`.

    If `absolute` is True, the absolute values of the Shapley values 
    are taken before averaging. This can help if you're interested 
    in the magnitude of importance rather than direction.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The model function that accepts a 3-D numpy array and returns predictions.
    data : np.ndarray
        Test sequences with shape (n_samples, seq_len, 1).
    baseline : Union[pd.DataFrame, np.ndarray]
        Baseline data (e.g., an average event or sequence), 
        shape (seq_len, 1) or (1, seq_len, 1).
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    nsamples : int, optional
        Number of samples for the kernel estimation (default is 1000).
    verbose : bool, optional
        If True, prints additional debugging information.
    absolute : bool, optional
        If True, take the absolute value of all Shapley values before averaging. 
        Defaults to False.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["Event Number", "Average Shapley Value"], 
        where "Event Number" is the absolute timestep index (0-based),
        and "Average Shapley Value" is the (optionally absolute) mean 
        Shapley value across all sequences at that timestep.
    """

    event_data_list = []
    pruning_idx = 0  # Fixed pruning index (no pruning)
    
    # Ensure baseline is a NumPy array and handle its shape
    if isinstance(baseline, pd.DataFrame):
        baseline = baseline.to_numpy()
    if baseline.ndim == 3:  # shape (1, seq_len, 1)
        baseline = baseline.squeeze(0)  # become (seq_len, 1)

    # Loop over each sequence in data
    for i, sequence in enumerate(data):
        sequence = sequence.astype(np.float64)
        
        # Do NOT mask zeros: we assume all sequences are valid
        # shape: (seq_len, 1)
        # just reshape to (1, seq_len, 1) for the explainer
        L = sequence.shape[0]
        sequence_3d = sequence[np.newaxis, :, :]         # shape (1, L, 1)
        trimmed_baseline = baseline[:L].reshape(1, L, 1) # matching shape

        # Compute local event explanation
        current_event_data = local_event_explainer(
            f, sequence_3d, trimmed_baseline,
            pruning_idx, random_seed, nsamples
        )

        # Reset index => "Event Number" column
        current_event_data = current_event_data.reset_index().rename(columns={'index': 'Event Number'})
        
        if verbose:
            print(f"Sequence {i}: length = {L}, "
                  f"event indices: {current_event_data['Event Number'].tolist()}")

        event_data_list.append(current_event_data)

    # Concatenate all sequences’ data
    combined_df = pd.concat(event_data_list, ignore_index=True)

    # If absolute=True, take abs of Shapley Value before we average
    if absolute:
        combined_df['Shapley Value'] = combined_df['Shapley Value'].abs()

    # Group by Event Number (absolute index) and compute average Shapley value
    aggregated_df = combined_df.groupby('Event Number')['Shapley Value'].mean().reset_index()
    aggregated_df.rename(columns={'Shapley Value': 'Average Shapley Value'}, inplace=True)

    return aggregated_df




def plot_event_explanation(
    event_df: pd.DataFrame,
    relative: bool = False,
    plot_params: Optional[dict] = None
) -> go.Figure:
    """
    Plots the global event explanation using the aggregated event explanation DataFrame.
    
    If `relative` is False, the function creates a plot based on absolute event numbers.
    If `relative` is True, it plots the normalized (binned) event explanation.
    
    Parameters
    ----------
    event_df : pd.DataFrame
        The aggregated event explanation DataFrame. Its structure depends on the `relative` flag.
    relative : bool, optional
        If True, the DataFrame is expected to be normalized (binned) and will be plotted accordingly.
        If False, the DataFrame is expected to be absolute event explanation data.
    plot_params : dict, optional
        Additional plotting parameters.
        
        For relative = False (absolute):
            - 'height': plot height (default 600)
            - 'width': plot width (default 1000)
            - 'axis_lims': y-axis range (default None)
            - 'event_limit': minimum event number to include (default None)
        
        For relative = True:
            - 'width': plot width (default 1000)
            - 'height': plot height (default 600)
            - 'title': plot title (default 'Normalized Global Event Explanation')
    
    Returns
    -------
    go.Figure
        A Plotly Figure representing the event explanation.
    """
    if relative:
        width = plot_params.get('width', 1000) if plot_params else 1000
        height = plot_params.get('height', 600) if plot_params else 600
        title = plot_params.get('title', 'Normalized Global Event Explanation') if plot_params else 'Normalized Global Event Explanation'
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=event_df['Bin Center'],
            y=event_df['Mean Shapley Value'],
            mode='lines+markers',
            name='Mean Shapley Value',
            marker=dict(color='blue'),
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=event_df['Bin Center'],
            y=event_df['Median Shapley Value'],
            mode='lines+markers',
            name='Median Shapley Value',
            marker=dict(color='red'),
            line=dict(color='red')
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Relative Position (Bin Center)',
            yaxis_title='Shapley Value',
            width=width,
            height=height
        )
        return fig
    else:
        if plot_params is None:
            plot_params = {}
        height = plot_params.get('height', 600)
        width = plot_params.get('width', 1000)
        axis_lims = plot_params.get('axis_lims', None)
        event_limit = plot_params.get('event_limit', None)
        
        df = event_df.copy()
        if event_limit is not None:
            df = df[df['Event Number'] >= event_limit]
        df['Plot Event Number'] = df['Event Number'] + 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Plot Event Number'],
            y=df['Average Shapley Value'],
            mode='lines+markers',
            marker=dict(size=8, color="#48caaa"),
            line=dict(width=2),
            name='Average Shapley Value'
        ))
        fig.update_layout(
            title="Global Event Explanation",
            xaxis_title="Event Number",
            yaxis_title="Average Shapley Value",
            width=width,
            height=height,
            yaxis=dict(range=axis_lims)
        )
        return fig

