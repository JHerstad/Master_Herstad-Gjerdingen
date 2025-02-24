import numpy as np
import pandas as pd
from typing import Callable, Union, Optional, Tuple
from timeshap.explainer.kernel import TimeShapKernel
from Load_and_Preprocess_Aachen import preprocess_aachen_dataset
from LSTM_Model_Training import load_model_structure_and_weights, plot_predictions_vs_actual, plot_residuals
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_predictions(y_test_rescaled: np.ndarray, y_pred_rescaled: np.ndarray) -> None:
    """Plots predictions vs actual values and their residuals."""
    plot_predictions_vs_actual(y_test_rescaled, y_pred_rescaled)
    plot_residuals(y_test_rescaled, y_pred_rescaled)


def avg_rul(X_train: np.ndarray) -> np.ndarray:
    """
    Computes the average remaining useful life (RUL) from non-zero entries in the training set.
    
    Returns a 2D numpy array of shape (1, 1).
    """
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    non_zero_mask = np.any(X_train_2d != 0, axis=1)
    avg = np.mean(X_train_2d[non_zero_mask])
    return np.array([[avg]])


def get_random_sequence(X_test: np.ndarray, index: Optional[int] = None, trimmed: bool = False) -> np.ndarray:
    """
    Selects a sequence from the test set, optionally trimming out all timesteps that are entirely zeros.
    
    Parameters
    ----------
    X_test : np.ndarray
        Test set with shape [n_samples, seq_len, n_features].
    index : int, optional
        Specific index to select a sequence. If None, a random index is selected.
    trimmed : bool, optional
        If True, returns the sequence with all-zero timesteps removed.
        If False, returns the full sequence.
    
    Returns
    -------
    np.ndarray
        The selected sequence with shape [1, seq_len, n_features] (or trimmed shape if `trimmed=True`).
    """
    if index is None:
        index = np.random.randint(0, len(X_test))
    
    sequence = X_test[index][np.newaxis, :, :]
    print("Selected index:", index)
    
    if trimmed:
        mask = ~np.all(sequence[0] == 0, axis=-1)
        sequence = sequence[:, mask, :]
        print("Trimmed sequence shape:", sequence.shape)
    else:
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


def compute_event_explanation(
    f: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    baseline: Union[pd.DataFrame, np.ndarray],
    relative: bool = False,
    random_seed: int = 42,
    nsamples: int = 1000,
    num_bins: int = 20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Computes event-level explanations for all sequences.
    
    If `relative` is False, the function computes absolute event explanations
    (aggregated by event number). If `relative` is True, it computes normalized (relative)
    event explanations by:
      - Calculating each event’s relative position within its sequence.
      - Binning these relative positions.
      - Aggregating Shapley values (mean, median, count) per bin.
    
    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The model function that accepts a 3-D numpy array and returns predictions.
    data : np.ndarray
        Test sequences with shape (n_samples, seq_len, 1).
    baseline : Union[pd.DataFrame, np.ndarray]
        Baseline data (e.g., an average event or sequence), shape (seq_len, 1) or (1, seq_len, 1).
    relative : bool, optional
        If True, compute normalized (binned) event explanations.
        If False, compute absolute event explanations.
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    nsamples : int, optional
        Number of samples for the kernel estimation (default is 1000).
    num_bins : int, optional
        Number of bins for relative explanations (default is 20).
    verbose : bool, optional
        If True, prints additional debugging information.
        
    Returns
    -------
    pd.DataFrame
        - If relative is False: DataFrame with columns 'Event Number' and 'Average Shapley Value'.
        - If relative is True: DataFrame with columns 'Relative Bin', 'Mean Shapley Value',
          'Median Shapley Value', 'Count', and 'Bin Center'.
    """
    event_data_list = []
    pruning_idx = 0  # Fixed pruning index (no pruning)
    
    # Ensure baseline is a numpy array and handle its shape
    if isinstance(baseline, pd.DataFrame):
        baseline = baseline.to_numpy()
    if baseline.ndim == 2:  # Shape (288, 1)
        baseline = baseline
    elif baseline.ndim == 3:  # Shape (1, 288, 1)
        baseline = baseline.squeeze(0)  # Convert to (288, 1)

    for i, sequence in enumerate(data):
        sequence = sequence.astype(np.float64)
        mask = ~np.all(sequence == 0, axis=-1)  # Mask zero-padded time steps
        trimmed_seq = sequence[mask, :][np.newaxis, :, :]  # Shape (1, L, 1)
        
        # Trim baseline to match trimmed_seq length
        L = trimmed_seq.shape[1]  # Length of the trimmed sequence
        trimmed_baseline = baseline[:L].reshape(1, L, 1)  # Trim and reshape to (1, L, 1)

        # Compute local event explanation with trimmed baseline
        current_event_data = local_event_explainer(f, trimmed_seq, trimmed_baseline, pruning_idx, random_seed, nsamples)
        current_event_data = current_event_data.reset_index().rename(columns={'index': 'Event Number'})
        
        if relative:
            current_event_data['Relative Position'] = (current_event_data['Event Number'] / (L - 1)) if L > 1 else 0.0
            if verbose:
                print(f"Sequence {i}: original length = {sequence.shape[0]}, trimmed length = {L}, "
                      f"event indices: {current_event_data['Event Number'].tolist()}, "
                      f"relative positions: {current_event_data['Relative Position'].tolist()}")
        else:
            if verbose:
                print(f"Sequence {i}: original length = {sequence.shape[0]}, trimmed length = {trimmed_seq.shape[1]}, "
                      f"event indices: {current_event_data['Event Number'].tolist()}")
        
        event_data_list.append(current_event_data)
    
    combined_df = pd.concat(event_data_list, ignore_index=True)
    
    if relative:
        bins = np.linspace(0, 1, num_bins + 1)
        combined_df['Relative Bin'] = pd.cut(
            combined_df['Relative Position'], bins=bins, include_lowest=True, labels=False
        )
        aggregated_df = combined_df.groupby('Relative Bin')['Shapley Value'].agg(['mean', 'median', 'count']).reset_index()
        aggregated_df.rename(columns={'mean': 'Mean Shapley Value',
                                      'median': 'Median Shapley Value',
                                      'count': 'Count'}, inplace=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        aggregated_df['Bin Center'] = aggregated_df['Relative Bin'].apply(lambda x: bin_centers[int(x)])
        return aggregated_df
    else:
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


if __name__ == "__main__":
    # Global variables and loading moved to main.
    file_path: str = '/Users/sigurdgjerdingen/Student/Master kode/Master_Herstad-Gjerdingen/data/Degradation_Prediction_Dataset_ISEA.mat'
    test_cell_count: int = 3
    random_state: int = 52
    np.random.seed(random_state)
    n_samples: int = 1000
    eol_capacity: float = 0.8

    preprocessed_full = preprocess_aachen_dataset(
        file_path,
        eol_capacity=eol_capacity,
        test_cell_count=test_cell_count,
        random_state=random_state,
        phase=None,
        log_transform=False
    )

    X_train_lstm: np.ndarray = preprocessed_full["X_train"]
    X_val_lstm: np.ndarray = preprocessed_full["X_val"]
    X_test_lstm: np.ndarray = preprocessed_full["X_test"]
    y_train_norm: np.ndarray = preprocessed_full["y_train"]
    y_val_norm: np.ndarray = preprocessed_full["y_val"]
    y_test_norm: np.ndarray = preprocessed_full["y_test"]
    y_max: float = preprocessed_full["y_max"]

    model = load_model_structure_and_weights('model_20250217_092343')
    average_event: np.ndarray = avg_rul(X_train_lstm)

    # Compute event-level explanations for all sequences in the test set.
    event_data = compute_event_explanation(
        f=model.predict,
        data=X_test_lstm,
        baseline=average_event,
        random_seed=random_state,
        nsamples=n_samples,
        relative=True,
        num_bins=30,
        verbose=True
    )

    # Plot and display the global event explanation.
    global_event_chart = plot_event_explanation(event_data, relative=True)
    global_event_chart.show()