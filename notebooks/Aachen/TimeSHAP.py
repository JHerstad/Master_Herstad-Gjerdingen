import numpy as np
import pandas as pd
from typing import Callable, Union, Optional, Tuple
from timeshap.explainer.kernel import TimeShapKernel
from timeshap.explainer.pruning import prune_all, local_pruning  # (if needed)
from Load_and_Preprocess_Aachen import preprocess_aachen_dataset
from LSTM_Model_Training import load_model_structure_and_weights, plot_predictions_vs_actual, plot_residuals
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  # (if needed)
# Note: You can remove unused imports if they’re not needed

# Define global variables
file_path: str = '/Users/sigurdgjerdingen/Student/Master kode/Master_Herstad-Gjerdingen/data/Degradation_Prediction_Dataset_ISEA.mat'
test_cell_count: int = 3
random_state: int = 52
np.random.seed(random_state)
n_samples: int = 1000
eol_capacity: float = 0.8

# Load and preprocess Aachen data
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

# Load model
model = load_model_structure_and_weights('model_20250217_092343')


def evaluate_model(model, X_test: np.ndarray, y_test_norm: np.ndarray, y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates the model on the test set and rescales the mean absolute error (MAE)."""
    test_loss, test_mae = model.evaluate(X_test, y_test_norm, verbose=1)
    test_mae_rescaled = test_mae * y_max
    print(f"Rescaled Test MAE: {test_mae_rescaled}")
    y_pred = model.predict(X_test)
    y_pred_rescaled = y_pred.flatten() * y_max
    y_test_rescaled = y_test_norm * y_max
    return y_pred_rescaled, y_test_rescaled


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


average_event: np.ndarray = avg_rul(X_train_lstm)


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
    
    # Select the sequence and add a batch dimension
    sequence = X_test[index][np.newaxis, :, :]
    print("Selected index:", index)
    
    if trimmed:
        # Create a mask for timesteps that are not all zeros
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
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add the Shapley Value trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=event_explanation.index,
            y=event_explanation["Shapley Value"],
            mode='lines',
            name='Shapley Value'
        ),
        secondary_y=False
    )
    
    # Add the Sequence Value trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=event_explanation.index,
            y=event_explanation["Sequence Value"],
            mode='lines',
            name='Sequence Value'
        ),
        secondary_y=True
    )
    
    # Update layout and axis titles
    fig.update_layout(
        title="Event Explanation",
        xaxis_title="Event Number",
        height=height,
        width=width
    )
    fig.update_yaxes(title_text="Shapley Value", secondary_y=False)
    fig.update_yaxes(title_text="Sequence Value", secondary_y=True)
    
    return fig


def event_explain_all(
    f: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    baseline: Union[pd.DataFrame, np.ndarray],
    random_seed: int = 42,
    nsamples: int = 1000,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculates event-level explanations for all sequences in the provided numpy array,
    after trimming out any zero-padded timesteps.
    
    Assumes that:
      - `data` is a NumPy array of shape (n_samples, seq_len, 1)
      - The model uses only one feature.
      - A fixed pruning index of 0 is used (i.e. no pruning).
    
    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The model function that accepts a 3-D numpy array (n_samples, seq_len, 1)
        and returns predictions.
    data : np.ndarray
        Test sequences with shape (n_samples, seq_len, 1).
    baseline : Union[pd.DataFrame, np.ndarray]
        Baseline data (e.g., an average event or sequence).
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    nsamples : int, optional
        Number of samples for the kernel estimation (default is 1000).
    verbose : bool, optional
        If True, prints additional debugging information.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated event-level explanations.
        The columns are:
          - 'Event Number': Numeric index of the event (time step)
          - 'Average Shapley Value': Averaged Shapley value for that event index across sequences.
    """
    event_data_list = []
    pruning_idx = 0  # Fixed pruning index (no pruning)

    # Loop over each sequence in the data array
    for i, sequence in enumerate(data):
        sequence = sequence.astype(np.float64)
        # Create mask for timesteps that are not all zeros
        mask = ~np.all(sequence == 0, axis=-1)
        trimmed_seq = sequence[mask, :][np.newaxis, :, :]  # add batch dimension
        
        # Compute the event-level explanation for the current trimmed sequence.
        current_event_data = local_event_explainer(f, trimmed_seq, baseline, pruning_idx, random_seed, nsamples)
        current_event_data = current_event_data.reset_index().rename(columns={'index': 'Event Number'})
        
        if verbose:
            print(f"Sequence {i}: original length = {sequence.shape[0]}, trimmed length = {trimmed_seq.shape[1]}, "
                  f"event indices: {current_event_data['Event Number'].tolist()}")
        
        event_data_list.append(current_event_data)
    
    combined_df = pd.concat(event_data_list, ignore_index=True)
    
    # Group by 'Event Number' and compute the mean of the Shapley values
    aggregated_df = combined_df.groupby('Event Number')['Shapley Value'].mean().reset_index()
    aggregated_df.rename(columns={'Shapley Value': 'Average Shapley Value'}, inplace=True)
    
    return aggregated_df


def plot_global_event(aggregated_df: pd.DataFrame, plot_parameters: Optional[dict] = None) -> go.Figure:
    """
    Plots global event explanations using Plotly.

    Parameters
    ----------
    aggregated_df : pd.DataFrame
        DataFrame with columns ['Event Number', 'Average Shapley Value'].
    plot_parameters : dict, optional
        Dictionary with optional plot parameters:
            - 'height': height of the plot (default 600)
            - 'width': width of the plot (default 1000)
            - 'axis_lims': y-axis domain for Average Shapley Value (default None)
            - 'event_limit': minimum event number to include (default: None, include all)

    Returns
    -------
    go.Figure
        A Plotly Figure representing the global event explanations.
    """
    # Use DataFrame's own copy method to avoid modifying the original DataFrame
    data = aggregated_df.copy()

    if plot_parameters is None:
        plot_parameters = {}
    height = plot_parameters.get('height', 600)
    width = plot_parameters.get('width', 1000)
    axis_lims = plot_parameters.get('axis_lims', None)
    event_limit = plot_parameters.get('event_limit', None)

    if event_limit is not None:
        data = data[data['Event Number'] >= event_limit]

    # For plotting, add 1 to the event number (e.g., event 0 becomes 1)
    data['Plot Event Number'] = data['Event Number'] + 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Plot Event Number'],
        y=data['Average Shapley Value'],
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


def plot_normalized_explanation(aggregated_df: pd.DataFrame,
                                width: int = 500,
                                height: int = 300,
                                title: str = 'Normalized Event Explanation') -> go.Figure:
    """
    Plots the normalized event-level explanations using Plotly.
    
    The DataFrame is expected to have the following columns:
      - 'Bin Center': The center of the relative bin (normalized position between 0 and 1).
      - 'Mean Shapley Value': The mean Shapley value for events in that bin.
      - 'Median Shapley Value': The median Shapley value for events in that bin.
      - 'Count': The number of events in that bin (optional for tooltip).
    
    Parameters
    ----------
    aggregated_df : pd.DataFrame
        Aggregated DataFrame with normalized event explanations.
    width : int, optional
        Width of the plot (default is 500).
    height : int, optional
        Height of the plot (default is 300).
    title : str, optional
        Title of the plot (default is 'Normalized Event Explanation').
        
    Returns
    -------
    go.Figure
        A Plotly Figure representing the normalized event-level explanations.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=aggregated_df['Bin Center'],
        y=aggregated_df['Mean Shapley Value'],
        mode='lines+markers',
        name='Mean Shapley Value',
        marker=dict(color='blue'),
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=aggregated_df['Bin Center'],
        y=aggregated_df['Median Shapley Value'],
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


def event_explain_all_normalized(
    f: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    baseline: Union[pd.DataFrame, np.ndarray],
    random_seed: int = 42,
    nsamples: int = 1000,
    num_bins: int = 20,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Calculates event-level explanations for all sequences, aggregates them based on their relative
    position within each trimmed sequence, and bins the results.
    
    The relative position is computed as:
        Relative Position = (Event Index) / (L - 1)
    where L is the length of the trimmed sequence.
    
    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The model function that accepts a 3-D numpy array and returns predictions.
    data : np.ndarray
        Test sequences with shape (n_samples, seq_len, 1).
    baseline : Union[pd.DataFrame, np.ndarray]
        Baseline data (e.g., an average event or sequence).
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    nsamples : int, optional
        Number of samples for the kernel estimation (default is 1000).
    num_bins : int, optional
        Number of bins to use when aggregating relative positions (default is 20).
    verbose : bool, optional
        If True, prints additional debugging information.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with aggregated event-level explanations on a relative scale.
        The columns include:
            - 'Relative Bin': The bin index for the normalized event position.
            - 'Mean Shapley Value': Mean Shapley value for events in that bin.
            - 'Median Shapley Value': Median Shapley value for events in that bin.
            - 'Count': Number of events in that bin.
            - 'Bin Center': The center of the relative bin (for plotting).
    """
    event_data_list = []
    pruning_idx = 0  # Fixed, no pruning

    for i, sequence in enumerate(data):
        sequence = sequence.astype(np.float64)
        mask = ~np.all(sequence == 0, axis=-1)
        trimmed_seq = sequence[mask, :][np.newaxis, :, :]  # add batch dimension
        
        current_event_data = local_event_explainer(f, trimmed_seq, baseline, pruning_idx, random_seed, nsamples)
        current_event_data = current_event_data.reset_index().rename(columns={'index': 'Event Number'})
        
        L = trimmed_seq.shape[1]
        current_event_data['Relative Position'] = current_event_data['Event Number'] / (L - 1) if L > 1 else 0.0
        
        if verbose:
            print(f"Sequence {i}: original length = {sequence.shape[0]}, trimmed length = {L}, "
                  f"event indices: {current_event_data['Event Number'].tolist()}, "
                  f"relative positions: {current_event_data['Relative Position'].tolist()}")
        
        event_data_list.append(current_event_data)
    
    combined_df = pd.concat(event_data_list, ignore_index=True)
    
    # Bin the relative positions into 'num_bins' equal bins in the range [0, 1]
    bins = np.linspace(0, 1, num_bins + 1)
    combined_df['Relative Bin'] = pd.cut(combined_df['Relative Position'], bins=bins, include_lowest=True, labels=False)
    
    aggregated_df = combined_df.groupby('Relative Bin')['Shapley Value'].agg(['mean', 'median', 'count']).reset_index()
    aggregated_df.rename(columns={'mean': 'Mean Shapley Value',
                                  'median': 'Median Shapley Value',
                                  'count': 'Count'}, inplace=True)
    
    # Compute the center of each bin for plotting purposes.
    bin_centers = (bins[:-1] + bins[1:]) / 2
    aggregated_df['Bin Center'] = aggregated_df['Relative Bin'].apply(lambda x: bin_centers[int(x)])
    
    return aggregated_df

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
        Baseline data (e.g., an average event or sequence).
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
    
    for i, sequence in enumerate(data):
        sequence = sequence.astype(np.float64)
        # Trim out timesteps where all features are zero.
        mask = ~np.all(sequence == 0, axis=-1)
        trimmed_seq = sequence[mask, :][np.newaxis, :, :]
        
        # Compute event-level explanation for the trimmed sequence.
        # (Assumes that the helper function `local_event_explainer` is defined.)
        current_event_data = local_event_explainer(f, trimmed_seq, baseline, pruning_idx, random_seed, nsamples)
        current_event_data = current_event_data.reset_index().rename(columns={'index': 'Event Number'})
        
        if relative:
            L = trimmed_seq.shape[1]
            # Compute the relative position of each event within the sequence.
            current_event_data['Relative Position'] = (current_event_data['Event Number'] /
                                                       (L - 1)) if L > 1 else 0.0
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
        # Bin the relative positions into num_bins equal bins in [0, 1]
        bins = np.linspace(0, 1, num_bins + 1)
        combined_df['Relative Bin'] = pd.cut(
            combined_df['Relative Position'], bins=bins, include_lowest=True, labels=False
        )
        # Aggregate the Shapley values for each bin.
        aggregated_df = combined_df.groupby('Relative Bin')['Shapley Value'].agg(['mean', 'median', 'count']).reset_index()
        aggregated_df.rename(columns={'mean': 'Mean Shapley Value',
                                      'median': 'Median Shapley Value',
                                      'count': 'Count'}, inplace=True)
        # Calculate bin centers for plotting.
        bin_centers = (bins[:-1] + bins[1:]) / 2
        aggregated_df['Bin Center'] = aggregated_df['Relative Bin'].apply(lambda x: bin_centers[int(x)])
        return aggregated_df
    else:
        # Aggregate by absolute event number.
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
        # Absolute explanation plot.
        if plot_params is None:
            plot_params = {}
        height = plot_params.get('height', 600)
        width = plot_params.get('width', 1000)
        axis_lims = plot_params.get('axis_lims', None)
        event_limit = plot_params.get('event_limit', None)
        
        # Optionally filter out events.
        df = event_df.copy()
        if event_limit is not None:
            df = df[df['Event Number'] >= event_limit]
        # Offset event numbers by 1 for plotting.
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
    # Instead of wrapping model.predict in a lambda, we can pass it directly.
    f: Callable[[np.ndarray], np.ndarray] = model.predict

    # Compute event-level explanations for all sequences in the test set.
    event_data = event_explain_all(f, X_test_lstm, baseline=average_event, random_seed=random_state, nsamples=n_samples, verbose=True)
    
    # Plot and display the global event explanation.
    global_event_chart = plot_global_event(event_data)
    global_event_chart.show()