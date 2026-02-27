"""Signal visualization for IMU data."""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import COLOR_FALL, COLOR_ADL, CHART_HEIGHT
from ..data.schema import TestWindow, TestSession, PredictionResult


def plot_window(
    window: TestWindow,
    prediction: Optional[PredictionResult] = None,
    show_smv: bool = True,
    fs: float = 32.0
) -> go.Figure:
    """Plot single window's IMU signals.

    Args:
        window: TestWindow with acc and gyro data
        prediction: Optional prediction result to overlay
        show_smv: Whether to show SMV (signal magnitude vector)
        fs: Sampling frequency in Hz
    """
    n_samples = window.acc.shape[0]
    time_s = np.arange(n_samples) / fs

    n_rows = 3 if show_smv else 2
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        subplot_titles=['Accelerometer', 'Gyroscope'] + (['SMV'] if show_smv else []),
        vertical_spacing=0.08
    )

    # Accelerometer
    colors_acc = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], colors_acc)):
        fig.add_trace(go.Scatter(
            x=time_s, y=window.acc[:, i],
            mode='lines', name=f'Acc {label}',
            line=dict(color=color),
            legendgroup='acc'
        ), row=1, col=1)

    # Gyroscope
    colors_gyro = ['#d62728', '#9467bd', '#8c564b']
    for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], colors_gyro)):
        fig.add_trace(go.Scatter(
            x=time_s, y=window.gyro[:, i],
            mode='lines', name=f'Gyro {label}',
            line=dict(color=color),
            legendgroup='gyro'
        ), row=2, col=1)

    # SMV
    if show_smv:
        smv = np.linalg.norm(window.acc, axis=1)
        fig.add_trace(go.Scatter(
            x=time_s, y=smv,
            mode='lines', name='SMV',
            line=dict(color='black', width=2),
            legendgroup='smv'
        ), row=3, col=1)

    # Title with metadata
    title_parts = [
        f'UUID: {window.uuid}',
        f'Label: {window.label}'
    ]
    if prediction:
        title_parts.append(f'Prob: {prediction.probability:.3f}')
        title_parts.append(f'Pred: {prediction.predicted_label}')
        if prediction.error_type:
            title_parts.append(f'({prediction.error_type})')

    label_color = COLOR_FALL if window.label.lower() == 'fall' else COLOR_ADL

    fig.update_layout(
        title=dict(
            text=' | '.join(title_parts),
            font=dict(color=label_color)
        ),
        height=CHART_HEIGHT * 1.5,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )

    fig.update_xaxes(title_text='Time (s)', row=n_rows, col=1)
    fig.update_yaxes(title_text='m/s²', row=1, col=1)
    fig.update_yaxes(title_text='rad/s', row=2, col=1)
    if show_smv:
        fig.update_yaxes(title_text='m/s²', row=3, col=1)

    return fig


def plot_session(
    session: TestSession,
    predictions: List[PredictionResult] = None,
    fs: float = 32.0
) -> go.Figure:
    """Plot entire session with predictions overlaid.

    Args:
        session: TestSession with multiple windows
        predictions: List of predictions for each window
        fs: Sampling frequency in Hz
    """
    if not session.windows:
        return go.Figure()

    # Concatenate all windows
    all_acc = np.vstack([w.acc for w in session.windows])
    all_gyro = np.vstack([w.gyro for w in session.windows])
    n_samples = all_acc.shape[0]
    time_s = np.arange(n_samples) / fs

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=['Accelerometer', 'Gyroscope', 'Predictions'],
        vertical_spacing=0.08,
        row_heights=[0.35, 0.35, 0.3]
    )

    # Accelerometer
    smv = np.linalg.norm(all_acc, axis=1)
    fig.add_trace(go.Scatter(
        x=time_s, y=smv, mode='lines', name='SMV',
        line=dict(color='black')
    ), row=1, col=1)

    # Gyroscope magnitude
    gyro_mag = np.linalg.norm(all_gyro, axis=1)
    fig.add_trace(go.Scatter(
        x=time_s, y=gyro_mag, mode='lines', name='Gyro Mag',
        line=dict(color='purple')
    ), row=2, col=1)

    # Predictions
    if predictions:
        window_size = session.windows[0].acc.shape[0]
        for i, (window, pred) in enumerate(zip(session.windows, predictions)):
            t_start = i * window_size / fs
            t_end = (i + 1) * window_size / fs
            color = COLOR_FALL if pred.predicted_label == 'Fall' else COLOR_ADL
            opacity = 0.3 + 0.4 * pred.probability

            fig.add_shape(
                type='rect',
                x0=t_start, x1=t_end, y0=0, y1=1,
                fillcolor=color, opacity=opacity,
                line_width=0,
                row=3, col=1
            )

            fig.add_trace(go.Scatter(
                x=[(t_start + t_end) / 2],
                y=[pred.probability],
                mode='markers+text',
                marker=dict(size=8, color=color),
                text=[f'{pred.probability:.2f}'],
                textposition='top center',
                showlegend=False
            ), row=3, col=1)

    fig.update_layout(
        title=f'Session: {session.uuid} ({len(session.windows)} windows)',
        height=CHART_HEIGHT * 2,
        showlegend=True
    )

    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='m/s²', row=1, col=1)
    fig.update_yaxes(title_text='rad/s', row=2, col=1)
    fig.update_yaxes(title_text='Probability', range=[0, 1], row=3, col=1)

    return fig


def plot_comparison(
    windows: List[TestWindow],
    labels: List[str] = None,
    fs: float = 32.0
) -> go.Figure:
    """Plot multiple windows side-by-side for comparison.

    Args:
        windows: List of windows to compare
        labels: Optional labels for each window
        fs: Sampling frequency in Hz
    """
    if not windows:
        return go.Figure()

    if labels is None:
        labels = [w.label for w in windows]

    n_windows = len(windows)
    fig = make_subplots(
        rows=2, cols=n_windows,
        subplot_titles=[f'{l} (SMV)' for l in labels] + [f'{l} (Gyro)' for l in labels],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )

    n_samples = windows[0].acc.shape[0]
    time_s = np.arange(n_samples) / fs

    for i, (window, label) in enumerate(zip(windows, labels)):
        col = i + 1
        color = COLOR_FALL if window.label.lower() == 'fall' else COLOR_ADL

        # SMV
        smv = np.linalg.norm(window.acc, axis=1)
        fig.add_trace(go.Scatter(
            x=time_s, y=smv, mode='lines',
            line=dict(color=color), showlegend=False
        ), row=1, col=col)

        # Gyro magnitude
        gyro_mag = np.linalg.norm(window.gyro, axis=1)
        fig.add_trace(go.Scatter(
            x=time_s, y=gyro_mag, mode='lines',
            line=dict(color=color), showlegend=False
        ), row=2, col=col)

    fig.update_layout(
        title='Signal Comparison',
        height=CHART_HEIGHT * 1.2
    )

    return fig
