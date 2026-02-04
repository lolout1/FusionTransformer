"""Visualization utilities for test results."""

from __future__ import annotations
from typing import Optional, List, Dict
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..data.schema import TestWindow, WindowPrediction, TestResults


class Visualizer:
    """Generate visualizations for test results."""

    @staticmethod
    def confusion_matrix_heatmap(cm: list[list[int]], title: str = "Confusion Matrix"):
        """Create confusion matrix heatmap."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        labels = ['ADL (0)', 'Fall (1)']
        cm_array = np.array(cm)

        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm_array,
            texttemplate='%{text}',
            textfont={'size': 20},
            hoverongaps=False,
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=400,
            height=400,
        )

        return fig

    @staticmethod
    def probability_histogram(
        fall_probs: list[float],
        adl_probs: list[float],
        threshold: float = 0.5,
        title: str = "Probability Distribution"
    ):
        """Create histogram of probabilities for falls vs ADLs."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=adl_probs,
            name='ADL',
            opacity=0.7,
            nbinsx=30,
            marker_color='blue',
        ))

        fig.add_trace(go.Histogram(
            x=fall_probs,
            name='Fall',
            opacity=0.7,
            nbinsx=30,
            marker_color='red',
        ))

        # Add threshold line
        fig.add_vline(
            x=threshold,
            line_dash='dash',
            line_color='green',
            annotation_text=f'Threshold={threshold}',
        )

        fig.update_layout(
            title=title,
            xaxis_title='Probability',
            yaxis_title='Count',
            barmode='overlay',
            width=600,
            height=400,
        )

        return fig

    @staticmethod
    def signal_plot(
        window: TestWindow,
        prediction: Optional[WindowPrediction] = None,
        features: Optional[np.ndarray] = None,
        title: str = "IMU Signals"
    ):
        """Plot raw IMU signals with optional Kalman features overlay."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        t = np.arange(128)
        rows = 3 if features is None else 4

        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=['Accelerometer', 'Gyroscope', 'SMV'] + (['Kalman Features'] if features is not None else []),
            shared_xaxes=True,
        )

        # Accelerometer
        for i, label in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(go.Scatter(x=t, y=window.acc[:, i], name=f'Acc {label}', mode='lines'), row=1, col=1)

        # Gyroscope
        for i, label in enumerate(['X', 'Y', 'Z']):
            fig.add_trace(go.Scatter(x=t, y=window.gyro[:, i], name=f'Gyro {label}', mode='lines'), row=2, col=1)

        # SMV
        smv = np.linalg.norm(window.acc, axis=1)
        fig.add_trace(go.Scatter(x=t, y=smv, name='SMV', mode='lines', line=dict(color='purple')), row=3, col=1)

        # Kalman features if provided
        if features is not None and features.shape[1] >= 7:
            for i, label in enumerate(['Roll', 'Pitch', 'Yaw/GyroMag']):
                if i + 4 < features.shape[1]:
                    fig.add_trace(
                        go.Scatter(x=t, y=features[:, i + 4], name=label, mode='lines'),
                        row=4, col=1
                    )

        # Add prediction info
        pred_text = ""
        if prediction:
            pred_text = f"Prob: {prediction.probability:.3f}"
        label_text = f"Label: {window.label}"

        fig.update_layout(
            title=f"{title} | {label_text} | {pred_text}",
            height=200 * rows,
            width=800,
            showlegend=True,
        )

        return fig

    @staticmethod
    def model_comparison_bar(comparison: dict, metric: str = 'f1'):
        """Create bar chart comparing models on a metric."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        configs = comparison['configs']
        window_values = comparison[f'window_{metric}']
        session_values = comparison[f'session_{metric}']

        fig = go.Figure(data=[
            go.Bar(name='Window-level', x=configs, y=window_values),
            go.Bar(name='Session-level', x=configs, y=session_values),
        ])

        fig.update_layout(
            title=f'{metric.upper()} Comparison',
            xaxis_title='Config',
            yaxis_title=metric.upper(),
            barmode='group',
            width=800,
            height=400,
        )

        return fig

    @staticmethod
    def roc_curve(roc_data: dict, title: str = "ROC Curve"):
        """Plot ROC curve."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=roc_data['fpr'],
            y=roc_data['tpr'],
            mode='lines',
            name='ROC',
            line=dict(color='blue'),
        ))

        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', dash='dash'),
        ))

        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=500,
            height=500,
        )

        return fig

    @staticmethod
    def threshold_sweep_plot(sweep_results: dict):
        """Plot metrics vs threshold."""
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        thresholds = list(sweep_results.keys())
        f1_scores = [sweep_results[t]['f1'] for t in thresholds]
        precisions = [sweep_results[t]['precision'] for t in thresholds]
        recalls = [sweep_results[t]['recall'] for t in thresholds]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, name='F1', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, name='Precision', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, name='Recall', mode='lines+markers'))

        fig.update_layout(
            title='Metrics vs Threshold',
            xaxis_title='Threshold',
            yaxis_title='Score',
            width=700,
            height=400,
        )

        return fig
