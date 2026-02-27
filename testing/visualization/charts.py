"""Plotly charts for metrics visualization."""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..config import COLOR_FALL, COLOR_ADL, CHART_HEIGHT


def confusion_matrix_chart(
    cm: np.ndarray,
    labels: List[str] = None
) -> go.Figure:
    """Create confusion matrix heatmap."""
    if labels is None:
        labels = ['ADL', 'Fall']

    # Compute percentages
    cm_pct = cm.astype(float) / cm.sum() * 100

    # Text annotations
    text = [[f'{cm[i,j]}<br>({cm_pct[i,j]:.1f}%)' for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=False
    ))

    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=CHART_HEIGHT,
        yaxis=dict(autorange='reversed')
    )
    return fig


def probability_histogram(
    fall_probs: List[float],
    adl_probs: List[float],
    threshold: float = 0.5,
    bins: int = 30
) -> go.Figure:
    """Create probability distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=adl_probs,
        name='ADL',
        marker_color=COLOR_ADL,
        opacity=0.7,
        nbinsx=bins
    ))

    fig.add_trace(go.Histogram(
        x=fall_probs,
        name='Fall',
        marker_color=COLOR_FALL,
        opacity=0.7,
        nbinsx=bins
    ))

    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash='dash',
        line_color='black',
        annotation_text=f'Threshold: {threshold}'
    )

    fig.update_layout(
        title='Probability Distribution by Class',
        xaxis_title='Probability',
        yaxis_title='Count',
        barmode='overlay',
        height=CHART_HEIGHT,
        legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99)
    )
    return fig


def roc_curve_chart(
    fpr: List[float],
    tpr: List[float],
    auc: float
) -> go.Figure:
    """Create ROC curve plot."""
    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc:.3f})',
        line=dict(color='darkorange', width=2)
    ))

    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='navy', width=2, dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=CHART_HEIGHT,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig


def pr_curve_chart(
    precision: List[float],
    recall: List[float],
    auc: float
) -> go.Figure:
    """Create precision-recall curve plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR (AUC = {auc:.3f})',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=CHART_HEIGHT,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig


def metrics_bar_chart(
    configs: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> go.Figure:
    """Create bar chart comparing metrics across configs."""
    if metrics is None:
        metrics = ['f1', 'precision', 'recall', 'accuracy']

    fig = go.Figure()

    for metric in metrics:
        values = [configs[c].get(metric, 0) for c in configs]
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=list(configs.keys()),
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))

    fig.update_layout(
        title='Metrics Comparison',
        xaxis_title='Configuration',
        yaxis_title='Score',
        barmode='group',
        height=CHART_HEIGHT,
        yaxis=dict(range=[0, 1])
    )
    return fig


def threshold_sweep_chart(
    sweep_data: Dict[float, Dict[str, float]],
    metrics: List[str] = None
) -> go.Figure:
    """Create threshold sweep plot."""
    if metrics is None:
        metrics = ['f1', 'precision', 'recall', 'specificity']

    thresholds = sorted(sweep_data.keys())

    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, metric in enumerate(metrics):
        values = [sweep_data[t].get(metric, 0) for t in thresholds]
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=values,
            mode='lines+markers',
            name=metric.capitalize(),
            line=dict(color=colors[i % len(colors)])
        ))

    # Find optimal F1 threshold
    if 'f1' in metrics:
        f1_values = [sweep_data[t].get('f1', 0) for t in thresholds]
        best_idx = np.argmax(f1_values)
        best_thresh = thresholds[best_idx]
        fig.add_vline(
            x=best_thresh,
            line_dash='dash',
            line_color='gray',
            annotation_text=f'Best F1: {best_thresh:.2f}'
        )

    fig.update_layout(
        title='Metrics vs Threshold',
        xaxis_title='Threshold',
        yaxis_title='Score',
        height=CHART_HEIGHT,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    return fig


def metrics_radar_chart(
    configs: Dict[str, Dict[str, float]],
    metrics: List[str] = None
) -> go.Figure:
    """Create radar chart for multi-metric comparison."""
    if metrics is None:
        metrics = ['f1', 'precision', 'recall', 'specificity', 'accuracy']

    fig = go.Figure()

    colors = px.colors.qualitative.Set1
    for i, (name, data) in enumerate(configs.items()):
        values = [data.get(m, 0) for m in metrics]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            name=name,
            line=dict(color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title='Multi-Metric Comparison',
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=CHART_HEIGHT
    )
    return fig
