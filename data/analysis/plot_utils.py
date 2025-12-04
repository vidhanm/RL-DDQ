"""
Plot Utilities
Consistent styling and helper functions for visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PlotStyle:
    """Consistent plot styling configuration"""
    
    # Figure sizes
    SINGLE_PLOT: Tuple[int, int] = (10, 6)
    DOUBLE_PLOT: Tuple[int, int] = (14, 6)
    QUAD_PLOT: Tuple[int, int] = (14, 10)
    WIDE_PLOT: Tuple[int, int] = (16, 6)
    
    # Colors - colorblind friendly palette
    COLORS: Dict[str, str] = None
    
    # DPI for saving
    DPI: int = 300
    
    # Font sizes
    TITLE_SIZE: int = 14
    LABEL_SIZE: int = 12
    TICK_SIZE: int = 10
    LEGEND_SIZE: int = 10
    
    def __post_init__(self):
        if self.COLORS is None:
            # Colorblind-friendly palette (IBM Design)
            self.COLORS = {
                'dqn': '#648FFF',      # Blue
                'ddq': '#DC267F',      # Magenta
                'ddq_k2': '#FE6100',   # Orange
                'ddq_k5': '#DC267F',   # Magenta (default)
                'ddq_k10': '#785EF0',  # Purple
                'success': '#009E73',  # Green
                'failure': '#D55E00',  # Red-orange
                'neutral': '#999999',  # Gray
            }
    
    def get_color(self, key: str) -> str:
        """Get color by key, with fallback"""
        if self.COLORS is None:
            self.__post_init__()
        return self.COLORS.get(key.lower(), '#333333')


# Global style instance
STYLE = PlotStyle()


def setup_plot_style():
    """Configure matplotlib for publication-quality plots"""
    plt.rcParams.update({
        'font.size': STYLE.TICK_SIZE,
        'axes.titlesize': STYLE.TITLE_SIZE,
        'axes.labelsize': STYLE.LABEL_SIZE,
        'xtick.labelsize': STYLE.TICK_SIZE,
        'ytick.labelsize': STYLE.TICK_SIZE,
        'legend.fontsize': STYLE.LEGEND_SIZE,
        'figure.titlesize': STYLE.TITLE_SIZE + 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 100,
        'savefig.dpi': STYLE.DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def create_figure(
    nrows: int = 1, 
    ncols: int = 1,
    figsize: Tuple[int, int] = None,
    **kwargs
) -> Tuple[plt.Figure, Any]:
    """
    Create figure with consistent styling
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns  
        figsize: Figure size (auto-calculated if None)
        **kwargs: Additional arguments to plt.subplots
        
    Returns:
        (fig, axes) tuple
    """
    setup_plot_style()
    
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = STYLE.SINGLE_PLOT
        elif nrows == 1 and ncols == 2:
            figsize = STYLE.DOUBLE_PLOT
        elif nrows == 2 and ncols == 2:
            figsize = STYLE.QUAD_PLOT
        else:
            figsize = (ncols * 6, nrows * 5)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "figures",
    formats: List[str] = None
):
    """
    Save figure to file(s)
    
    Args:
        fig: Matplotlib figure
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save (default: ['png', 'pdf'])
    """
    if formats is None:
        formats = ['png']  # Default to PNG only for speed
    
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, format=fmt, dpi=STYLE.DPI, bbox_inches='tight')
        print(f"[OK] Saved: {filepath}")


def smooth_data(data: List[float], window: int = 10) -> np.ndarray:
    """Apply moving average smoothing"""
    if len(data) < window:
        return np.array(data)
    
    smoothed = np.zeros(len(data))
    for i in range(len(data)):
        start = max(0, i - window + 1)
        smoothed[i] = np.mean(data[start:i+1])
    return smoothed


def add_confidence_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha: float = 0.2,
    window: int = 10
):
    """Add confidence band around line (based on rolling std)"""
    if len(y) < window:
        return
    
    # Calculate rolling std
    rolling_std = np.zeros(len(y))
    for i in range(len(y)):
        start = max(0, i - window + 1)
        rolling_std[i] = np.std(y[start:i+1])
    
    ax.fill_between(x, y - rolling_std, y + rolling_std, color=color, alpha=alpha)


def format_percentage_axis(ax: plt.Axes, axis: str = 'y'):
    """Format axis to show percentages"""
    from matplotlib.ticker import FuncFormatter
    
    def to_percent(x, pos):
        return f'{x*100:.0f}%'
    
    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(to_percent))


def create_legend_patches(labels_colors: Dict[str, str]) -> List[mpatches.Patch]:
    """Create legend patches from label-color mapping"""
    return [mpatches.Patch(color=color, label=label) 
            for label, color in labels_colors.items()]


def add_improvement_annotation(
    ax: plt.Axes,
    x: float,
    y1: float,
    y2: float,
    text: str = None
):
    """Add annotation showing improvement between two values"""
    if text is None:
        improvement = (y2 - y1) / max(abs(y1), 0.01) * 100
        text = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
    
    mid_y = (y1 + y2) / 2
    ax.annotate(
        text,
        xy=(x, mid_y),
        fontsize=10,
        ha='left',
        va='center',
        color='green' if y2 > y1 else 'red'
    )


# Action names for visualization
ACTION_NAMES = [
    "Empathetic Listening",
    "Ask About Situation", 
    "Firm Reminder",
    "Offer Payment Plan",
    "Propose Settlement",
    "Hard Close"
]

# Persona names
PERSONA_NAMES = [
    "Angry",
    "Cooperative",
    "Sad/Overwhelmed",
    "Avoidant"
]


def get_action_color(action_idx: int) -> str:
    """Get color for action index"""
    action_colors = [
        '#4CAF50',  # Empathetic - Green
        '#2196F3',  # Ask - Blue
        '#FF9800',  # Firm - Orange
        '#9C27B0',  # Payment Plan - Purple
        '#00BCD4',  # Settlement - Cyan
        '#F44336',  # Hard Close - Red
    ]
    return action_colors[action_idx % len(action_colors)]


def get_persona_color(persona_idx: int) -> str:
    """Get color for persona index"""
    persona_colors = [
        '#F44336',  # Angry - Red
        '#4CAF50',  # Cooperative - Green
        '#2196F3',  # Sad - Blue
        '#9E9E9E',  # Avoidant - Gray
    ]
    return persona_colors[persona_idx % len(persona_colors)]
