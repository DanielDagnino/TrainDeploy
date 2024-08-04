import matplotlib.pyplot as plt
import numpy as np


def plot_curve(x, y, label, ax, x_label="", y_label="", title="", zoom=None,
               step=False, where='post', log_scale=False):
    """Base plotting function.

    Args:
        x: x data for plot
        y: y data for plot
        label: label for this curve
        ax: matplotlib axis
        x_label: label for the x axes
        y_label: label for the y axes
        title: plot title
        zoom: None or zoom to region (x0, x1, y0, y1)
        step: Whether to use `step` plotting or `plot`.
        where: If step is True, it defines the kind of steps. Otherwise, it is not used.
        log_scale: use log scale?
    Returns:
        matplotlib.axes._subplots.AxesSubplot: subplot with curve
    """

    if ax is None:
        ax = plt.gca()

    x = np.array(x)
    y = np.array(y)

    if step:
        ax.step(x, y, label=label, linewidth=1, where=where)
        if log_scale:
            ax.set_xscale('log')
    else:
        ax.plot(x, y, label=label, linewidth=1)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])

    ax.title.set_text(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax


def plot_tpr_fpr(fpr, tpr, label, ax, title="", zoom=None, show_grid=False, log_scale=False):
    if show_grid:
        add_grid(ax, log_scale)
    return plot_curve(fpr, tpr, label, ax,
                      x_label="false positive rate",
                      y_label="true positive rate",
                      title="ROC curve (tpr vs fpr)" + title,
                      zoom=zoom, step=True, log_scale=log_scale)


def plot_pr_curve(precisions, recalls, label, ax, title="", show_grid=False):
    if show_grid:
        add_grid(ax)
    return plot_curve(recalls, precisions, label, ax,
                      x_label="recall",
                      y_label="precision",
                      title="Precision-Recall curve " + title,
                      step=True)


def add_grid(ax, log_scale=False):
    if not log_scale:
        ax.set_xlim([-0.005, 1.005])
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim([-0.005, 1.005])
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.minorticks_on()
    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
    ax.grid(visible=True, which='minor', linestyle=':', linewidth=0.5)
