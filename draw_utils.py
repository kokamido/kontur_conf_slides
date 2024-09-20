from graphviz import Digraph
from micrograd.micrograd.engine import Value
import matplotlib.pyplot as plt


COLOR_PINK = "#ff007f"
COLOR_VIOLET = "#9700ff"


def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root, show_grads=False, format="svg", rankdir="LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        label = f"name {n.label} | data {n.data:.4}"
        if show_grads:
            label = label + f" | grad {n.grad:.4}"
        dot.node(name=str(id(n)), label=label, shape="record")
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def fix_axes_style(ax: plt.Axes) -> plt.Axes:
    for spine in ax.spines.values():
        spine.set_color(COLOR_VIOLET)
        spine.set_linewidth(5)
    ax.xaxis.label.set_color(COLOR_VIOLET)
    ax.yaxis.label.set_color(COLOR_VIOLET)
    ax.tick_params(axis="x", colors=COLOR_VIOLET)
    ax.tick_params(axis="y", colors=COLOR_VIOLET)
    return ax
