from graphviz import Digraph
import matplotlib.pyplot as plt
from matplotlib import font_manager


COLOR_PINK = "#ff007f"
COLOR_VIOLET = "#9700ff"
COLOR_BLUE = "#0082ff"
COLOR_CYAN = "#00ffc3"
COLOR_GREEN = "#02fe00"


font_path = "etc/LabGrotesqueK-Regular.ttf"  # Replace with your font file path

font_manager.fontManager.addfont(font_path)
font_prop = font_manager.FontProperties(fname=font_path, size=24)

plt.rcParams["font.size"] = 24
plt.rcParams["legend.labelcolor"] = COLOR_VIOLET
plt.rcParams["font.family"] = font_prop.get_name()

print("Глобальные настройки шрифтов посеттил")
# Ну да, грязь, а что ты мне сделаешь?


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


def draw_dot(root, format="svg", rankdir="LR", layout="dot", show_grad = True, show_data=True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    dot = Digraph(
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "bgcolor": "transparent",
            "fontname": "Lab Grotesque K",
            "overlap": "false",  # Prevent node overlap
            "linelength": "60",
            "layout": layout,  # Packing mode
        },
        node_attr={"fontname": "Lab Grotesque K", "fontsize": "16"},
        edge_attr={"fontname": "Lab Grotesque K", "fontsize": "16"},
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        label = f"{n.label}"
        if show_data:
            label = label + f'| data {round(n.data,2)}'
        if show_grad and n.show_grads:
            label = label + f" | grad {round(n.grad,2)}"
        dot.node(
            name=str(id(n)),
            label=label,
            shape="record",
            fontcolor=COLOR_PINK,
            color=COLOR_VIOLET,
            penwidth="3.0",
        )
        if n._op:
            dot.node(
                name=str(id(n)) + n._op,
                label=n._op,
                fontcolor=COLOR_PINK,
                color=COLOR_VIOLET,
                penwidth="3.0",
            )
            dot.edge(str(id(n)) + n._op, str(id(n)), color=COLOR_VIOLET, penwidth="3.0")

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op, color=COLOR_VIOLET, penwidth="3.0")

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
