import os

import graphviz
from matplotlib import pyplot as plt

from go_ai.measurements import matplot_format


def plot_tree(go_env, policy, outdir, state_info=None):
    go_env.reset()
    if state_info is not None:
        for actions in state_info:
            for i, a in enumerate(actions):
                go_env.step(a)
                if i < len(actions) - 1:
                    go_env.step(None)

    _, _, root = policy(go_env, debug=True)
    imgdir = os.path.join(outdir, 'node_imgs/')
    imgdir = os.path.abspath(imgdir)
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    for engine in ['twopi', 'dot']:
        graph = get_graph(root, imgdir, engine)
        graph.render(os.path.join(outdir, f'tree_{engine}'))


def get_graph(treenode, imgdir, engine):
    graph = graphviz.Digraph('MCT', engine=engine, node_attr={'shape': 'none'}, graph_attr={'ranksep': "8"},
                             format='pdf')
    graph.attr(overlap='false')
    register_nodes(treenode, graph, imgdir)
    register_edges(treenode, graph)
    return graph


def register_nodes(treenode, graph, imgdir):
    plt.figure()
    plt.axis('off')
    plt.title(str(treenode))
    plt.imshow(matplot_format(treenode.state))
    imgpath = os.path.join(imgdir, f'{str(id(treenode))}.jpg')
    plt.savefig(imgpath, bbox_inches='tight')
    plt.close()
    graph.node(str(id(treenode)), image=imgpath, label='')
    for child in treenode.canon_children:
        if child is not None:
            register_nodes(child, graph, imgdir)


def register_edges(treenode, graph):
    for a in range(treenode.actionsize()):
        child = treenode.canon_children[a]
        if child is not None:
            label = ''
            if treenode.prior_pi is not None:
                label = f'{treenode.prior_pi[a]:.2f}'
            graph.edge(str(id(treenode)), str(id(child)), label=label)
            register_edges(child, graph)