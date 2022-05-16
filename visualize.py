import networkx as nx
import matplotlib.pyplot as plt
import torch

import create_data as cd
import helper

def filter_edge(nodelist, edge):
    return all(e in nodelist for e in [edge[0], edge[1]])

def create_plot(G, nodelist, colors, name):
    '''Takes in a Graph, color string or list and name of file to save plot in'''
    pos = nx.random_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist, node_color = colors, node_size = 100, alpha = 1)
    nx.draw_networkx_labels(G, pos, font_size=9, )
    ax = plt.gca()
    for e in G.edges(nodelist):
        if(filter_edge(nodelist, e)):
            ax.annotate("",
                        xy=pos[e[0]], xycoords='data',
                        xytext=pos[e[1]], textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="0.5",
                                        shrinkA=5, shrinkB=5,
                                        patchA=None, patchB=None,
                                        connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*0)
                                        ),
                                        ),
                        )
    plt.axis('off')
    plt.savefig('graphplots/%s.png' %(name))
    plt.close()

def evaluate_graph(G):
    A = helper.get_adjacency_matrix(G)
    demoGraph.x = torch.from_numpy(A).float()
    demoGraph.y = helper.return_labels(G)
    demoGraph.train_mask, demoGraph.val_mask, demoGraph.test_mask = helper.retrieve_masks(demoGraph.y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("models/gat_300_2").to(device)

    model.eval()
    out, accs = model(demoGraph.x, demoGraph.edge_index), []
    acc = float((out.argmax(-1) == demoGraph.y).sum() / demoGraph.y.shape[0])
    accs.append(acc)
    print("OUT:\n")
    print(out.argmax(-1))
    print('ACCS:\n')
    print(accs)
    return out.argmax(-1)

def colormap_predictions(predictions):
    colormap = []
    for i, val in enumerate(predictions):
        if val == 1:
            colormap.append('#FFAC1C')
        else:
            colormap.append('#00FFFF')
    return colormap

def get_predicted_nodes(predictions):
    nodeids = []
    for i, val in enumerate(predictions):
        if val == 1:
            nodeids.append(i)
        else:
            continue
    return nodeids



if __name__ == "__main__":

    # Create Graph to be predicted, 300 nodes
    G = nx.MultiDiGraph
    num_nodes = 300
    G_er = cd.erdosrenyi_generator(n=num_nodes, p = 3/num_nodes)
    G = cd.addedges(G_er, k=1)

    create_plot(G, list(G.nodes), '#00ffff', 'background')

    # Generate Diamond pattern to be added to background and save a plot of just the Diamond
    diamond = cd.generateDiamond(G, 19, 78, 3, 4, True)

    create_plot(diamond, list(diamond.nodes), '#00ffff', 'just-diamond')

    cd.addPattern(G, diamond)

    create_plot(G, list(G.nodes), '#00ffff', 'background-with-diamond')

    demoGraph = helper.get_data_from_graph(G)

    predictions = evaluate_graph(G)

    predictions_list = predictions.tolist()
    colormap = colormap_predictions(predictions_list)
    create_plot(G, list(G.nodes), colormap, 'highlighted-predictions')

    predicted_nodes = get_predicted_nodes(predictions_list)
    predicted_edges = G.edges(predicted_nodes)
    predicted_edgelist = []
    for e in predicted_edges:
        if filter_edge(predicted_nodes, e):
            predicted_edgelist.append(e)
        else:
            continue
    predicted_graph = nx.MultiDiGraph(predicted_edgelist)

    create_plot(predicted_graph, list(predicted_graph.nodes), '#FFAC1C', 'just-predicted-diamond')
