import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import tens2img, normalize_image
import cv2
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

def plot_batch(inputs, labels, classes, preds=None, fs=10, ft=5, save=None):
    
    n = inputs.shape[0]
    nrows  = np.ceil(np.sqrt(n)).astype(int)

    mpl.rcParams.update({'font.size': ft})

    _, axs = plt.subplots(nrows=nrows, ncols=nrows, figsize=(fs, fs))
    axs = axs.flatten()

    for ax in range(n):
        img = normalize_image(tens2img(inputs[ax])) 
        label = labels[ax]
        
        if preds == None:
            axs[ax].set_title(f"{classes[label]}")
        else: 
            pred = preds[ax]
            correct = (pred == label)
            color = np.array([0.0, 1.0, 0.0]) if correct else np.array([1.0, 0.0, 0.0])
            off = img.shape[2]
            img = (cv2.copyMakeBorder(img, off, off, off, off, 
                                  cv2.BORDER_CONSTANT, value=color)
                                  *255).astype(np.uint8)
            axs[ax].set_title(f"GT / PRED: {classes[label]} / {classes[pred]}")
        
        axs[ax].imshow(img) 
        axs[ax].axis("off")
        
    if save: plt.savefig(save, dpi=120)

def plot_batch_view(inputs, labels=None, classes=None, preds=None, fs=10, ft=5, save=None):
    
    n = inputs[0].shape[0]
    nrows = np.ceil(np.sqrt(n)).astype(int)

    mpl.rcParams.update({'font.size': ft})

    fig, axs = plt.subplots(nrows=nrows, ncols=nrows, figsize=(2*fs, fs))
    axs = axs.flatten()

    for ax in range(n):
        img = torch.cat((inputs[0][ax], inputs[1][ax]), 2)
        img = normalize_image(tens2img(img)) 
        if labels: label = labels[ax]
        
        if preds == None:
            if labels: axs[ax].set_title(f"{classes[label]}")
        else: 
            pred = preds[ax]
            correct = (pred == label)
            color = np.array([0.0, 1.0, 0.0]) if correct else np.array([1.0, 0.0, 0.0])
            off = img.shape[2]
            img = (cv2.copyMakeBorder(img, off, off, off, off, 
                                  cv2.BORDER_CONSTANT, value=color)
                                  *255).astype(np.uint8)
            axs[ax].set_title(f"GT / PRED: {classes[label]} / {classes[pred]}")
        
        axs[ax].imshow(img) 
        axs[ax].axis("off")
    
    if save: plt.savefig(save, dpi=300)
        
    return fig

def graph_visualize(h, color=None, epoch=None, loss=None, cmap='Set2', ax=None, labels=False):
		
	# node level
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        ax.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap=cmap)
        if epoch is not None and loss is not None:
            ax.set_xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    
	# graph level
    if ax is None:
        ax = plt.subplot()
    else: nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), ax=ax, with_labels=labels,
                         node_color=color, cmap=cmap)

def graph_rplan_visualize(g_pg, visualize_graph=True, fs=3):
    
    # set up figure
    _, axs = plt.subplots(nrows=3, ncols=4, figsize=[4*fs, 3*fs])
    axs = axs.flatten()

    # room cutouts (fig 1)
    for i, cutout in enumerate(g_pg.cutout):      # i := room instance number
        axs[i].imshow(cutout, cmap='gray')
        axs[i].set_title(f'node nr: {i}', fontsize=fs*5)

    # topology (fig 2)
    g_nx = to_networkx(g_pg, to_undirected=True)
    if visualize_graph: graph_visualize(g_nx, ax=axs[i+1])

    # axis off
    for i in range(12):
        axs[i].axis('off')

def plot_graph_rplan(G, ax, fs=3, c_node='red', c_edge='black', dw_edge=False, pos=None, node_size=None):
    
        '''
        c_node can be a list of len(G.nodes()) with different colors: 
        > list of np.array((r,g,b))) or other color way
        '''

        # figure settings
        if node_size is None: node_size = fs*100
        width_edge = fs/1.5

        # position
        if pos is None:
            pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=c_node, ax=ax)

        # edges
        if dw_edge:
            edoor = [(u, v) for (u, v, d) in G.edges(data=True) if d["door"]]
            ewall = [(u, v) for (u, v, d) in G.edges(data=True) if not d["door"]]
            # dashed line for adjacent, full line for door.
            nx.draw_networkx_edges(G, pos, edgelist=edoor, edge_color=c_edge, 
                                width=width_edge, ax=ax)
            nx.draw_networkx_edges(G, pos, edgelist=ewall, edge_color=c_edge, 
                                width=width_edge, style="dashed", ax=ax)
        else:
            nx.draw_networkx_edges(G, pos, edge_color=c_edge, 
                                width=width_edge, ax=ax)

        ax.axis('off')

def plot_polygon(ax, poly, **kwargs):
    x, y = poly.exterior.xy
    ax.plot(x, y, **kwargs)
    return 

def plot_polygons_rplan(polygons, ax, colors, face_color='black', **kwargs):
    
    room_polygons, room_types = polygons[0][0], polygons[1][0]
    door_polygons, door_types = polygons[0][1], polygons[1][1]
    
    ax.imshow(np.zeros((256, 256)), alpha=0)
    ax.axis('off')

    for key in room_polygons.keys():
        polygon = room_polygons[key]
        cls = room_types[key]
        plot_polygon(ax, polygon, c=colors[cls], **kwargs)

    for key in door_polygons.keys():
        polygon = door_polygons[key]
        cls = door_types[key]
        plot_polygon(ax, polygon, c=colors[cls], **kwargs)

    ax.set_facecolor(face_color)