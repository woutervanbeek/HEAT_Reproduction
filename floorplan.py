# general imports
import os
import numpy as np

# graph processing
import networkx as nx

# plotting
import matplotlib.pyplot as plt

# own
from rplan_process.structurize import extract_polygons_from_image, extract_full_graph_from_polygons
from utils import COLORS_ORDERED, colorize_floorplan
from plot import plot_polygon

# Class to encapsulate a floorplan [vectorize, extract graph, render it]
class FloorPlan():

    def __init__(self, id, rplan_folder):
        '''
        id: floor plan identity (in file name of original dataset)
        mode: color mode (colored 'color' or grayscale 'bw')
        rplan_folder: path the rplan folder
        '''

        # original image (from image path)
        self.image_path = os.path.join(rplan_folder, 'original', f'{id}.png')
        self.image = (255*plt.imread(self.image_path)).astype(int)[..., 1] # room type as 2d array

        # room, wall, and door classes
        self.room_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.door_classes = [17]
        self.wall_classes = [12, 14, 15, 16, 17]

        # coloring
        self.colors = COLORS_ORDERED

        # to be cut image
        self.image_color = colorize_floorplan(self.image, COLORS_ORDERED)

        # polygons
        self.polygons = self.extract_polygons()

        # full and access graph
        self.graph = self.extract_graph()
        self.access_graph = nx.Graph()
        self.access_graph.add_edges_from([(u, v) for u, v, d in self.graph.edges(data=True) if d["door"]])

    
    def extract_polygons(self, min_area=10):
        '''
        Extracts all polygons from the original image file.
        Output: list of multipolygons (can be more of the same room in)
        '''
        return extract_polygons_from_image(self.image, self.room_classes, self.door_classes, self.wall_classes, min_area)
        

    def extract_graph(self):
        '''
        Extracts the full grpah from the polygon lists.
        Output: full graph ('color': for plotting, 'category':room type, 
            'centroid': for positional plotting, 'polygon': geometrical attributes)
        '''
        return extract_full_graph_from_polygons(self.polygons, self.image_color, self.colors)

    def plot_polygons_rplan(self, ax, face_color='black', **kwargs):
    
        room_polygons, room_types = self.polygons[0][0], self.polygons[1][0]
        door_polygons, door_types = self.polygons[0][1], self.polygons[1][1]
        
        ax.imshow(self.image_color, alpha=0)
        ax.axis('off')

        for key in room_polygons.keys():
            polygon = room_polygons[key]
            cls = room_types[key]
            plot_polygon(ax, polygon, c = self.colors[cls], **kwargs)

        for key in door_polygons.keys():
            polygon = door_polygons[key]
            cls = door_types[key]
            plot_polygon(ax, polygon, c = self.colors[cls], **kwargs)

        ax.set_facecolor(face_color)

    def plot_graph_rplan(self, ax, fs=3, c_node='red', c_edge='black', dw_edge=False, pos=None):
    
        '''
        c_node can be a list of len(G.nodes()) with different colors: 
        > list of np.array((r,g,b))) or other color way
        '''
        G = self.graph

        # figure settings
        node_size = fs*100
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

