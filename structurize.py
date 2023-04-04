# general imports
import numpy as np
from collections import defaultdict

# graph processing
import networkx as nx

# geometry processing
from shapely import geometry
from shapely.geometry import Polygon

import torch

# image processing
from rasterio import features
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

from utils_rplan import COLORS_ORDERED

# settings
FOLDER = "C:\\Users\\caspervanengel\\OneDrive\\Documents\\PHD\\1_data\\rplan\\0-full"

# ///// GEOMETRY [POLYGONS]
def valid_poly(poly):
    poly_mapped = geometry.mapping(poly)
    if len(poly_mapped['coordinates']) != 1:
        poly_image = Polygon(((0, 0), (0, 256), (256, 256), (256, 0), (0, 0)))
        for subpoly_mapped in poly_mapped['coordinates']:
            subpoly = Polygon(subpoly_mapped)
            if subpoly == poly_image: return False # if polygon is the whole image
            else: return True
    else: return True

def get_image_mask_from_polygon(polygon, out_shape=(256, 256)):

    # // mask from polygon + dilate to "see" a bit further per room
    mask = features.rasterize([polygon], out_shape=out_shape)
    struct = generate_binary_structure(2, 2)

    return binary_dilation(mask, iterations=10, structure=struct).astype(int)

def polygons_from_image(image, classes, wall=False, min_area=10):
    
    polygons = {}
    types = {}
    room_id = 0

    # iterate over room classes
    for cls in classes:
        # add type to list

        # get mask
        if wall: # sum up all wall 
            mask = np.zeros_like(image).astype(np.int16)
            for w_cls in classes:
                mask += (image == w_cls).astype(np.int16)
        else: mask = (image == cls).astype(np.int16)

        # get geometry of every individual room
        shapes = features.shapes(mask, connectivity=4)

        # can be multipe shapes in shapes
        for shape, _ in shapes:

            # get polygon and add to list
            polygon = Polygon(geometry.shape(shape))
            if valid_poly(polygon):
                if 250*250 > polygon.area > min_area:
                    polygons[room_id] = polygon
                    types[room_id] = cls
                    room_id += 1

        
        if wall: break

    return polygons, types

def extract_polygons_from_image(image, room_classes, door_classes, wall_classes, min_area):

    room_polygons, room_types = polygons_from_image(image, room_classes, min_area=min_area)
    door_polygons, door_types = polygons_from_image(image, door_classes, min_area=min_area)
    wall_polygons, wall_types = polygons_from_image(image, wall_classes, min_area=min_area, wall=True) 

    return [room_polygons, door_polygons, wall_polygons], [room_types, door_types, wall_types] # polygons; types


# ///// TOPOLOGY [ACCESS/ADJACENCY GRAPHS]
def extract_full_graph_from_polygons(polygons, img, colors=COLORS_ORDERED):

    room_polygons, room_types = polygons[0][0], polygons[1][0]
    door_polygons, _ = polygons[0][1], polygons[1][1]
    
    # // NODES
    room_properties = {}
    for key in room_polygons.keys():
        poly = room_polygons[key]
        mask = get_image_mask_from_polygon(poly, out_shape=(img.shape[0], img.shape[1]))
        room_properties[key] = {
            'category': torch.tensor(room_types[key]),
            # 'polygon': poly,
            'cutout': torch.tensor(img*np.stack([mask]*3, axis=2)),
            'color': torch.tensor(colors[room_types[key]]),
            'centroid': torch.tensor(np.array([poly.centroid.x, poly.centroid.y]))
        }

    # /// graph definition and nodes to graph
    G = nx.Graph()
    G.add_nodes_from([(u, v) for u, v in room_properties.items()]) #u: node, v: property / attribute
    

    # // EDGES
    # /// connectivity
    door_edges = defaultdict(lambda: [])
    for i, key in enumerate(door_polygons.keys()):
        door = door_polygons[key]

        # loop over nodes
        for room_id in G.nodes:
            room = room_polygons[room_id]

            if door.intersection(room).length:
                door_edges[i].append(room_id)

    for i in range(len(door_polygons)):
        assert len(door_edges[
                    i]) == 2, f"Expected door to be overlapping with exactly 2 rooms, but was: {len(door_edges[i])}"

    # /// adjacency 
    '''
    protocol:
    > check if in 
    > dilate room a and room b (every combination) with >>>4<<< (max clearance = 6)
    > overlap must exceed 20! (this means for a minimum clearance of 3:
            the overlap for two diagonally touching division is about 16
    therefore: min overlap for max clearance (=6) = 2*length wall between two adjacent rooms (should be at least 8 pixels)
    '''

    adj_edges = defaultdict(lambda: [])
    i = 0 # amount of adjacency edges
    for u in range(len(room_polygons)):
        for v in range(u+1, len(room_polygons)):
            if [u, v] in door_edges.values():
                continue
            rm1 = room_polygons[u]
            rm2 = room_polygons[v]

            rm1 = rm1.buffer(4, join_style=geometry.JOIN_STYLE.mitre)
            rm2 = rm2.buffer(4, join_style=geometry.JOIN_STYLE.mitre)

            if rm1.intersection(rm2).area > 16: 
                adj_edges[i] = [u, v]
                i += 1
            else: continue

    # edges to one dictionary
    edges = []
    for edge in door_edges.values():
        u, v = edge[0], edge[1]
        edges.append([u, v, {'door': True}])
    for edge in adj_edges.values():
        u, v = edge[0], edge[1]
        edges.append([u, v, {'door': False}])

    
    # /// edges to graph
    G.add_edges_from(edges)

    return G