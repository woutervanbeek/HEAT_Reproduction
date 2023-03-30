import torch
import numpy as np
import pickle
import math

import torch.distributed as dist

# // Training (including parallelization / distribution)
def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def adjust_learning_rate(optimizer, init_lr, epoch, cfg):
    '''
    Decay the learning rate based on schedule
    '''
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / cfg.train.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# // Image processing 
def tens2img(img):
    img = img.cpu().clone().detach().numpy() #make sure works always
    return img.transpose(1,2,0)

def normalize_image(img):
    img_n = img - np.min(img)
    img_n /= np.max(img_n)
    return img_n

def normalize(mat):
    mat_n = mat - np.min(mat, axis=0)
    mat_n /= np.max(mat_n, axis=0)
    return mat_n



# // Helper functions RPLAN
ROOM_ARRAY = [[0,  (238,42,59)],     #living room
                [1,  (0,217,174)],     #master room
                [2,  (169,195,230)],   #kitchen
                [3,  (96,59,138)],     #bathroom
                [4,  (130,112,178)],   #dining room
                [5,  (121,255,229)],   #child room
                [6,  (80, 70, 70)],    #study room
                [7,  (121,200,100)],   #second room
                [8,  (0,94,144)],      #guest room 
                [9,  (252,216,4)],     #balcony
                [10, (225,147,37)],    #entrance
                [11, (182,2,189)],     #storage
                [12, (70, 80, 70)],    #wall-in 
                [13, (230, 230, 230)], #external area 
                [14, (0,0,0)],         #exterior wall
                [15, (200,200,200)],   #front door
                [16, (0,0,0)],         #interior wall
                [17, (150,150,150)]]   #interior door

ROOM_ARRAY_Z = [[0,  (238,42,59)],     #living room -> living room
                    [1,  (0,217,174)],     #master room -> bedroom
                    [2,  (169,195,230)],   #kitchen -> kitchen
                    [3,  (96,59,138)],     #bathroom -> bathroom
                    [4,  (130,112,178)],   #dining room -> dining room
                    [5,  (0,217,174)],     #child room -> bedroom
                    [6,  (80, 70, 70)],    #study room -> study room
                    [7,  (0,217,174)],     #second room -> bedroom
                    [8,  (0,217,174)],     #guest room -> bedroom
                    [9,  (252,216,4)],     #balcony -> balcony
                    [10, (225,147,37)],    #entrance -> entrance
                    [11, (182,2,189)],     #storage -> storage
                    [12, (0,0,0)],         #wall-in -> unknown
                    [13, (230, 230, 230)], #external area -> unknown
                    [14, (0,0,0)],         #exterior wall -> unknown
                    [15, (200,200,200)],   #front door -> front door
                    [16, (80,80, 80)],     #interior wall -> unknown
                    [17, (150,150,150)]]   #interior door -> interior door

COLORS_ORDERED = [np.array(ROOM_ARRAY_Z[n][1])/255 for n in range(len(ROOM_ARRAY_Z))]
ROOM_ARRAY_TOGETHER = [[0], [1, 5, 7, 8], [2], [3], [4], [6], [9], [10], [11]]

def colorize_floorplan(img, color_array=COLORS_ORDERED):
    
    h, w = np.shape(img)
    img_c = np.zeros((h, w, 3)).astype(int)
    for i, color in enumerate(color_array):
        img_c[img == i, :] = (color*255).astype(int)

    return img_c

def remove_attributes_from_graph(graph, list_attr=['polygons']):
    for attr in list_attr:
        for n in graph.nodes(): # delete irrelevant nodes
            del graph.nodes[n][attr]
    return graph


# // Remaining
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)