### from https://github.com/KrishnaswamyLab/MIOFlow/blob/main/MIOFlow/datasets.py

import pandas as pd
import numpy as np
import phate
from phate import PHATE

from sklearn import datasets

import seaborn as sns
sns.color_palette("bright")

import warnings


def construct_diamond(
    points_per_petal:int=200,
    petal_width:float=0.25,
    direction:str='y'
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        direction (str): Defaults to 'y'. Options `'y'` or `'x'`. Whether to make vertical
            or horizontal diamonds.
    Returns:
    ---------
        points (numpy.ndarray): the 2d array of points. 
    '''
    n_side = int(points_per_petal/2)
    axis_1 = np.concatenate((
                np.linspace(0, petal_width, int(n_side/2)), 
                np.linspace(petal_width, 0, int(n_side/2))
            ))
    axis_2 = np.linspace(0, 1, n_side)
    axes = (axis_1, axis_2) if direction == 'y' else (axis_2, axis_1)
    points = np.vstack(axes).T
    points = np.vstack((points, -1*points))
    points = np.vstack((points, np.vstack((points[:, 0], -1*points[:, 1])).T))
    return points

def make_diamonds(
    points_per_petal:int=200,
    petal_width:float=0.25,
    colors:int=5,
    scale_factor:float=30,
    use_gaussian:bool=True   
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        colors (int): Defaults to `5`. The number of timesteps (colors) to produce.
        scale_factor (float): Defaults to `30`. How much to scale the noise by 
            (larger values make samller noise).
        use_gaussian (bool): Defaults to `True`. Whether to use random or gaussian noise.
    Returns:
    ---------
        df (pandas.DataFrame): DataFrame with columns `samples`, `x`, `y`, where `samples`
            are the time index (corresponds to colors) 
    '''    
    upper = construct_diamond(points_per_petal, petal_width, 'y')
    lower = construct_diamond(points_per_petal, petal_width, 'x')
    data = np.vstack((upper, lower)) 
    
    noise_fn = np.random.randn if use_gaussian else np.random.rand
    noise = noise_fn(*data.shape) / scale_factor
    data = data + noise
    df = pd.DataFrame(data, columns=['d1', 'd2'])
    
    c_values = np.linspace(colors, 1, colors)
    c_thresholds = np.linspace(1, 0+1/(colors+1), colors)
    df.insert(0, 'samples', colors)
    df['samples'] = colors 
    for value, threshold in zip(c_values, c_thresholds):
        index = ((np.abs(df.d1) <= threshold) & (np.abs(df.d2) <= threshold))
        df.loc[index, 'samples'] = value
    df.set_index('samples')
    return df


def make_swiss_roll(n_points=1500):
    '''
    Arguments:
    ----------
        n_points (int): Default to `1500`. 

    Returns:
    ---------
        df (pandas.DataFrame): DataFrame with columns `samples`, `d1`, `d2`, `d3`, 
            where `samples` are the time index (corresponds to colors) 
    '''  
    X, color = datasets.make_swiss_roll(n_samples=n_points)
    df = pd.DataFrame(np.hstack((np.round(color).reshape(-1, 1), X)), columns='samples d1 d2 d3'.split())
    df.samples -= np.min(df.samples)
    return df


def make_tree():
    '''
    Arguments:
    ----------
        n_points (int): Default to `1500`. 
        
    Returns:
    ---------
        df (pandas.DataFrame): DataFrame with columns `samples`, `d1`, `d2`, `d3`, 
            `d4`, `d5` where `samples` are the time index (corresponds to colors) 
    '''  
    tree, branches = phate.tree.gen_dla(
        n_dim = 200, n_branch = 10, branch_length = 300, 
        rand_multiplier = 2, seed=37, sigma = 5
    )
    phate_operator = phate.PHATE(n_components=5, n_jobs=-1)
    tree_phate = phate_operator.fit_transform(tree)
    df = pd.DataFrame(np.hstack((branches.reshape(-1, 1), tree_phate)), columns='samples d1 d2 d3 d4 d5'.split())
    return df

def rings(
    N:int, M:int = None, 
    data_scale:float = 1, 
    add_noise:bool = True, 
    noise_scale_theta:float = 0.7, 
    noise_scale_radius:float = 0.03,
    buffer:float = 0.8,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Arguments:
        N (int): Number of points to make.
        M (int): Defaults to `None`. If `M='auto'` will automatically determine how many circles to make.
        data_scale (float): Defaults to `1`. Multiplier to rescale the data.
        add_noise (bool): Defaults to `True`. Whether or not to add noise to the data.
        noise_scale_theta (float): Defaults to `0.7`. How much to scale the noise added to `theta`.
        noise_scale_radius (float): Defaults to `0.3`. How much to scale the noise added to `radius`.
        buffer (float): Defaults to `0.8`. How much to scale the `radius` to add some padding between circles.
        **kwargs    
        
    Returns:
        X (np.ndarray): The x, y coordinates for the points.
        C (np.ndarray): The cluster number of each point.
    '''
    
    """Generate petal data set."""
    X = []  # points in respective petals
    Y = []  # auxiliary array (points on outer circle)
    C = []

    assert N > 4, "Require more than four data points"

    # Number of 'petals' to point into the data set. This is required to
    # ensure that the full space is used.
    if M is None:
        M = int(np.floor(np.sqrt(N)))
    thetas = np.linspace(0, 2 * np.pi, M, endpoint=False)
    

    for theta in thetas:
        Y.append(np.asarray([np.cos(theta), np.sin(theta)]))
        
    # Radius of the smaller cycles is half of the chord distance between
    # two 'consecutive' points on the circle.
    radius = 0.5 * np.linalg.norm(Y[0] - Y[1])    

    for i, x in enumerate(Y):
        for theta in thetas:
            for j in range(N // M // len(thetas)):
                r = radius if not add_noise else radius + np.random.randn() * noise_scale_radius
                t = theta if not add_noise else theta + np.random.randn() * noise_scale_theta
                r *= buffer
                X.append(np.asarray([r * np.cos(t) - x[0], r * np.sin(t) - x[1]]))

                # Indicates that this point belongs to the $i$th circle.
                C.append(i)
    X = np.asarray(X)
    C = np.asarray(C)
    X *= data_scale
    return X, C