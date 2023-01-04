from matplotlib import pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline

cmap = plt.get_cmap('viridis')

def draw_largest_bounding_area(interval, ax):
    ax.add_patch(patches.Rectangle(
        xy=(interval[0,0], interval[1,0]),
        width=interval[0,1] - interval[0,0],
        height=interval[1,1] - interval[1,0],
        facecolor='none',
        edgecolor=cmap(0.5),
        label= "largest bounding area",
        linewidth=3.0
        )
    )

def draw2dpattern(interval, ax, classfication_result, minimized_f_hat, total):
    ax.add_patch(patches.Rectangle(
        xy=(interval[0,0], interval[1,0]),
        width=interval[0,1] - interval[0,0],
        height=interval[1,1] - interval[1,0],
        facecolor='none',
        edgecolor=cmap(0/total),
        label=f"point anomalous: {classfication_result} | f_hat: {minimized_f_hat}"
        )
    )
