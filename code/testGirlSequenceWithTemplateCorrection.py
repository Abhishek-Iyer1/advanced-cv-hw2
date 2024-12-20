import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--template_threshold',
    type=float,
    default=5,
    help='threshold for determining whether to update template',
)
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("data/girlseq.npy")
seq = seq - 255 # invert the image
rect = [280, 152, 330, 318]

rect_history = np.zeros((seq.shape[2], 4))
rect_history[0] = np.array(rect.copy()).T
p_history = np.zeros((seq.shape[2], 2, 1))

# Create subplots
fig, ax = plt.subplots(1, 5)
j = 0
I1 = seq[:, :, 0]
rect0 = rect.copy()

for i in tqdm(range(1, seq.shape[2] - 1)):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]

    p_history[i] = LucasKanade(It, It1, rect_history[i-1], threshold, num_iters, p0=p_history[i - 1])
    p_diff = np.array((rect_history[i-1][0] - rect_history[0][0], rect_history[i-1][1] - rect_history[0][1])).reshape((-1, 1))
    p_star = LucasKanade(I1, It1, rect_history[0], threshold, num_iters, p0=p_history[i]+p_diff)

    # Update the template if the norm of p_star is greater than the template threshold+
    if np.linalg.norm(p_star - (p_history[i] + p_diff)) < template_threshold:
        p_history[i] = p_star - p_diff
    
    rect_history[i][0] = rect_history[i-1][0] + p_history[i][0]
    rect_history[i][2] = rect_history[i-1][2] + p_history[i][0]
    rect_history[i][1] = rect_history[i-1][1] + p_history[i][1]
    rect_history[i][3] = rect_history[i-1][3] + p_history[i][1]

    # Do subplots using cv2.rectangle
    if i in [1, 20, 40, 60, 80]:
        It1 = (It1*255).astype(np.uint8)
        rect = [int(x) for x in rect_history[i]]
        if len(It1.shape) == 2:  # If grayscale, convert to BGR for rectangle drawing
            It1 = cv2.cvtColor(It1, cv2.COLOR_GRAY2BGR)
        It1_rect = cv2.rectangle(It1, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        plt.imshow(It1_rect, cmap="gray")
        plt.show()

plt.savefig("data/girl_rects_subplots_wcrt.png")
plt.show()
np.save("data/girlseqrects-wcrt.npy", rect_history)
