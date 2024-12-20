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
    '--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("data/carseq.npy")
rect = [59, 116, 145, 151]

rect_history = np.zeros((seq.shape[2], 4))

# Create subplots
fig, ax = plt.subplots(1, 5)
j = 0

for i in tqdm(range(seq.shape[2] - 1)):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] += p[0]
    rect[2] += p[0]
    rect[1] += p[1]
    rect[3] += p[1]
    rect_history[i, :] = np.array(rect.copy()).T

    # Do subplots using cv2.rectangle
    if i in [1, 100, 200, 300, 400]:
        It1 = (It1*255).astype(np.uint8)
        rect = [int(x) for x in rect]
        if len(It1.shape) == 2:  # If grayscale, convert to BGR for rectangle drawing
            It1 = cv2.cvtColor(It1, cv2.COLOR_GRAY2BGR)
        It1_rect = cv2.rectangle(It1, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        ax[j].imshow(It1_rect, cmap="gray")
        ax[j].set_title(f"Frame {i}")
        ax[j].axis("off")
        j += 1

plt.savefig("data/car_rects_subplots.png")
plt.show()
np.save("data/carseqrects.npy", rect_history)
