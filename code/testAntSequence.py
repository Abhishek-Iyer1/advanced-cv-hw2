import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument(
    '--num_iters', type=int, default=1000, help='number of iterations of Lucas-Kanade'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=1e-2,
    help='dp threshold of Lucas-Kanade for terminating optimization',
)
parser.add_argument(
    '--tolerance',
    type=float,
    default=0.2,
    help='binary threshold of intensity difference when computing the mask',
)
parser.add_argument(
    '--seq_file',
    default='data/antseq.npy',
)

args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance
seq_file = args.seq_file

seq = np.load(seq_file)

'''
HINT:
1. Create an empty array 'masks' to store the motion masks for each frame.
2. Set the initial mask for the first frame to False.
3. Use the SubtractDominantMotion function to compute the motion mask between consecutive frames.
4. Use the motion 'masks; array for visualization.
'''

# Create an empty array 'masks' to store the motion masks for each frame
masks = np.zeros((seq.shape[0], seq.shape[1], seq.shape[2] - 1), dtype=bool)

# Set the initial mask for the first frame to False
masks[:, :, 0] = False
last_frame = seq.shape[2] - 1
# last_frame = 10 # For testing purposes

for i in tqdm(range(last_frame)):
    It = seq[:, :, i]
    It1 = seq[:, :, i + 1]
    # M = LucasKanadeAffine(It, It1, threshold, int(num_iters))
    # NOTE: Uncomment the line below to use Inverse Composition
    M = InverseCompositionAffine(It, It1, threshold, int(num_iters))

    # Compute the motion mask
    mask = SubtractDominantMotion(It, It1, M, threshold, num_iters, tolerance)
    masks[:, :, i] = mask

fig, ax = plt.subplots(1, 4, figsize=(10, 10))
for i in range(4):
    
    it1 = seq[:, :, (i+1) * 30]
    color_img = cv2.cvtColor((255 * it1).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    mask = masks[:, :, (i+1) * 30]
    full_mask = (255 * mask).astype(np.uint8)
    masked_ants = cv2.bitwise_and(color_img, color_img, mask=~full_mask)

    solid_color = np.zeros((it1.shape[1], it1.shape[0], 3), np.uint8)
    solid_color[:] = (255, 0, 0)
    color_mask = cv2.bitwise_and(solid_color, solid_color, mask=full_mask)
    final_img = cv2.add(color_mask, masked_ants)

    ax[i].imshow(final_img)
    ax[i].set_title(f'Frame {(i+1) * 30}')
    ax[i].axis('off')

plt.savefig('data/antseq.png')
plt.close()


#Visualize the motion masks by overlaying masks on original frames
video_writer = cv2.VideoWriter('data/antseqmask.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (seq.shape[1], seq.shape[0]))
for i in range(masks.shape[2]):
    mask = masks[:, :, i]
    mask = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    video_writer.write(mask)