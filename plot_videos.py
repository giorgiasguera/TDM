#NUOVA VERSIONE (normalizzazione+ tolto *2 + modifiche wandb)
import os
import cv2
import math
import numpy as np
from dtw import dtw
import imageio #fix

from loss import TARGET_PAD   # [2] invece di PAD_TOKEN da constants


# Tripla (start_joint, end_joint, bone_colour_idx) diversa da quella di helpers(che usa(parent, child))
# Usata solo per disegnare le ossa in draw_frame_2D
def getSkeletalModelStructure():
    # Definition of skeleton model structure:
    #   The structure is an n-tuple of:
    #
    #   (index of a start point, index of an end point, index of a bone)
    #
    #   E.g., this simple skeletal model
    #
    #             (0)
    #              |
    #              |
    #              0
    #              |
    #              |
    #     (2)--1--(1)--1--(3)
    #      |               |
    #      |               |
    #      2               2
    #      |               |
    #      |               |
    #     (4)             (5)
    #
    #   has this structure:
    #
    #   (
    #     (0, 1, 0),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 2),
    #     (3, 5, 2),
    #   )
    #
    #  Warning 1: The structure has to be a tree.
    #  Warning 2: The order isn't random. The order is from a root to lists.
    #

    return (
        # head
        (0, 1, 0),
        # left shoulder
        (1, 2, 1),
        # left arm
        (2, 3, 2),
        # (3, 4, 3),
        # Changed to avoid wrist, go straight to hands
        (3, 29, 3),
        # right shoulder
        (1, 5, 1),
        # right arm
        (5, 6, 2),
        # (6, 7, 3),
        # Changed to avoid wrist, go straight to hands
        (6, 8, 3),

        # left hand - wrist
        # (7, 8, 4),

        # left hand - palm
        (8, 9, 5),
        (8, 13, 9),
        (8, 17, 13),
        (8, 21, 17),
        (8, 25, 21),


        # left hand - 1st finger
        (9, 10, 6),
        (10, 11, 7),
        (11, 12, 8),
        
        # left hand - 2nd finger
        (13, 14, 10),
        (14, 15, 11),
        (15, 16, 12),
        
        # left hand - 3rd finger
        (17, 18, 14),
        (18, 19, 15),
        (19, 20, 16),
        
        # left hand - 4th finger
        (21, 22, 18),
        (22, 23, 19),
        (23, 24, 20),
        
        # left hand - 5th finger
        (25, 26, 22),
        (26, 27, 23),
        (27, 28, 24),
        
        # right hand - wrist
        # (4, 29, 4),

        # right hand - palm
        (29, 30, 5),
        (29, 34, 9),
        (29, 38, 13),
        (29, 42, 17),
        (29, 46, 21),
        
        # right hand - 1st finger
        (30, 31, 6),
        (31, 32, 7),
        (32, 33, 8),
        
        # right hand - 2nd finger
        (34, 35, 10),
        (35, 36, 11),
        (36, 37, 12),
        
        # right hand - 3rd finger
        (38, 39, 14),
        (39, 40, 15),
        (40, 41, 16),
        
        # right hand - 4th finger
        (42, 43, 18),
        (43, 44, 19),
        (44, 45, 20),
        
        # right hand - 5th finger
        (46, 47, 22),
        (47, 48, 23),
        (48, 49, 24),
    )


# Draw a line between two points, if they are positive points (disegna l'osso)
def draw_line(im, joint1, joint2, c=(0, 0, 255), t=1, width=3):
    """
    Disegna una linea ellittica tra due joint 2D.
    Se uno dei joint è sotto la soglia (joint non valido), non disegna.
    """
    thresh = -100
    if (joint1[0] > thresh and joint1[1] > thresh and
            joint2[0] > thresh and joint2[1] > thresh):
        center = (int((joint1[0] + joint2[0]) / 2),
                  int((joint1[1] + joint2[1]) / 2))
        length = int(math.sqrt(((joint1[0] - joint2[0]) ** 2) +
                               ((joint1[1] - joint2[1]) ** 2)) / 2)
        angle = math.degrees(math.atan2((joint1[0] - joint2[0]),
                                        (joint1[1] - joint2[1])))
        cv2.ellipse(im, center, (width, length), -angle, 0.0, 360.0, c, -1)


# get bone colour given index
def get_bone_colour(skeleton, j):
    """
    Restituisce il colore del bone j. Colori di PT
    """
    bone = skeleton[j, 2]

    if bone == 0:  # head
        c = (0, 153, 0)
    elif bone == 1:  # Shoulder
        c = (0, 0, 255)

    elif bone == 2 and skeleton[j, 1] == 3:  # left arm
        c = (0, 102, 204)
    elif bone == 3 and skeleton[j, 0] == 3:  # left lower arm
        c = (0, 204, 204)

    elif bone == 2 and skeleton[j, 1] == 6:  # right arm
        c = (0, 153, 0)
    elif bone == 3 and skeleton[j, 0] == 6:  # right lower arm
        c = (0, 204, 0)

    # Hands
    elif bone in [5, 6, 7, 8]:
        c = (0, 0, 255)
    elif bone in [9, 10, 11, 12]:
        c = (51, 255, 51)
    elif bone in [13, 14, 15, 16]:
        c = (255, 0, 0)
    elif bone in [17, 18, 19, 20]:
        c = (204, 153, 255)
    elif bone in [21, 22, 23, 24]:
        c = (51, 255, 255)

    return c


# Draw the frame given 2D joints that are in the Inverse Kinematics format
def draw_frame_2D(frame, joints, already_scaled=False):
    # Line to be between the stacked (linea di separazione)
    draw_line(frame, [1, 650], [1, 1], c=(0, 0, 0), t=1, width=1)
    # Give an offset to center the skeleton around
    offset = [350, 250]

    # Get the skeleton structure details of each bone, and size
    skeleton = getSkeletalModelStructure()
    skeleton = np.array(skeleton)
    
    number = skeleton.shape[0]

    # Scala e centra i joint nel frame
    if not already_scaled:
      joints = joints * 10 * 12 #* 2 QUA
      joints = joints + np.ones((50, 2)) * offset

    # Loop through each of the bone structures, and plot the bone
    for j in range(number):
        c = get_bone_colour(skeleton, j)
        draw_line(frame,
                  [joints[skeleton[j, 0]][0], joints[skeleton[j, 0]][1]],
                  [joints[skeleton[j, 1]][0], joints[skeleton[j, 1]][1]],
                  c=c, t=1, width=1)


# Find the average of the given frames
def avg_frames(frames):
    frames_sum = np.zeros_like(frames[0])
    for frame in frames:
        frames_sum += frame
    return frames_sum / len(frames)


# Plot a video given a tensor of joints, a file path, video name and references/sequence ID
def plot_video(joints,
               file_path,
               video_name,
               references=None,
               skip_frames=1,
               sequence_ID=None):

    # Create video template
    FPS = (25 // skip_frames)

    # ipdb.set_trace()
    video_file = file_path + "/{}.mp4".format(sequence_ID.split(".")[0])
    video_path, _ = os.path.split(video_file) #qua
    
    if not os.path.exists(video_path):
        os.mkdir(video_path,exist_ok=True)

    # --- nuova Normalizzazione  ---
    all_seqs = [f * 3 for f in joints if not np.allclose(f, TARGET_PAD)]
    if references is not None:
        all_seqs += [r * 3 for r in references if not np.allclose(r, TARGET_PAD)]

    if len(all_seqs) > 0:
      all_xy   = np.array(all_seqs).reshape(-1, 50, 3)[:, :, :2].reshape(-1, 2)
      xy_min   = all_xy.min(axis=0)
      xy_max   = all_xy.max(axis=0)
      xy_range = np.maximum(xy_max - xy_min, 1e-8)
      # Scala uniforme: il lato più lungo occupa 500 pixel
      seq_scale  = 500.0 / xy_range.max()
      seq_offset = np.array([325.0, 300.0]) - seq_scale * (xy_min + xy_max) / 2.0
    else:
      seq_scale  = 240.0
      seq_offset = np.array([350.0, 250.0])
    # -------------------------------------

    frames_da_salvare = []

    for (j, frame_joints) in enumerate(joints):

        is_pred_padding = np.allclose(frame_joints, TARGET_PAD)
        frame = np.ones((650, 650, 3), np.uint8) * 255

        if not is_pred_padding:
            fj = frame_joints * 3
            fj_2d = np.reshape(fj, (50, 3))[:, :2]
            # Applica la scala calcolata sulla sequenza intera
            fj_2d = fj_2d * seq_scale + seq_offset
            draw_frame_2D(frame, fj_2d, already_scaled=True)

        cv2.putText(frame, "Predicted Sign Pose", (180, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # If reference (frame GT) is provided, create and concatenate on the end
        if references is not None:
            # Extract the reference joints
            ref_joints = references[j]
            # Initialise frame of white
            ref_frame = np.ones((650, 650, 3), np.uint8) * 255

            if not np.allclose(ref_joints, TARGET_PAD):
                rj = ref_joints * 3
                rj_2d = np.reshape(rj, (50, 3))[:, :2]
                rj_2d = rj_2d * seq_scale + seq_offset
                draw_frame_2D(ref_frame, rj_2d, already_scaled=True)

            cv2.putText(ref_frame, "Ground Truth Pose", (190, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            frame = np.concatenate((frame, ref_frame), axis=1)

            sequence_ID_write = "Sequence ID: " + sequence_ID.split("/")[-1]
            cv2.putText(frame, sequence_ID_write, (150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames_da_salvare.append(frame_rgb)

    if frames_da_salvare:
        imageio.mimsave(video_file, frames_da_salvare, fps=float(FPS), macro_block_size=1)



# alter_DTW_timing — riallineamento temporale tramite DTW
def alter_DTW_timing(pred_seq, ref_seq):

    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    # no counter
    d, cost_matrix, acc_cost_matrix, path = dtw(ref_seq, pred_seq, dist=euclidean_norm)
    
    # Normalise the dtw cost by sequence length
    d = d / acc_cost_matrix.shape[0]

    # Costruisce la sequenza predetta riallineata
    new_pred_seq = np.zeros_like(ref_seq)
    # j tracks the position in the reference sequence
    j = 0
    skips = 0
    squeeze_frames = []

    for (i, pred_num) in enumerate(path[0]):
        if i == len(path[0]) - 1:
            break

        if path[1][i] == path[1][i + 1]:
            skips += 1

        # If a double coming up
        if path[0][i] == path[0][i + 1]:
            squeeze_frames.append(pred_seq[i - skips])
            j += 1
        elif path[0][i] == path[0][i - 1]:
            new_pred_seq[pred_num] = avg_frames(squeeze_frames)
            squeeze_frames = []
        else:
            new_pred_seq[pred_num] = pred_seq[i - skips]

    return new_pred_seq, ref_seq, d
