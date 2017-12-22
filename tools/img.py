import numpy as np
import tensorflow as tf
import scipy
import math

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = scale
    t = np.zeros((3, 3))
    t[0][0] = float(res[1]) / h[1]
    t[1][1] = float(res[0]) / h[0]
    t[0][2] = res[1] * (-float(center[0]) / h[1] + .5)
    t[1][2] = res[0] * (-float(center[1]) / h[0] + .5)
    t[2][2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    return scipy.misc.imresize(new_img, res)

def two_pt_crop(img, scale, pt1, pt2, pad, res, chg=None):
    center = (pt1+pt2) / 2
    scale = max(20*scale, np.linalg.norm(pt1-pt2)) * .007
    scale *= pad
    angle = math.atan2(pt2[1]-pt1[1],pt2[0]-pt1[0]) * 180 / math.pi - 90
    flip = False

    # Handle data augmentation
    if chg is not None:
        # Flipping
        if 'flip' in chg:
            if np.random.rand() < .5:
                flip = True
        # Scaling
        if 'scale' in chg:
            scale *= min(1+chg['scale'], max(1-chg['scale'], (np.random.randn() * chg['scale']) + 1))
        # Rotation
        if 'rotate' in chg:
            angle += np.random.randint(-chg['rotate'], chg['rotate'] + 1)
        # Translation
        if 'translate' in chg:
            for i in range(2):
                offset = np.random.randint(-chg['translate'], chg['translate'] + 1) * scale
                center[i] += offset

    # Create input image
    cropped = crop(img, center, scale, res, rot=angle)
    inp = np.zeros((3, res[0], res[1]))
    for i in range(3):
        inp[i, :, :] = cropped[:, :, i]

    # Create heatmap
    hm = np.zeros((2,res[0],res[1]))


    if flip:
        inp = np.array([np.fliplr(inp[i]) for i in range(len(inp))])
        hm = np.array([np.fliplr(hm[i]) for i in range(len(hm))])

    return inp, hm