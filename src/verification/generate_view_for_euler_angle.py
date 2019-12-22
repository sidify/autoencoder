import cv2
import numpy as np
from ..renderer import meshrenderer_phong as mp
from ..renderer.pysixd_stuff import view_sampler
import math as m

def __euler_to_rotation_matrix__(euler):
    R = []
    for angle in euler:
        rx, ry, rz = angle
        Rx = np.array((
            [1, 0, 0],
            [0, m.cos(rx), -m.sin(rx)],
            [0, m.sin(rx), m.cos(rx)]
        ))
        Ry = np.array((
            [m.cos(ry), 0, m.sin(ry)],
            [0, 1, 0],
            [-m.sin(ry), 0, m.cos(ry)]
        ))
        Rz = np.array((
            [m.cos(rz), -m.sin(rz), 0],
            [m.sin(rz), m.cos(rz), 0],
            [0, 0, 1]
        ))

        R.append(np.matmul(Rz, np.matmul(Ry, Rx)))
    return R


def __render_for_rotations__(R, renderer, dim):
    height = 720
    width = 960
    clip_near = 10
    clip_far = 1000

    rendered = np.empty((len(R), dim[0], dim[1], dim[2]), dtype=np.uint8)
    K = np.array(([1029.87472, 0, 480, 0, 1029.69249, 350, 0, 0, 1])).reshape((3, 3))
    t = np.array(([0, 0, 700]), dtype=np.float16)
    i = 0
    for rot in R:
        color, depth_x = renderer.render(
            0, int(width), int(height), K
            , rot, t, clip_near, clip_far)
        ys, xs = np.nonzero(depth_x > 0)
        ys = np.array(ys, dtype=np.int16)
        xs = np.array(xs, dtype=np.int16)
        x, y, w, h = view_sampler.calc_2d_bbox(xs, ys, (width, height))
        img = color[y:y + h, x:x + w]
        img = cv2.resize(img, (dim[0], dim[1]))
        rendered[i] = img
        i += 1

    return rendered

if __name__ = "__main__":

    cad_model = '/home/sid/thesis/ply/models_cad/obj_05_red.ply'
    euler_angles = np.array((
        [0, 0, 1.2]
    ))
    dim = (128, 128, 3)

    R = __euler_to_rotation_matrix__(euler_angles)
    renderer = mp.Renderer(cad_model, samples=1, vertex_tmp_store_folder='.', clamp=False, vertex_scale=1.0)

    rendered = __render_for_rotations__(R, renderer, dim)

    for i, img in enumerate(rendered):
        print(euler_angles[i])
        cv2.imshow()
        cv2.waitKey(0)
        cv2.destroyAllWindows()





