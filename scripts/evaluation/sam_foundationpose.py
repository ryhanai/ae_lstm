import argparse
import os
from pathlib import Path

import cv2 as cv
import numpy as np
import pyrealsense2 as rs

# from datareader import *
# from estimater import *
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

SELECT_ROI_WINDOW = "Select ROI window"

SAM_CHECKPOINTS = {"L": "sam_vit_l_0b3195.pth", "B": "sam_vit_b_01ec64.pth", "H": "sam_vit_h_4b8939.pth"}


SAM_MODEL_KEYS = {"L": "vit_l", "B": "vit_b", "H": "default"}

MASKS = []


rgb_file = "/home/ryo/Dataset/forcemap_evaluation/05_UP_1/000000-color.jpg"
depth_file = "/home/ryo/Dataset/forcemap_evaluation/05_UP_1/000000-depth.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument("--mesh_file", type=str, default=f"{code_dir}/demo_data/mustard0/mesh/textured_simple.obj")
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    parser.add_argument("--sam_model", choices=["L", "B", "H"], default="H")
    args = parser.parse_args()

    # set_logging_format()
    # set_seed(0)

    # pipeline = rs.pipeline()
    # cfg = pipeline.start()
    # profile = cfg.get_stream(rs.stream.depth)
    # intr = profile.as_video_stream_profile().get_intrinsics()
    # print(intr)

    # intr_mtx = np.eye(3, 3)
    # intr_mtx[0, 0] = intr.fx
    # intr_mtx[1, 1] = intr.fy
    # intr_mtx[0, 2] = intr.ppx
    # intr_mtx[1, 2] = intr.ppy

    # mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    # os.system(f"rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam")

    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # scorer = ScorePredictor()
    # refiner = PoseRefinePredictor()
    # glctx = dr.RasterizeCudaContext()
    # est = FoundationPose(
    #     model_pts=mesh.vertices,
    #     model_normals=mesh.vertex_normals,
    #     mesh=mesh,
    #     scorer=scorer,
    #     refiner=refiner,
    #     debug_dir=debug_dir,
    #     debug=debug,
    #     glctx=glctx,
    # )
    # logging.info("estimator initialization done")

    sam = sam_model_registry[SAM_MODEL_KEYS[args.sam_model]](checkpoint=SAM_CHECKPOINTS[args.sam_model])
    sam.to("cuda")
    mask_predictor = SamPredictor(sam)

    segmented = False
    while True:
        # frames = pipeline.wait_for_frames()
        # depth = np.asanyarray(frames.get_depth_frame().as_frame().get_data()).astype(np.float32) / 1000
        # color = np.asanyarray(frames.get_color_frame().as_frame().get_data())
        depth = cv.imread(depth_file, cv.IMREAD_UNCHANGED)
        color = cv.imread(rgb_file, cv.IMREAD_UNCHANGED)
        color = cv.cvtColor(color, cv.COLOR_BGR2RGB)

        if not segmented:
            mask_predictor.set_image(color)
            segmented = True
            while True:
                x, y, w, h = cv.selectROI(SELECT_ROI_WINDOW, cv.cvtColor(color, cv.COLOR_RGB2BGR), showCrosshair=False)
                if x == y == w == h == 0:
                    print("Empty ROI")
                    cv.destroyWindow(SELECT_ROI_WINDOW)
                    continue
                box = np.array([x, y, x + w, y + h])
                masks, scores, logits = mask_predictor.predict(box=box, multimask_output=False)
                cv.namedWindow(f"Mask")
                cv.imshow("Mask", masks[0].astype(np.uint8) * 255)
                mask_for_merged = np.expand_dims(masks[0], 2)
                cv.imshow("Merged", mask_for_merged * cv.cvtColor(color, cv.COLOR_RGB2BGR))
                print("Check the results and press enter to accept it, or press 'u' to undo and retry")
                key = cv.waitKey()
                if key == 27:
                    exit()
                elif key == 13:
                    print(masks[0].dtype)
                    cv.imwrite("mask.png", masks[0].astype(np.uint8) * 255)
                    mask = masks[0].astype(bool)
                    print(mask.shape)
                    print(np.mean(depth))
                    # pose = est.register(
                    #     K=intr_mtx, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter
                    # )
                    break
                elif key == ord("u"):
                    continue
            cv.destroyAllWindows()

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f"{debug_dir}/model_tf.obj")
                xyz_map = depth2xyzmap(depth, intr_mtx)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f"{debug_dir}/scene_complete.ply", pcd)
        # else:
        #     pose = est.track_one(rgb=color, depth=depth, K=intr_mtx, iteration=args.track_refine_iter)

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(intr_mtx, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color, ob_in_cam=center_pose, scale=0.1, K=intr_mtx, thickness=3, transparency=0, is_input_rgb=True
            )
            cv2.imshow("1", vis[..., ::-1])
            cv2.waitKey(1)
