import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualize_as_pc(
    rgb_file="/home/ryo/Dataset/forcemap_evaluation/03_GAFS_1/000000-color.jpg",
    depth_file="/home/ryo/Dataset/forcemap_evaluation/03_GAFS_1/000000-depth.png",
    mask_file="/home/ryo/Program/moonshot/ae_lstm/scripts/evaluation/mask.png",
):
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Camera parameters:
    # FX_DEPTH = 5.8262448167737955e02
    # FY_DEPTH = 5.8269103270988637e02
    # CX_DEPTH = 3.1304475870804731e02
    # CY_DEPTH = 2.3844389626620386e02

    FX_DEPTH = 593.7068481445312
    CX_DEPTH = 313.9930419921875
    FY_DEPTH = 593.7068481445312
    CY_DEPTH = 236.69000244140625

    # Read depth and color image:
    # depth_image = cv2.imread("./depth_00000.tif", cv2.IMREAD_UNCHANGED)
    # rgb_image = cv2.imread("./rgb_00000.tif", cv2.IMREAD_UNCHANGED)

    depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # print(f"type(depth_image):{type(depth_image)}")
    # print(f"type(rgb_image):{type(rgb_image)}")
    # print(f"depth_image.shape:{depth_image.shape}")
    # print(f"rgb_image.shape:{rgb_image.shape}")
    # print(f"depth_image.dtype:{depth_image.dtype}")
    # print(f"rgb_image.dtype:{rgb_image.dtype}")

    depth_image = depth_image.astype(np.int32)
    rgb_image = rgb_image.astype(np.uint8)

    if mask_file != "":
        mask_image = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        th, mask_image = cv2.threshold(mask_image, 128, 1, cv2.THRESH_BINARY)
        depth_image = depth_image * mask_image

    color = o3d.geometry.Image(rgb_image)
    depth = o3d.geometry.Image(depth_image.astype(np.uint16))

    # Display depth and grayscale image:
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(depth, cmap="gray")
    axs[0].set_title("Depth image")
    axs[1].imshow(color)
    axs[1].set_title("RGB image")
    plt.show()

    # compute point cloud:
    # Both images has the same resolution
    height, width = depth_image.shape
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, FX_DEPTH, FY_DEPTH, CX_DEPTH, CY_DEPTH)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    # Convert to Open3D.PointCLoud:
    pcd_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # target_mesh = o3d.io.read_triangle_mesh("/home/ryo/Dataset/ycb_video/models/061_foam_brick/textured_simple.obj")
    # target_mesh = o3d.io.read_triangle_mesh("/home/ryo/Dataset/ycb_video/models/009_gelatin_box/textured_simple.obj")
    # target_mesh.compute_vertex_normals()
    # target_mesh.transform([[1, 0, 0, 0.2], [0, -1, 0, 0], [0, 0, -1, -0.5], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd_o3d, target_mesh], mesh_show_back_face=True)

    o3d.visualization.draw_geometries([pcd_o3d])
    # o3d.io.write_point_cloud("test.ply", pcd_o3d)


if __name__ == "__main__":
    # project_dir = "/home/ryo/Dataset/forcemap_evaluation.no_align/"
    project_dir = "/home/ryo/Dataset/forcemap_evaluation/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="01_UP_1")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--target", type=str, default="")
    args = parser.parse_args()

    rgb_file = f"{project_dir}/{args.scene}/{args.frame:06d}-color.jpg"
    depth_file = f"{project_dir}/{args.scene}/{args.frame:06d}-depth.png"
    if args.target != "":
        mask_file = f"{project_dir}/{args.scene}/{args.frame:06d}-label-{args.target}.png"
    else:
        mask_file = ""

    visualize_as_pc(rgb_file, depth_file, mask_file)
