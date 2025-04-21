import os
import numpy as np
import polyscope as ps
import polyscope.imgui as psg
import trimesh

from mimo.data_gen.data_split_cfg import MUG_TRAIN
from mimo.data_gen.panda_sample import Object, PandaGripper, sample_multiple_grasps
from mimo.vis_utils.ps_vis_utils import ps_draw_panda_gripper

from dataset.object_loader import ObjectInfo

oi = ObjectInfo()
asset_dir = "/home/ryo/Program/MIMO/src/mimo/eval/ndf/assets"

# f_ns_list = os.listdir(mesh_dir)
# bad_obj_list = np.load(os.path.join(asset_dir, f"bad_{obj}s_all.npz"))['bad_ids'].tolist()
# mug_train_list = MUG_TRAIN

class VisGrasp:
    def __init__(self):
        self.n_objs = len(oi.names())
        self.obj_id = 0
        self.mesh_scale = 0.5
        self.n_grasps = 500
        self.obj = None

        self.panda = PandaGripper()
        self.collision_manager = trimesh.collision.CollisionManager()


    def load_mesh(self):
        # f_name = os.path.join(mesh_dir, self.obj_list[self.obj_id], "models/model_128_df.obj")
        f_name = oi.obj_file(oi.names()[self.obj_id])[0]
        self.obj = Object(f_name, self.mesh_scale)
        print(f"mesh {f_name} is watertight: {self.obj.mesh.is_watertight}")
        ps.register_surface_mesh("obj", self.obj.mesh.vertices, self.obj.mesh.faces)

    def sample_grasp(self):
        # grasps = [np.eye(4)]
        trans, quality = sample_multiple_grasps(self.n_grasps, self.obj.mesh, self.panda, systematic_sampling=False)
        quality = np.array(quality["quality_antipodal"])
        # trans = trans[quality > 0.01]  # default
        trans = trans[quality > 0.05]        
        grasp = trans.tolist()
        print(grasp)
        print(f"Number of high quality grasps: {len(grasp)}")
        ps_draw_panda_gripper(grasp)

    def callback(self):
        ps.set_ground_plane_mode("none")
        psg.PushItemWidth(200)

        psg.TextUnformatted("Load mesh.")
        changed, self.mesh_scale = psg.SliderFloat("mesh scale", self.mesh_scale, v_min=0.2, v_max=0.8)

        changed, self.obj_id = psg.SliderInt("obj_id", self.obj_id, v_min=0, v_max=self.n_objs - 1)
        if changed:
            print(f"The obj idx is now: {oi.obj_file(oi.names()[self.obj_id])[0]}")

        if psg.Button("Load mesh."):
            self.load_mesh()

        psg.Separator()
        changed, self.n_grasps = psg.InputInt("number of samples", self.n_grasps)

        if psg.Button("Sample grasp."):
            self.sample_grasp()
            print(1)

    def run(self):
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_user_callback(self.callback)
        ps.show()

        ps.clear_user_callback()

def main():
    vis_grasp = VisGrasp()
    vis_grasp.run()

if __name__ == "__main__":
    main()