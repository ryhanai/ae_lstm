from pathlib import Path
from typing import List, Tuple
from numpy import object_
import yaml

from dataset.object_loader import ObjectInfo


object_info = ObjectInfo('ycb_conveni_v1', split='all')


def list_objects() -> List[Tuple[str, str, float]]:
    return [(name,
             object_info.usd_file(name),
             object_info.mass(name)) for name in object_info.names()]
    

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
# parser.add_argument("--usd_path", type=str, required=True)
# parser.add_argument("--object_prim_path", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("[DEBUG] launching app...", flush=True)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[DEBUG] app launched", flush=True)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaacsim.core.utils.stage import clear_stage
from pxr import UsdPhysics


import fnmatch

def find_prims_glob(stage, pattern: str):
    matches = []
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if fnmatch.fnmatch(path, pattern):
            matches.append(prim)
    return matches


def get_parent_prim_of_mesh(stage):
    results = []

    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            parent = prim.GetParent()
            if parent.IsValid():
                results.append((prim, parent))

    return results


def ensure_rigid_body_setup(stage, obj_prim, mass_value: float):
    if not obj_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb_api = UsdPhysics.RigidBodyAPI.Apply(obj_prim)
        print("[DEBUG] Applied RigidBodyAPI to root", flush=True)
        rb_api.CreateKinematicEnabledAttr(True)        

    if not obj_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(obj_prim)
    else:
        mass_api = UsdPhysics.MassAPI.Get(stage, obj_prim.GetPath())
    mass_api.CreateMassAttr(mass_value)
    print(f"[DEBUG] Set mass = {mass_value}", flush=True)

    mesh_children = [c for c in obj_prim.GetChildren() if c.GetTypeName() == "Mesh"]
    print(f"[DEBUG] mesh children = {[c.GetPath().pathString for c in mesh_children]}", flush=True)

    for mesh_prim in mesh_children:
        if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
            print(f"[DEBUG] Applied CollisionAPI to {mesh_prim.GetPath()}", flush=True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh_prim).CreateApproximationAttr("convexDecomposition")
            print(f"[DEBUG] Applied MeshCollisionAPI to {mesh_prim.GetPath()}", flush=True)            


# def get_spawned_object_root(stage, asset_path="/World/Asset"):
#     asset_root = stage.GetPrimAtPath(asset_path)
#     if not asset_root.IsValid():
#         raise RuntimeError(f"{asset_path} not found")

#     children = [c for c in asset_root.GetChildren()]
#     if len(children) == 0:
#         raise RuntimeError("No child prim under Asset")

#     if len(children) > 1:
#         print("[WARN] multiple roots, taking first")

#     return children[0]


def main():
    object_data = list_objects()

    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    CoMs = {}

    for name, usd_path, mass_value in object_data:
        print("=" * 80, flush=True)
        print(f"[DEBUG] processing: {usd_path}", flush=True)

        try:
            clear_stage()

            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
            light_cfg.func("/World/Light", light_cfg)

            asset_cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
            asset_cfg.func("/World/Asset", asset_cfg)

            prim = get_parent_prim_of_mesh(sim.stage)[0][1]  # get parent of the first mesh found

            ensure_rigid_body_setup(sim.stage, prim, mass_value=mass_value)

            print(f"[DEBUG] Using prim: {prim.GetPath()}", flush=True)

            rigid_obj = RigidObject(
                RigidObjectCfg(prim_path=prim.GetPath().pathString, spawn=None)
            )

            sim.reset()
            for _ in range(5):
                sim.step()

            rigid_obj.update(sim.get_physics_dt())

            world_com = rigid_obj.data.root_com_pos_w[0]
            local_com = rigid_obj.data.body_com_pose_b[0, 0, :3]

            print(f"{name}", flush=True)
            print(f"  World CoM: {world_com.cpu().numpy()}", flush=True)
            print(f"  Local CoM: {local_com.cpu().numpy()}", flush=True)

            CoMs[name] = (world_com.cpu().numpy(), local_com.cpu().numpy())

        except Exception as e:
            print(f"{name}: ERROR: {e}", flush=True)
            traceback.print_exc()


    # update objects.yaml
    for name, (world_com, local_com) in CoMs.items():
        try:
            object_info._info[name]["com"] = local_com.tolist()
        except Exception as e:
            print(f"{name} not found in object_info", flush=True)
            pass

    yaml_path = Path('./objects.yaml')
    exclude_keys = ["usd_file", "dataset"]
    
    filtered = {}
    for k, v in object_info._info.items():
        v2 = {k2: v2 for k2, v2 in v.items() if k2 not in exclude_keys}    
        filtered[k] = v2

    print(filtered)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            filtered,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR] exception occurred:", flush=True)
        traceback.print_exc()
    finally:
        print("[DEBUG] closing app", flush=True)
        simulation_app.close()