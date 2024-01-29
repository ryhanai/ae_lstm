import os
from pathlib import Path

import yaml


class ObjectInfo:
    object_dir = f"{os.environ['HOME']}/Dataset/ycb_conveni"
    config_dir = f"{os.environ['HOME']}/Program/moonshot/ae_lstm/specification/config"

    def __init__(self, dataset="ycb_conveni"):
        self.load_config(ObjectInfo.object_dir, ObjectInfo.config_dir, dataset)

    def load_config(self, object_dir, config_dir, dataset):
        with open(os.path.join(config_dir, f"dataset_{dataset}.yaml")) as f:
            dataset_def = yaml.safe_load(f)["train"]

        with open(os.path.join(f"{config_dir}", "objects.yaml")) as f:
            obj_def = yaml.safe_load(f)

        self._info = {}
        try:
            for name in dataset_def["ycb"]:
                prop = obj_def["ycb"][name]
                self._info[name] = {
                    "mass": prop["mass"],
                    "usd_file": f"{object_dir}/ycb/{name}/google_16k/textured/textured.usd",
                    "dataset": "ycb",
                }
        except:
            pass

        try:
            for name in dataset_def["conveni"]:
                prop = obj_def["conveni"][name]
                self._info[name] = {
                    "mass": prop["mass"],
                    "usd_file": f"{object_dir}/conveni/{name}/{name}/{name}.usd",
                    "dataset": "conveni",
                }
        except:
            pass

    def usd_file(self, name):
        return self._info[name]["usd_file"]

    def obj_file(self, name):
        """
        return obj file name and scale
        """
        p = Path(self.usd_file(name))
        if self.dataset(name) == "ycb":
            obj_file = p.parent.parent / "textured.obj"
        if self.dataset(name) == "conveni":
            obj_file = p.parent.parent / "textured.obj"
        scale = 1.0
        return str(obj_file), scale

        # if name == "seria_basket":
        #     return f"{obj_dir}/seria_basket_body_collision.obj", 0.001
        # else:
        #     return f"{obj_dir}/ycb/{name}/google_16k/textured.obj", 1.0

    def rviz_mesh_file(self, name):
        if self.dataset(name) == "ycb":
            mesh_file = f"package://force_estimation/meshes/ycb/{name}/google_16k/textured.dae"
        if self.dataset(name) == "conveni":
            mesh_file = f"package://force_estimation/meshes/conveni/{name}/textured.obj"

        scale = 1.0
        return mesh_file, scale

    def mass(self, name):
        return self._info[name]["mass"]

    def dataset(self, name):
        return self._info[name]["dataset"]

    def names(self):
        return list(self._info.keys())

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        names = self.names()
        if self._i >= len(names):
            raise StopIteration
        name = names[self._i]
        self._i += 1
        return name, self.usd_file(name), self.mass(name)


if __name__ == "__main__":
    conf = ObjectInfo("ycb_conveni_v1")
