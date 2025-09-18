from os import read
from re import A
import readline
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import glob


class LeRobotRecorder:
    """
    A class to handle writing data for the LeRobot dataset.
    """

    VALIDITY_LABEL = "valid"

    def __init__(self, output_base="Downloads/conveni_gr00t", image_size=[120, 160]):
        self.OUTPUT_BASE = Path.home() / output_base
        self.image_size = image_size
        self.FPS = 60.0  # Isaac sim is 60 FPS by default, ALOHA is 30FPS

        # scan data directory and find the next episode index
        self._tasks = {}
        self._idx = 0

    def new_episode(self, task_description):
        self.VIDEO_DIR1 = self.OUTPUT_BASE / "videos/chunk-000/observation.images.left_view"
        self.VIDEO_DIR2 = self.OUTPUT_BASE / "videos/chunk-000/observation.images.right_view"
        self.DATA_DIR = self.OUTPUT_BASE / "data/chunk-000"
        self.META_DIR = self.OUTPUT_BASE / "meta"

        # Create necessary directories
        for d in [self.VIDEO_DIR1, self.VIDEO_DIR2, self.DATA_DIR, self.META_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self._task_description = task_description
        self._qpos = []
        self._qvel = []
        self._effort = []
        self._action = []
        self._image_left = []
        self._image_right = []        

    def step(self, qpos, qvel, effort, action, image_left, image_right) :
        self._qpos.append(qpos)
        self._qvel.append(qvel)
        self._effort.append(effort)
        self._action.append(action)
        self._image_left.append(image_left)
        self._image_right.append(image_right)

    def save_episode(self):
        # === This is called at the end of an episode ===
        self.save_video()
        self.update_task_meta()  # This must be before save_parquet()
        self.save_parquet()
        self.add_episode_meta()
        self.write_other_meta()
        self._idx += 1

    # def __del__(self):
    #     self.write_other_meta()

    def save_video(self):
        image_left = np.array(self._image_left)
        image_right = np.array(self._image_right)

        episode_id = f"episode_{self._idx:06d}"
        print(f"Processing {episode_id}...")
        video_path1 = self.VIDEO_DIR1 / f"{episode_id}.mp4"
        video_path2 = self.VIDEO_DIR2 / f"{episode_id}.mp4"

        h, w, _ = image_left[0].shape
        out1 = cv2.VideoWriter(str(video_path1), cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, (w, h))
        out2 = cv2.VideoWriter(str(video_path2), cv2.VideoWriter_fourcc(*'mp4v'), self.FPS, (w, h))
        for i in range(len(image_left)):
            out1.write(image_left[i])
            out2.write(image_right[i])
        out1.release()
        out2.release()

    def save_parquet(self):
        qpos = np.array(self._qpos)
        # qvel = np.array(self._qvel)
        # effort = np.array(self._effort)
        action = np.array(self._action)

        episode_id = f"episode_{self._idx:06d}"
        records = []
        num_steps = len(action)

        for t in range(num_steps):
            # state = np.concatenate([qpos[t], qvel[t], effort[t]])
            state = qpos[t]
            
            if t == num_steps - 1:
                next_done = True
                next_reward = 1.0
            else:
                next_done = False
                next_reward = 0.0

            records.append({
                "observation.state": state.tolist(),
                "action": action[t].tolist(),
                "timestamp": float(t) / self.FPS,
                # "frame_index": t,
                "episode_index": self._idx,
                "index": t,
                "task_index": self._tasks[self._task_description],  # task index
                "annotation.human.task_description": self._tasks[self._task_description],
                "next.reward": next_reward,
                "next.done": next_done,
            })
            
        df = pd.DataFrame(records)
        pq.write_table(pa.Table.from_pandas(df), self.DATA_DIR / f"{episode_id}.parquet")

    def add_episode_meta(self):
        # === Write meta/episodes.jsonl ===
        with open(str(self.META_DIR / "episodes.jsonl"), "a") as f:
            ep = {
                "episode_index": self._idx,
                "tasks": [self._task_description, self.VALIDITY_LABEL],
                "length": len(self._action)
            }
            f.write(json.dumps(ep) + "\n")

    def update_task_meta(self):
        print(f"Updating task metadata for episode {self._idx}...")
        task_descriptions = []
        task_json = self.META_DIR / "tasks.jsonl"
        # if task_json.exists():
        #     with open(task_json, "r") as f:
        #         while True:
        #             l = f.readline()
        #             if l == "":
        #                 break
        #             task_descriptions.append(json.loads(l)["task"])

        #         if not self._task_description in task_descriptions:
        #             task_descriptions.append(self._task_description)

        if not (self._task_description in self._tasks):
            self._tasks[self._task_description] = len(self._tasks)

        # === Write meta/tasks.jsonl ===
        with open(task_json, "w") as f:
            for task_description, task_index in self._tasks.items():
                task_entry = {
                    "task_index": task_index,
                    "task": task_description
                }
                f.write(json.dumps(task_entry) + "\n")

    def write_other_meta(self):
        print("Writing other metadata...")

        action_dof = 7
        obs_dof = 7

        # === Write meta/modality.json ===
        modality_config = {
            "state": {
                "qpos": {"start": 0, "end": obs_dof},
                # "qvel": {"start": 0, "end": obs_dof},
                "effort": {"start": 0, "end": obs_dof}
            },
            "action": {
                "qpos": {"start": 0, "end": action_dof},
                # "qvel": {"start": 0, "end": action_dof}
            },
            "video": {
                "left_view": {
                    "original_key": "observation.images.left_view"
                },
                "right_view": {
                    "original_key": "observation.images.right_view"
                },
            },
            "annotation": {
                "human.task_description": {
                    "original_key": "task_index",
                },
            }
                # "human": {
                #     "action": {
                #         "task_description": { "type": "text" },
                #         "validity": { "type": "text" }
                #     }
                # }
        }
        with open(self.META_DIR / "modality.json", "w") as f:
            json.dump(modality_config, f, indent=2)

        # === Write meta/info.json ===
        data_files = self.DATA_DIR.glob("episode_*.parquet")
        number_of_episodes = len(list(data_files))

        info = {
            "codebase_version": "v1.0",
            "robot_type": "UR5e_2F140",
            "total_episodes": number_of_episodes,
            "total_frames": sum([len(pd.read_parquet(f)) for f in data_files]),
            "total_tasks": len(self._tasks),
            "total_videos": number_of_episodes,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.FPS,
            "splits": {
                "train": f"0:{number_of_episodes}",
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.left_view": {
                    "dtype": "video",
                    "shape": self.image_size + [3],  # Replace with actual resolution if different
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.FPS,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.right_view": {
                    "dtype": "video",
                    "shape": self.image_size + [3],  # Replace with actual resolution if different
                    "names": ["height", "width", "channel"],
                    "video_info": {
                        "video.fps": self.FPS,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": [obs_dof],
                    "names": [f"motor_{i}" for i in range(obs_dof)]
                },
                "action": {
                    "dtype": "float64",
                    "shape": [action_dof],
                    "names": [f"motor_{i}" for i in range(action_dof)]
                },
                "timestamp": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "annotation.human.task_description": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "annotation.human.validity": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "next.reward": {
                    "dtype": "float64",
                    "shape": [1]
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1]
                }
            }
        }

        with open(self.META_DIR / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # === Generate meta/stats.json ===
        print("Generating stats.json ...")
        all_parquet_files = sorted(self.DATA_DIR.glob("episode_*.parquet"))

        # Assign task_index based on file order or file name
        all_dfs = []
        for idx, f in enumerate(all_parquet_files):
            df = pd.read_parquet(f)
            df["task_index"] = idx  # You can customize this logic if needed
            all_dfs.append(df)

        df_all = pd.concat(all_dfs, ignore_index=True)

        stats = {}
        for col in df_all.columns:
            try:
                values = np.vstack(df_all[col].values).astype(np.float32)
                stats[col] = {
                    "mean": np.mean(values, axis=0).tolist(),
                    "std": np.std(values, axis=0).tolist(),
                    "min": np.min(values, axis=0).tolist(),
                    "max": np.max(values, axis=0).tolist(),
                    "q01": np.quantile(values, 0.01, axis=0).tolist(),
                    "q99": np.quantile(values, 0.99, axis=0).tolist()
                }
            except Exception as e:
                print(f"⚠️ Skipping column '{col}': {e}")

        self.META_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.META_DIR / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✅ All data and metadata generated at: {self.OUTPUT_BASE.resolve()}")


# def save_as_lerobot_data():
#     TASK_DESCRIPTION = "pick a java curry box"
#     VALIDITY_LABEL = "valid"

#     episodes_meta = []
#     unique_tasks = set()

#     idx = 0
#     qpos, qvel, effort, action, images = prepare_data()
#     num_steps = len(action)

#     save_video(images, idx)
#     save_parquet(qpos, qvel, effort, action, idx)

#     # --- Save metadata info per episode ---
#     episodes_meta.append({
#         "episode_index": idx,
#         "tasks": [TASK_DESCRIPTION, VALIDITY_LABEL],
#         "length": num_steps
#     })

#     unique_tasks.add((TASK_DESCRIPTION, VALIDITY_LABEL))

#     save_meta(episodes_meta, unique_tasks, META_DIR)


# if __name__ == "__main__":
#     save_as_lerobot_data()