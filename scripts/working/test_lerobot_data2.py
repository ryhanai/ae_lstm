import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


task_name = "pick_java_curry_box"
image_size = [120, 160]

OUTPUT_BASE = Path(f"./Downloads/{task_name}_gr00t/")
VIDEO_DIR = OUTPUT_BASE / "videos/chunk-000/observation.images.ego_view"
DATA_DIR = OUTPUT_BASE / "data/chunk-000"
META_DIR = OUTPUT_BASE / "meta"
FPS = 30.0

# Create necessary directories
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)


# === Process all episodes ===
# episode_files = sorted(INPUT_DIR.glob("episode_*.hdf5"))


def prepare_data():
    qpos = np.zeros((1, 7))
    qvel = np.zeros((1, 7))
    effort = np.zeros((1, 7))
    action = np.zeros((1, 7))
    images = np.zeros([1] + image_size + [3], dtype=np.uint8)
    return qpos, qvel, effort, action, images


def save_video(images, idx):
    episode_id = f"episode_{idx:06d}"
    print(f"Processing {episode_id}...")
    video_path = VIDEO_DIR / f"{episode_id}.mp4"

    qpos, qvel, effort, action, images = prepare_data()

    h, w, _ = images[0].shape
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    for img in images:
        out.write(img)
    out.release()


def save_parquet(qpos, qvel, effort, action, idx):
    episode_id = f"episode_{idx:06d}"
    records = []
    num_steps = len(action)

    for t in range(num_steps):
        state = np.concatenate([qpos[t], qvel[t], effort[t]])
        
        records.append({
            "observation.state": state.tolist(),
            "action": action[t].tolist(),
            "timestamp": float(t),
            "frame_index": t,
            "episode_index": idx,
            "index": idx * 10000 + t,
            "task_index": 0
        })
        
    df = pd.DataFrame(records)
    pq.write_table(pa.Table.from_pandas(df), DATA_DIR / f"{episode_id}.parquet")


def save_meta(episodes_meta, unique_tasks, META_DIR):
    # === Write meta/episodes.jsonl ===
    with open(META_DIR / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    # === Write meta/tasks.jsonl ===
    with open(META_DIR / "tasks.jsonl", "w") as f:
        for task_index, (task_description, _) in enumerate(sorted(unique_tasks)):
            task_entry = {
                "task_index": task_index,
                "task": task_description
            }
            f.write(json.dumps(task_entry) + "\n")

    # === Write meta/modality.json ===
    modality_config = {
        "state": {
            "qpos": {"start": 0, "end": 14},
            "qvel": {"start": 0, "end": 14},
            "effort": {"start": 0, "end": 14}
        },
        "action": {
            "qpos": {"start": 0, "end": 14},
            "qvel": {"start": 0, "end": 14}
        },
        "video": {
            "ego_view": {
                "original_key": "observation.images.ego_view"
            }
        },
        "annotation": {
            "human": {
                "action": {
                    "task_description": { "type": "text" },
                    "validity": { "type": "text" }
                }
            }
        }
    }
    with open(META_DIR / "modality.json", "w") as f:
        json.dump(modality_config, f, indent=2)

    # === Write meta/info.json ===
    number_of_episodes = len(episodes_meta)

    info = {
        "codebase_version": "v1.0",
        "robot_type": "UR5e_2F140",
        "total_episodes": number_of_episodes,
        "total_frames": sum([len(pd.read_parquet(f)) for f in DATA_DIR.glob("episode_*.parquet")]),
        "total_tasks": 1,
        "total_videos": number_of_episodes,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30.0,
        "splits": {
            "train": f"0:{number_of_episodes}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": image_size + [3],  # Replace with actual resolution if different
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 30.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.egoview": {
                "dtype": "video",
                "shape": image_size + [3],  # Replace with actual resolution if different
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": 30.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.state": {
                "dtype": "float64",
                "shape": [21],
                "names": [f"motor_{i}" for i in range(21)]
            },
            "action": {
                "dtype": "float64",
                "shape": [7],
                "names": [f"motor_{i}" for i in range(14)]
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [1]
            },
            "annotation.human.action.task_description": {
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

    with open(META_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # === Generate meta/stats.json ===
    print("Generating stats.json ...")
    all_parquet_files = sorted(DATA_DIR.glob("episode_*.parquet"))

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

    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"✅ All data and metadata generated at: {OUTPUT_BASE.resolve()}")


def save_as_lerobot_data():
    TASK_DESCRIPTION = "pick a java curry box"
    VALIDITY_LABEL = "valid"

    episodes_meta = []
    unique_tasks = set()

    idx = 0
    qpos, qvel, effort, action, images = prepare_data()
    num_steps = len(action)

    save_video(images, idx)
    save_parquet(qpos, qvel, effort, action, idx)

    # --- Save metadata info per episode ---
    episodes_meta.append({
        "episode_index": idx,
        "tasks": [TASK_DESCRIPTION, VALIDITY_LABEL],
        "length": num_steps
    })

    unique_tasks.add((TASK_DESCRIPTION, VALIDITY_LABEL))

    save_meta(episodes_meta, unique_tasks, META_DIR)


if __name__ == "__main__":
    save_as_lerobot_data()