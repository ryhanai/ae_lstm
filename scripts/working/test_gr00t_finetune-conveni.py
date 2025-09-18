from gr00t.utils.eval import calc_mse_for_single_trajectory
import warnings

from working.data_config import UR5eDataConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset
import numpy as np
import torch
import gr00t
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


device = "cuda" if torch.cuda.is_available() else "cpu"

warnings.simplefilter("ignore", category=FutureWarning)

PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
EMBODIMENT_TAG = EmbodimentTag.NEW_EMBODIMENT

REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = str(Path.home() / "Downloads" / "conveni_gr00t")


# data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
# modality_config = data_config.modality_config()
# modality_transform = data_config.transform()


data_config = UR5eDataConfig()


# pre_trained_policy = Gr00tPolicy(
#     model_path=PRE_TRAINED_MODEL_PATH,
#     embodiment_tag=EMBODIMENT_TAG,
#     modality_config=data_config.modality_config(),
#     modality_transform=data_config.transform(),
#     device=device,
# )

train_dataset = LeRobotSingleDataset(
    dataset_path=DATASET_PATH,
    modality_configs=data_config.modality_config(),
    # video_backend="decord",
    video_backend="torchvision_av",    
    video_backend_kwargs=None,
    transforms=data_config.transform(),
    embodiment_tag=EMBODIMENT_TAG,
)


# mse = calc_mse_for_single_trajectory(
#     pre_trained_policy,
#     dataset,
#     traj_id=0,
#     modality_keys=["qpos"],   # we will only evaluate the right arm and right hand
#     steps=150,
#     action_horizon=16,
#     plot=True
# )

# print("MSE loss for trajectory 0:", mse)



from gr00t.model.gr00t_n1 import GR00T_N1_5

BASE_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
TUNE_LLM = False            # Whether to tune the LLM
TUNE_VISUAL = False          # Whether to tune the visual encoder
TUNE_PROJECTOR = True       # Whether to tune the projector
TUNE_DIFFUSION_MODEL = True # Whether to tune the diffusion model

model = GR00T_N1_5.from_pretrained(
    pretrained_model_name_or_path=BASE_MODEL_PATH,
    tune_llm=TUNE_LLM,  # backbone's LLM
    tune_visual=TUNE_VISUAL,  # backbone's vision tower
    tune_projector=TUNE_PROJECTOR,  # action head's projector
    tune_diffusion_model=TUNE_DIFFUSION_MODEL,  # action head's DiT
)

# Set the model's compute_dtype to bfloat16
model.compute_dtype = "bfloat16"
model.config.compute_dtype = "bfloat16"
model.to(device)


from transformers import TrainingArguments

output_dir = "/data2/SB_gr00t/model/path"    # CHANGE THIS ACCORDING TO YOUR LOCAL PATH
per_device_train_batch_size = 8     # CHANGE THIS ACCORDING TO YOUR GPU MEMORY
max_steps = 10000                      # CHANGE THIS ACCORDING TO YOUR NEEDS
report_to = "wandb"
dataloader_num_workers = 1          # 8 by default

training_args = TrainingArguments(
    output_dir=output_dir,
    run_name=None,
    remove_unused_columns=False,
    deepspeed="",
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=1,
    dataloader_num_workers=dataloader_num_workers,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=True,
    optim="adamw_torch",
    adam_beta1=0.95,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    learning_rate=1e-4,
    weight_decay=1e-5,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10.0,
    num_train_epochs=300,
    max_steps=max_steps,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=8,
    report_to=report_to,
    seed=42,
    do_eval=False,
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=100,
    torch_compile_mode=None,
)


from gr00t.experiment.runner import TrainRunner

experiment = TrainRunner(
    train_dataset=train_dataset,
    model=model,
    training_args=training_args,
    resume_from_checkpoint=False,
)

experiment.train()