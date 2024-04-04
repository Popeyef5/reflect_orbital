import modal
import os
from modal import Image
from settings import DEFAULT_RESULTS_DIR, DROPOUTS
from reflect_torch import orbital_efficiency
from safetensors.torch import save_file

reflect_image = (
  Image.debian_slim()
  .pip_install(["tqdm", "torch", "pandas", "safetensors", "wandb", "openpyxl"])
)

stub = modal.Stub("reflect-orbital")

@stub.function(
  image=reflect_image, gpu='t4', 
  mounts=[modal.Mount.from_local_dir(os.path.join(os.getcwd(), "extra"), remote_path="/root/extra")],
  secrets=[modal.Secret.from_name("my-wandb-secret")],
  timeout=60*60
)
def run():
  import wandb
  import os
  import torch
  from safetensors.torch import save_file
  
  wandb_enabled = bool(os.environ.get("WANDB_API_KEY"))
  if wandb_enabled:
    wandb.init(
      id=stub.app_id,
      project="reflect_orbital",
      entity=None,
    )

  power = None
  for dropout in DROPOUTS:
    tmp_power, tmp_farms = orbital_efficiency(dropout=dropout)
    if power is None:
      power = tmp_power[None, :]
      farms = tmp_farms[None, :]
    else:
      power = torch.cat((power, tmp_power[None, :]), dim=0)
      farms = torch.cat((farms, tmp_farms[None, :]), dim=0)

  if wandb_enabled:
    wandb.finish()

  save_file({"power": power, "farms": farms}, DEFAULT_RESULTS_DIR)

  #return power, farms 


@stub.local_entrypoint()
def main():
  run.remote()
