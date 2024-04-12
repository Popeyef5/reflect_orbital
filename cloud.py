import modal
import os
from modal import Image
from settings import DEFAULT_RESULTS_DIR, DROPOUTS
from simulation import simulate

reflect_image = (
  Image.debian_slim()
  .pip_install(["tqdm", "torch", "pandas", "safetensors", "openpyxl"])
)

vol = modal.Volume.from_name("reflect-orbital", create_if_missing=True)

stub = modal.Stub("reflect-orbital")

@stub.function(
  image=reflect_image, gpu='a100', 
  mounts=[modal.Mount.from_local_dir(os.path.join(os.getcwd(), "extra"), remote_path="/root/extra")],
  volumes={'/root/out': vol},
  timeout=10*60*60
)
def run():
  import torch
  from safetensors.torch import save_file

  transmission = None
  cumulative = None
  for dropout in DROPOUTS:
    tmp_transmission, tmp_farms, tmp_cumulative = simulate(dropout=dropout)

    if transmission is None:
      transmission = tmp_transmission[None, :]
      farms = tmp_farms[None, :]
    else:
      transmission = torch.cat((transmission, tmp_transmission[None, :]), dim=0)
      farms = torch.cat((farms, tmp_farms[None, :]), dim=0)

    if cumulative is None:
      cumulative = tmp_cumulative

  save_file({"transmission": transmission, "farms": farms, "cumulative": cumulative}, DEFAULT_RESULTS_DIR)
  vol.commit()

  #return power, farms 


@stub.local_entrypoint()
def main():
  run.remote()
