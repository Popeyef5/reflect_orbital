import modal
import os
from modal import Image
from settings import DEFAULT_RESULTS_DIR, DROPOUTS
from reflect_torch import orbital_efficiency

reflect_image = (
  Image.debian_slim()
  .pip_install(["tqdm", "torch", "pandas", "safetensors", "openpyxl"])
)

vol = modal.Volume.from_name("reflect-orbital", create_if_missing=True)

stub = modal.Stub("reflect-orbital")

@stub.function(
  image=reflect_image, gpu='t4', 
  mounts=[modal.Mount.from_local_dir(os.path.join(os.getcwd(), "extra"), remote_path="/root/extra")],
  volumes={'/root/out': vol},
  timeout=60*60
)
def run():
  import torch
  from safetensors.torch import save_file

  power = None
  for dropout in DROPOUTS:
    tmp_power, tmp_farms = orbital_efficiency(dropout=dropout)
    if power is None:
      power = tmp_power[None, :]
      farms = tmp_farms[None, :]
    else:
      power = torch.cat((power, tmp_power[None, :]), dim=0)
      farms = torch.cat((farms, tmp_farms[None, :]), dim=0)

  save_file({"power": power, "farms": farms}, DEFAULT_RESULTS_DIR)
  vol.commit()

  #return power, farms 


@stub.local_entrypoint()
def main():
  run.remote()
