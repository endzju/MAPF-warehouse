from pathlib import Path

import torch
import torch.nn as nn


def load_model(model_path: str, model_class: nn.Module):
    device = "cpu"
    view_size = int(model_path.split("_")[-1].split(".")[0])
    vshape = (4, view_size, view_size)
    model = model_class(view_shape=vshape, goal_vec_size=2, n_actions=5).to(device)
    script_path = Path(__file__).parent
    path = script_path.parent / "neural_networks" / "models" / model_path
    weights_dict = torch.load(path, map_location=device)
    model.load_state_dict(weights_dict)
    return model
