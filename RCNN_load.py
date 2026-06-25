import torch
from RCNN import RCNN 

def load_model(config, device):
    torch.set_num_threads(12)
    model = RCNN(config).to(device)
    state_dict = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
