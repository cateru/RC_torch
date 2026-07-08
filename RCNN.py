from torchesn.nn import Reservoir
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class RCNN(nn.Module):
    def __init__(self, config): 
        super(RCNN, self).__init__() 
        self.reservoir = Reservoir('RES_TANH', config["input_size"], config["reservoir_size"], config["num_layers"], config["leaking_rate"], config["spectral_radius"], config["input_scaling"], config["rc_connectivity"], batch_first = True)
        self.reservoir.requires_grad_(False)
        if config["hidden_sizes"] is None:
            config["hidden_sizes"] = []
        layers = []
        in_features = config["reservoir_size"]
        for h in config["hidden_sizes"]:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            in_features = h
        layers.append(nn.Linear(in_features, 10))
        self.readout = nn.Sequential(*layers)  
        self.scaling_factor = config["scaling_factor"]
        self.lookup_table()

    def tuning_factor(self, time):
        y0 = 62.63
        a = -61.16
        R0 = -0.1518
        percent_decrease = (y0 + a*torch.exp((time*R0))) * self.scaling_factor
        return percent_decrease
    
    def exp_outputs(self, energy):
        a = 0.04558
        b = 0.52236
        c = 0.39704
        o = a + b*torch.pow(energy, c)
        return o
    
    def lookup_table(self): 
        self.time = torch.arange(0.1, 30.01, step = 0.01)
        self.percent_decrease = self.tuning_factor(self.time) 

    def convert_tangent(self, states):
        percent = (states + 1) / 2 
        input_energy = percent * 1000
        return input_energy

    def forward(self, data):
        data = data.squeeze(1)
        reservoir_out, _ = self.reservoir(data)
        state = reservoir_out[:, -10: , :].mean(dim=1)
        input_energy = self.convert_tangent(state)
        response = self.exp_outputs(input_energy)
        return self.readout(response), state, input_energy

    def step_manhattan_differential(self):
        results = []
        with torch.no_grad():
            lookup = self.percent_decrease
            lookup = lookup - torch.min(lookup)
            for module in self.readout:
                if isinstance(module, nn.Linear):
                    for param in [module.weight, module.bias]:
                        if param is None:
                            continue
                        param_flat = param.view(-1) 
                        value = torch.abs(param_flat)
                        sign = torch.sign(param_flat)
                        idx = torch.searchsorted(lookup, value) 
                        idx = torch.clamp(idx, 1, len(lookup) - 1) 
                        left = lookup[idx - 1] 
                        right = lookup[idx]
                        idx_closest = torch.where(torch.abs(param_flat - left) < torch.abs(param_flat - right), idx - 1, idx) 
                        new_param = lookup[idx_closest]
                        param.copy_((sign * new_param).view_as(param))
                        results.append((param.clone(), idx_closest.clone()))
        return results