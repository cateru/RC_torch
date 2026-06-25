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
        self.initialize_from_lookup()

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
        state = reservoir_out[:, -10:, :].mean(dim=1) 
        input_energy = self.convert_tangent(state)
        response = self.exp_outputs(input_energy)
        return self.readout(response), state, input_energy
    
    def initialize_from_lookup(self):
        with torch.no_grad():
            lookup = self.percent_decrease
            for module in self.readout:
                if isinstance(module, nn.Linear):
                    for param in [module.weight, module.bias]:
                        if param is None:
                            continue
                        flat = param.view(-1)
                        idx = torch.randint(0, len(lookup)//10, flat.shape)
                        flat.copy_(lookup[idx])
        
    def step_manhattan(self, learning_rate=1e-3, max_step=5):
        results = []
        with torch.no_grad():
            lookup = self.percent_decrease 
            for module in self.readout:
                if isinstance(module, nn.Linear):
                    for param in [module.weight, module.bias]:
                        if param is None or param.grad is None:
                            continue
                        flat = param.view(-1)
                        grad = param.grad.view(-1)
                        idx = torch.searchsorted(lookup, flat)
                        idx = torch.clamp(idx, 1, len(lookup) - 1)
                        grad_sign = grad.sign()
                        step = grad.abs()
                        step = step / (step.mean() + 1e-8)
                        step = (step * max_step).long()
                        step = torch.clamp(step, 1, max_step)
                        idx = idx - grad_sign.long() #* step #+ torch.randint(-1, 2, idx.shape)
                        idx = torch.clamp(idx, 0, len(lookup) - 1)
                        new_param = lookup[idx]
                        flat.copy_(new_param)
                        results.append((new_param.clone(), idx.clone()))
        return results
