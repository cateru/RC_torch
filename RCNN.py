from reservoirpy.nodes import Reservoir
import torch
import torch.nn as nn


class RCNN(nn.Module):
    def __init__(self, reservoir_size, hidden_sizes=None): 
        super(RCNN, self).__init__() 
        self.reservoir = Reservoir(reservoir_size, seed=42) 
        if hidden_sizes is None:
            hidden_sizes = []
        layers = []
        in_features = reservoir_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Sigmoid())
            in_features = h
        layers.append(nn.Linear(in_features, 10))
        self.readout = nn.Sequential(*layers)  
        self.lookup_table()

    def tuning_factor(self, time):
        a = -2246.04
        b = 2266.63
        c = 0.00757
        scaling_factor = 0.001
        percent_decrease = (a + b*torch.pow(time, c))*scaling_factor
        return percent_decrease
    
    def exp_outputs(self, energy):
        a = 0.04558
        b = 0.52236
        c = 0.39704
        o = a + b*torch.pow(energy, c)
        return o
    
    def lookup_table(self): 
        self.time = torch.arange(0.30, 200.01, step = 0.01)
        self.percent_decrease = self.tuning_factor(self.time) 

    def convert_tangent(self, states):
        percent = (states + 1) / 2 
        input_energy = percent * 1000 
        return input_energy
        
    def collect_states(self, x):
        inputs = [] 
        for i in range(x.size(0)):
            seq = x[i].squeeze(0).cpu().numpy() 
            s = self.reservoir.run(seq) 
            state = torch.tensor(s[-1]).float()
            input_energy = self.convert_tangent(state)
            inputs.append(input_energy) 
        return torch.stack(inputs), state
    
    def forward(self, data):
        input_energy, state = self.collect_states(data)
        response = self.exp_outputs(input_energy)
        return self.readout(response), state, input_energy

    def update_weights(self):
        with torch.no_grad(): 
            lookup = self.percent_decrease  
            for module in self.readout:
                if isinstance(module, nn.Linear):
                    for param in [module.weight, module.bias]:
                        param_flat = param.view(-1) 
                        idx = torch.searchsorted(lookup, param_flat) 
                        idx = torch.clamp(idx, 1, len(lookup) - 1) 
                        left = lookup[idx - 1] 
                        right = lookup[idx]
                        idx_closest = torch.where(
                            torch.abs(param_flat - left) < torch.abs(param_flat - right),
                            idx - 1,
                            idx
                        ) 
                        param_flat.copy_(lookup[idx_closest])  
        return param, idx_closest
