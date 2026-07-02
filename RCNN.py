from torchesn.nn import Reservoir
import torch
import torch.nn as nn


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
            layers.append(nn.LeakyReLU())
            in_features = h
        layers.append(nn.Linear(in_features, 10))
        self.readout = nn.Sequential(*layers)  
        self.scaling_factor = config["scaling_factor"]
        self.lookup_table()
        self.add_differential_indices()
        self.initialize_from_lookup_manhattan()

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
    
    def add_differential_indices(self):
        for module in self.readout:
            if isinstance(module, nn.Linear):
                module.weight_pos_idx = torch.zeros_like(module.weight, dtype=torch.long)
                module.weight_neg_idx = torch.zeros_like(module.weight, dtype=torch.long)
                if module.bias is not None:
                    module.bias_pos_idx = torch.zeros_like(module.bias, dtype=torch.long)
                    module.bias_neg_idx = torch.zeros_like(module.bias, dtype=torch.long)

    def initialize_from_lookup_manhattan(self):
        with torch.no_grad():
            lookup = self.percent_decrease
            max_init = len(lookup) // 100
            for module in self.readout:
                if isinstance(module, nn.Linear): 
                    for param, idx_pos, idx_neg in [(module.weight, module.weight_pos_idx, module.weight_neg_idx), (module.bias, module.bias_pos_idx, module.bias_neg_idx)]:                 
                        if param is None:
                            continue
                        idx_pos.random_(0, max_init)
                        idx_neg.random_(0, max_init)
                        # param.copy_()
                        param.copy_(lookup[idx_pos] - lookup[idx_neg])
        
    def step_manhattan_differential(self, learning_rate=1e-3, max_step=5):
        results = []
        with torch.no_grad():
            lookup = self.percent_decrease
            max_idx = len(lookup) - 1
            for module in self.readout:
                if not isinstance(module, nn.Linear):
                    continue
                params = [(module.weight, module.weight_pos_idx, module.weight_neg_idx,),]
                if module.bias is not None:
                    params.append((module.bias, module.bias_pos_idx, module.bias_neg_idx,))
                for param, idx_pos, idx_neg in params:
                    if param.grad is None:
                        continue
                    grad = param.grad.view(-1)
                    idx_pos = idx_pos.view(-1)
                    idx_neg = idx_neg.view(-1)
                    # step = torch.ones_like(grad)
                    step = grad.abs()
                    step = step / (step.mean() + 1e-8)
                    step = (step * max_step).long()
                    step = torch.clamp(step, 1, max_step)
                    # grad_sign = grad.sign()
                    step = step.long()
                    # pos_mask = grad_sign < 0
                    # neg_mask = grad_sign > 0
                    # idx_pos[pos_mask] += step[pos_mask]
                    # idx_neg[neg_mask] += step[neg_mask]
                    idx_pos[grad < 0] += step[grad < 0]
                    idx_neg[grad < 0] -= step[grad < 0]
                    idx_pos[grad > 0] -= step[grad > 0]
                    idx_neg[grad > 0] += step[grad > 0]
                    idx_pos = torch.clamp(idx_pos, 0, max_idx)
                    idx_neg = torch.clamp(idx_neg, 0, max_idx)
                    param.copy_(lookup[idx_pos].view_as(param) - lookup[idx_neg].view_as(param))
                    results.append((param.clone(), idx_pos.clone(), idx_neg.clone(),))
        return results
