import logging


def train(epoch, model, train_load, loss_fn, device):
    model.train()
    total_loss = 0  
    for batch_idx, (data, target) in enumerate(train_load):
        data, target = data.to(device), target.to(device) 
        model.zero_grad()
        output, state, input_energy = model(data) 
        loss = loss_fn(output, target)
        loss.backward() 
        # if batch_idx % 20 == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             logging.info(f"[{name}] Gradients sample: {param.grad.view(-1)[:5].detach().cpu().numpy()}...")
        model.step_manhattan()
        total_loss += loss.item() 
        if batch_idx % 20 == 0: 
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_load.dataset)} ({100. * batch_idx / len(train_load):.0f}%)]\tLoss: {loss.item():.6f}")
            # logging.info(f"Readout weights: {results[0][0].view(-1)[:5].detach().numpy()}...")
            # logging.info(f"Readout closest indices: {results[0][1][:5].detach().numpy()}...")
            # logging.info(f"Readout input energy: {input_energy[:5, :1].detach().numpy()}...")
            # logging.info(f"Reservoir state: {state[:5].detach().numpy()}...")
    avg_loss = total_loss / len(train_load) 
    logging.info(f"Average Loss per Epoch {epoch}: {avg_loss:.6f}") 
    weights = model.readout[0].weight.detach().cpu().view(-1)
    return model, avg_loss, weights