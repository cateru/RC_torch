import logging
from xml.parsers.expat import model


def train(epoch, model, train_load, optimizer, loss_fn, device):
    model.train()
    total_loss = 0  
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_load):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output, state, input_energy = model(data) 
        predicted = output.argmax(dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)
        loss = loss_fn(output, target)
        loss.backward() 
        optimizer.step()
        # if batch_idx % 20 == 0:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad and param.grad is not None:
        #             logging.info(f"[{name}] Weights sample: {param.data.view(-1)[:5].detach().cpu().numpy()}...")
        #             logging.info(f"[{name}] Gradients sample: {param.grad.view(-1)[:5].detach().cpu().numpy()}...")
        results = model.update_weights()
        total_loss += loss.item() 
        if batch_idx % 20 == 0: 
            logging.info(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_load.dataset)} ({100. * batch_idx / len(train_load):.0f}%)]\tLoss: {loss.item():.6f}")
            # logging.info(f"Readout weights: {results[0][0].view(-1)[:5].detach().numpy()}...")
            # logging.info(f"Readout closest indices: {results[0][1][:5].detach().numpy()}...")
            # logging.info(f"Readout input energy: {input_energy[:5, :1].detach().numpy()}...")
            # logging.info(f"Reservoir state: {state[:5].detach().numpy()}...")
    avg_loss = total_loss / len(train_load) 
    logging.info(f"Average Loss per Epoch {epoch}: {avg_loss:.6f}") 
    accuracy = 100 * correct / total
    logging.info(f"Average Accuracy: {accuracy:.2f}%")
    weights = model.readout[0].weight.detach().cpu()
    return model, avg_loss, accuracy, weights