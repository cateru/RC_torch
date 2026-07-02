import logging
import torch
import torch.nn as nn
from RCNN import RCNN
from RCNN_data import load_mnist
from RCNN_train import train
from RCNN_eval import test
from RCNN_predict import plot_predictions
from RCNN_load import load_model
from RCNN_log import setup_logger
from RCNN_plot import plot_training_results, plot_readout_weights


def main(mode = "train"):
    config = {
            "epochs": 30,
            "batch_size": 32,
            "reservoir_size": 1000,
            "input_size": 28,
            "num_layers": 1,
            "learning_rate": 0.001,
            "model_path": "model.pth",
            "num_images": 5,
            "hidden_sizes": None, #[64],
            "scaling_factor": 0.05,
            "leaking_rate": 0.2,
            "spectral_radius": 1.3,
            "input_scaling": torch.ones(29),  
            "input_connectivity": 0.1,
            "rc_connectivity": 0.3,
            "bias": 1
        }

    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_load = load_mnist(batch_size=config["batch_size"], train=True) 
    test_load = load_mnist(batch_size=config["batch_size"], train=False)

    loss_fn = nn.CrossEntropyLoss()

    if mode == "train":
        logging.info("Starting training...")
        model = RCNN(config=config).to(device)

        losses = []
        weights = []
        for epoch in range(1, config["epochs"] + 1):
            model, avg_loss, weight = train(epoch, model, train_load, loss_fn, device)  
            losses.append(avg_loss)
            weights.append(weight)

        plot_training_results(losses, weights)
        # plot_readout_weights(model)

        logging.info("Saving model...")
        torch.save(model.state_dict(), config["model_path"])

        logging.info("Training finished.")

    elif mode == "test":

        model = load_model(config, device)
        test(model, test_load, loss_fn, device)

    elif mode == "predict":

        model = load_model(config, device)
        plot_predictions(model, test_load, device, config["num_images"])

if __name__ == "__main__":
    main(mode = "test")        
