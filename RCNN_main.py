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
from RCNN_plot import plot_training_results


def main(mode = "train"):
    config = {
        "epochs": 10,
        "batch_size": 32,
        "reservoir_size": 300,
        "learning_rate": 0.001,
        "model_path": "model.pth",
        "num_images": 5,
        "hidden_sizes": None #[128, 64],
    }

    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_load = load_mnist(batch_size=config["batch_size"], train=True) 
    test_load = load_mnist(batch_size=config["batch_size"], train=False)

    loss_fn = nn.CrossEntropyLoss()

    if mode == "train":
        logging.info("Starting training...")
        model = RCNN(reservoir_size=config["reservoir_size"], hidden_sizes=config["hidden_sizes"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        losses = []
        weights = []
        for epoch in range(1, config["epochs"] + 1):
            model, avg_loss, weight = train(epoch, model, train_load, optimizer, loss_fn, device)
            logging.info(f"First five weights of epoch {epoch}: {weight[:5].numpy()}...")  
            losses.append(avg_loss)
            weights.append(weight)

        plot_training_results(losses, weights)

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
