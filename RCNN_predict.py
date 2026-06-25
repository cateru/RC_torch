import matplotlib.pyplot as plt
import torch


def plot_predictions(model, test_load, device, num_images=5):
    model.eval()
    with torch.no_grad():
        for images, labels in test_load:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1)
            for i in range(min(num_images, images.size(0))):
                print(f"Predicted: {preds[i].item()}, Actual: {labels[i].item()}")
                img = images[i].cpu().squeeze()
                plt.imshow(img, cmap="gray")
                plt.show()
            break  