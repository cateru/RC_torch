import matplotlib.pyplot as plt


def plot_predictions(model, test_load, device, num_images=5):
    model.eval()  
    images, labels = test_load[0]  
    images = images.unsqueeze(0).to(device) 
    outputs = model(images)  
    prediction = outputs.argmax(dim=1, keepdim=True).item()
    print(f"Predicted: {prediction}, Actual: {labels[0].item()}") 
    picture = images.squeeze(0).squeeze(0).cpu().numpy() 
    plt.imshow(picture, cmap="gray")
    plt.show() 