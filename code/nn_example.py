import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.model import Net
from data_loader.data_module import load_iris_data, BATCH_SIZE
from utils.metrics_reporter import plot_metrics, generate_report


CODEOCEAN_BASE_DIR = os.environ.get('CODEOCEAN_BASE_DIR', '/')

RESULTS_PATH = os.path.join(CODEOCEAN_BASE_DIR, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'iris_nn_model.pth')
REPORT_SAVE_PATH = os.path.join(RESULTS_PATH, 'analysis_report.txt')
TRAINING_PLOT_PATH = os.path.join(RESULTS_PATH, 'training_metrics.png')

NUM_EPOCHS = 50 
LEARNING_RATE = 0.01 



def train_model(net, trainloader, testloader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    train_losses = []
    test_accuracies = []

    for epoch in range(NUM_EPOCHS):
        net.train() 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.3f}')
        
        
        net.eval() 
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        print(f'Accuracy: {accuracy:.2f} %')

    print('Finished Training')
    return train_losses, test_accuracies


if __name__ == '__main__':

    DATA_PATH = os.path.join(CODEOCEAN_BASE_DIR, 'data') 
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainloader, testloader, classes, input_size, num_classes = load_iris_data()

    net = Net(input_size, num_classes).to(device)

    train_losses, test_accuracies = train_model(net, trainloader, testloader, device)

    print('Saving model...')
    torch.save(net.state_dict(), MODEL_SAVE_PATH)

    plot_metrics(train_losses, test_accuracies)

    generate_report(train_losses, test_accuracies, classes, MODEL_SAVE_PATH)

    print("Neural network example finished successfully with tabular data!")
