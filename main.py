import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_collection.process import *
from data_collection.midi import *

import os

class AudioDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        spectrogram = process_one_shot(filepath)
        
        return spectrogram, torch.tensor(label, dtype=torch.long)

class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1_input_size = 128 * (257 // 8) * (8 // 8)
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc1_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')   

# HIHAT = 'dataset/one_shots/beatbox/hh/'
# SNARE = 'dataset/one_shots/beatbox/snare/'
# KICK = 'dataset/one_shots/beatbox/kick/'

# all_hihats = {os.path.join(HIHAT, f):0 for f in os.listdir(HIHAT) if os.path.isfile(os.path.join(HIHAT, f)) and f.lower().endswith(('.wav'))}
# all_snares = {os.path.join(SNARE, f):1 for f in os.listdir(SNARE) if os.path.isfile(os.path.join(SNARE, f)) and f.lower().endswith(('.wav'))}
# all_kicks = {os.path.join(KICK, f):2 for f in os.listdir(KICK) if os.path.isfile(os.path.join(KICK, f)) and f.lower().endswith(('.wav'))}
# all_sounds = all_hihats | all_snares | all_kicks

# dataset = AudioDataset(filepaths=list(all_sounds.keys()), labels=list(all_sounds.values()))
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# # first_batch = next(iter(train_dataloader))
# # feature, label = first_batch

# model = SpectrogramCNN()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train_model(model, train_dataloader, criterion, optimizer)
# evaluate_model(model, test_dataloader)

# torch.save(model.state_dict(), 'model.pth')

# # model = SpectrogramCNN()
# # model.load_state_dict(torch.load('model/model.pth'))

# a = process_full_loop('./dataset/full_loops/beatbox/1131.wav')

# # fig, axs = plt.subplots(8, 1)

# note_times = []

# for index, i in enumerate(a):
#     time, spectrogram = i
#     print(time)
#     outputs = model(spectrogram)
#     _, predicted = torch.max(outputs, 1)
#     note_times.append((int(predicted[0].item()), time))
#     # plot_spectrogram(a[index][0], ylabel="Original", ax=axs[index])

# print(note_times)
# # fig.tight_layout()
# # plt.show()

# # note_times_to_midi(note_times, 120)
