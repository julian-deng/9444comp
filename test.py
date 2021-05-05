
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #self.vocab_size,self.n_hidden,self.n_out = vocab_size, embedding_dim, n_hidden, n_out
        #self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        '''
        self.lstm=tnn.LSTM(
            input_size=50,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
        )
        self.fc1= tnn.Linear(100,64)
        self.fc2= tnn.Linear(64,1)
        self.conv1 = tnn.Sequential(
        tnn.Conv1d(
                in_channels=50,
                out_channels=50,
                kernel_size=8,
                stride=1,
                padding=5,
            ),
            tnn.ReLU(),
            tnn.MaxPool1d(4),
        )

        self.conv2 = tnn.Sequential(
            tnn.Conv1d(in_channels=50,
                out_channels=50,
                kernel_size=8,
                stride=1,
                padding=5,),
            tnn.ReLU(),
            tnn.MaxPool1d(4),
        )
        '''
        self.lstm=tnn.LSTM(
            input_size=50,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
        )
        self.fc1= tnn.Linear(100,64)
        self.fc2= tnn.Linear(64,1)

    def forward(self, input, length):
        '''
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        '''
        r_out, (h_n, h_c) = self.lstm(input,  None)
        out=self.fc1(torch.squeeze(h_n))
        out = torch.relu(out)
        #out = self.fc2(out[:, -1, :])
        out = self.fc2(out)
        out=out.squeeze(1)
        #print(out)
        return out

        '''
        r_out, (h_n, h_c) = self.lstm(input,  None)
        out=self.fc1(torch.squeeze(h_n))
        out = torch.relu(out)
        print(out.shape)
        out = self.fc2(out[:, -1, :])
        #out = self.fc2(out)
        out=out.squeeze(1)
        
        #out = self.fc2(out)
        out=input.permute(0, 2,1)
        out = self.conv1(out)
        out  = self.conv2(out)
        #out  = self.conv3(out)
        goble_pool=tnn.MaxPool1d(kernel_size=out.shape[-1])
        out=goble_pool(out)
        out=out.view(out.shape[0],out.shape[1])

        output = self.out(out )
        output=output.squeeze(1)
        '''



class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch, vocab

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return tnn.BCEWithLogitsLoss()
def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
