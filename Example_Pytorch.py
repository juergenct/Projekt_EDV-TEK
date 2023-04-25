import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

# Define the dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_len):
        self.texts = [torch.tensor(tokenizer(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.pad_token = 0
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text = self.pad(text)
        return text, label

    def pad(self, text):
        padded_text = pad_sequence([text], batch_first=True, padding_value=self.pad_token)
        padded_text = padded_text[:, :self.max_seq_len]
        return padded_text.squeeze(0)

# Define the neural network
class TextClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    # Define the Neural Network Architecture - in this case a simple feed-forward network with one hidden layer
    def forward(self, x):
        embedded = self.embedding(x)
        pooled = embedded.mean(dim=1)
        out = self.fc(pooled)
        return out

# Define the training function
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for texts, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define the evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            outputs = model(texts)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# Define the main function
def main():
    # Load the data
    texts_train = ["An apparatus for metering the delivery of an aerosol. The apparatus has a variable acoustic source and a microphone, both acoustically coupled to a volume having a fluid region and an air region. The apparatus may also include a processor to determine a volume of the air region based on signals received from the microphone and the variable acoustic source. A fluid valve is coupled to the processor, and is configured to allow an amount of fluid to exit the fluid region associated with the volume of the air region. An atomizer, coupled to the fluid region, is configured to aerosolize at least a portion of the fluid.", "A data processing system receives location data points from computing devices. The system annotates the location data points with entities and determines a duration each of the computing devices was at corresponding entities. The system aggregates the location data points into a set of sequences based on the duration and the entities and stores the set of sequences in a data record. The system accesses the database record including a set of sequences generated from location data points received from computing devices. The system receives, from a computing device, a request for a location sequence that includes a query. The system identifies an attribute of the computing device. The system identifies a sequence based on the set of sequences using the query and the attribute. The system transmits the sequence for display on a display device.", "A gas turbine engine includes a compressor section and a turbine section arranged in serial flow order. A shaft is provided rotatable with at least a portion of the compressor section and with at least a portion of the turbine section. A bearing is also provided supporting rotation of the shaft, with a support element in turn supporting the bearing. The gas turbine engine also includes a superelastic member formed of a shape memory alloy supporting at least one of the support element or the bearing. The superelastic member is installed in a pre-stressed condition to enhance a dampening function of the superelastic member."]
    labels_train = ["Not Cleantech", "Not Cleantech", "Cleantech"]
    texts_val = ["The present disclosure provides method and system to facilitate definition, tuning and visualization of a geo-fence at a computer system. The method comprises: receiving input parameters for a geo-fence, the input parameters including one or more parameters specifying a geographical region; sampling historical mobile signals based on one or more of the input parameters; dividing the geographical region into a plurality of areas; determining a weight for each respective area of the plurality of areas based at least on density of sampled mobile signals associated with geographical locations in the respective area; selecting a subset of the plurality of areas based on respective weights of the plurality of areas; and forming the geo-fence using the subset of the plurality of areas, the geo-fence including one or more contiguously closed regions each formed by a cluster of adjacent areas among the subset of the plurality of areas."]
    labels_val = ["Not Cleantech"]

    # Tokenize the texts
    tokenizer = get_tokenizer("basic_english")
    max_seq_len = 100

    dataset_train = TextDataset(texts_train, labels_train, tokenizer, max_seq_len)
    dataset_val = TextDataset(texts_val, labels_val, tokenizer, max_seq_len)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

    # Define the model, criterion, and optimizer
    vocab_size = 10000  # Replace with the actual size of your vocabulary
    model = TextClassifier(input_size=vocab_size, hidden_size=256, output_size=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(10):
        train_loss = train(model, dataloader_train, criterion, optimizer)
        val_loss, val_acc = evaluate(model, dataloader_val, criterion)
        print("Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}".format(epoch+1, 10, train_loss, val_loss, val_acc))

if __name__ == "__main__":
    main()