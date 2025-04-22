from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from ramphy import Hyperparameter
from sklearn.model_selection import train_test_split

# RAMP START HYPERPARAMETERS
hidden_size_1 = Hyperparameter(dtype='int', default=128, values=[32, 64, 128, 256, 512, 1024])
hidden_size_2 = Hyperparameter(dtype='int', default=64, values=[16, 32, 64, 128, 256, 512])
activation = Hyperparameter(dtype='str', default='relu', values=['relu', 'leaky_relu', 'elu', 'gelu', 'tanh'])
dropout_rate = Hyperparameter(dtype='float', default=0.1, values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
batch_size = Hyperparameter(dtype='int', default=128, values=[128, 256, 512, 1024])
learning_rate = Hyperparameter(dtype='float', default=0.001, values=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1])
weight_decay = Hyperparameter(dtype='float', default=1e-5, values=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
n_epochs = Hyperparameter(dtype='int', default=3, values=[3, 5, 10, 20, 30, 50])
optimizer_name = Hyperparameter(dtype='str', default='adam', values=['adam', 'sgd', 'adamw', 'rmsprop'])
scheduler_type = Hyperparameter(dtype='str', default='none', values=['none', 'step', 'cosine', 'plateau'])
# RAMP END HYPERPARAMETERS
HIDDEN_SIZE_1 = int(hidden_size_1)
HIDDEN_SIZE_2 = int(hidden_size_2)
ACTIVATION = str(activation)
DROPOUT_RATE = float(dropout_rate)
BATCH_SIZE = int(batch_size)
LEARNING_RATE = float(learning_rate)
WEIGHT_DECAY = float(weight_decay)
NUM_EPOCHS = int(n_epochs)
OPTIMIZER_NAME = str(optimizer_name)
SCHEDULER_TYPE = str(scheduler_type)


class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        # Convert to torch tensor
        self.X = torch.tensor(X, dtype=torch.float32)

        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class PyTorchTabularModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size_1, hidden_size_2, activation, dropout_rate):
        super(PyTorchTabularModel, self).__init__()

        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default

        # Define layers
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_size_1),
            nn.BatchNorm1d(hidden_size_1),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.BatchNorm1d(hidden_size_2),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        self.output_layer = nn.Linear(hidden_size_2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x

    def fit(self, X, y):
        # Create a validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalize input features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Convert to numpy arrays
        X_train_np = np.array(X_train_scaled, dtype=np.float32)
        y_train_np = np.array(y_train, dtype=np.int64).ravel()
        X_val_np = np.array(X_val_scaled, dtype=np.float32)
        y_val_np = np.array(y_val, dtype=np.int64).ravel()

        # Print some values to verify data
        print(
            f"Target values distribution: {{np.unique(y_train_np, return_counts=True)}}"
        )

        # Print model architecture
        print(f"Model architecture: {{self}}")

        # Create dataset and dataloader
        train_dataset = TabularDataset(X_train_np, y_train_np)
        val_dataset = TabularDataset(X_val_np, y_val_np)

        train_loader = DataLoader(
            train_dataset, batch_size=min(BATCH_SIZE, len(X_train_np)), shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=min(BATCH_SIZE, len(X_val_np)))

        # Loss function - always use CrossEntropyLoss for classification
        # PyTorch's CrossEntropyLoss works for both binary and multiclass
        criterion = nn.CrossEntropyLoss()

        # Optimizer
        if OPTIMIZER_NAME == "adam":
            optimizer = optim.Adam(
                self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_NAME == "adamw":
            optimizer = optim.AdamW(
                self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        elif OPTIMIZER_NAME == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                momentum=0.9,
            )
        elif OPTIMIZER_NAME == "rmsprop":
            optimizer = optim.RMSprop(
                self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        else:
            optimizer = optim.Adam(
                self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )

        # Learning rate scheduler
        if SCHEDULER_TYPE == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif SCHEDULER_TYPE == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS
            )
        elif SCHEDULER_TYPE == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=10
            )
        else:
            scheduler = None

        # Early stopping variables
        best_val_loss = float("inf")
        patience = 15
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(NUM_EPOCHS):
            # Training phase
            self.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            train_acc = correct_train / total_train

            # Validation phase
            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Forward pass
                    outputs = self.forward(inputs)

                    # Calculate loss
                    loss = criterion(outputs, labels)

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            val_acc = correct_val / total_val

            # Print progress
            if epoch % 1 == 0:  # Print every epoch since we have fewer epochs now
                print(
                    f"Epoch {{epoch}}: Train Loss: {{train_loss:.4f}}, Train Acc: {{train_acc:.4f}}, Val Loss: {{val_loss:.4f}}, Val Acc: {{val_acc:.4f}}"
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {{epoch}}")
                    break

            # Update learning rate if scheduler is set
            if scheduler is not None:
                if SCHEDULER_TYPE == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # Load the best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            print("Loaded best model from early stopping")

        # Final evaluation
        self.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Final Training Accuracy: {{100 * correct / total:.2f}}%")

    def predict_proba(self, X):
        # Convert to numpy and then to PyTorch tensor
        X_np = np.array(X, dtype=np.float32)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)

        # Set model to evaluation mode
        self.eval()

        # Forward pass with no gradient calculation
        with torch.no_grad():
            logits = self.forward(X_tensor)

        # Apply softmax for class probabilities
        probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()


class Classifier(BaseEstimator):
    def __init__(self, metadata):
        self.metadata = metadata
        target_cols = metadata["data_description"]["target_cols"]
        if len(target_cols) > 1:
            raise NotImplementedError("Multi-output classification is not yet supported.")
        target_value_dict = metadata["data_description"]["target_values"]
        target_values = target_value_dict[target_cols[0]]

        self.num_classes = len(target_values)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {{self.device}}")

        # Store for later use
        self.target_values = target_values

        # Initialize scaler for normalization
        self.scaler = StandardScaler()

        self.model: Optional[PyTorchTabularModel] = None

    def fit(self, X, y):
        # Print shapes for debugging
        print(f"X shape: {{X.shape}}, y shape: {{y.shape}}")
        print(f"Number of classes: {{self.num_classes}}")

        # Prepare data
        input_dim = X.shape[1]
        output_dim = self.num_classes

        # Create model
        self.model = PyTorchTabularModel(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size_1=HIDDEN_SIZE_1,
            hidden_size_2=HIDDEN_SIZE_2,
            activation=ACTIVATION,
            dropout_rate=DROPOUT_RATE,
        ).to(self.device)

        self.model.fit(X, y)

    def predict_proba(self, X):
        # First normalize the input data
        X_scaled = self.scaler.transform(X)
        y_proba = self.model.predict_proba(X_scaled)
        return y_proba