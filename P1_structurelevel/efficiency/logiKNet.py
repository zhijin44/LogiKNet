import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import torch
from kan import KAN
import time
import warnings
warnings.filterwarnings(
    "ignore",
    "CUDA initialization: Unexpected error from cudaGetDeviceCount"
)


################################setup######################################
def load_csv_data(input_folder: str,
                  train_fname: str,
                  test_fname: str):
    """
    Reads train & test CSVs from disk.
    
    Returns:
      train_df, test_df (both pandas.DataFrame)
    """
    train_path = os.path.join(input_folder, train_fname)
    test_path  = os.path.join(input_folder, test_fname)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df


def extract_features_labels(df: pd.DataFrame):
    """
    Splits a DataFrame into numpy feature array X and label vector y.
    
    The last column is the label.
    """
    X = df.iloc[:, :-1].values
    y = df.iloc[:,  -1].values
    return X, y

# this is a standard PyTorch DataLoader to load the dataset for the training and testing of the model
class DataLoader(object):
    def __init__(self,
                 data,
                 labels,
                 batch_size=64,
                 shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(self.data.shape[0] / self.batch_size))

    def __iter__(self):
        n = self.data.shape[0]
        idxlist = list(range(n))
        if self.shuffle:
            np.random.shuffle(idxlist)

        for _, start_idx in enumerate(range(0, n, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n)
            data = self.data[idxlist[start_idx:end_idx]]
            labels = self.labels[idxlist[start_idx:end_idx]]
            ############################################################
            # Check if any class is missing in the batch
            # present_classes = np.unique(labels.cpu().numpy())
            # all_classes = np.arange(len(label_mapping))  # Adjust based on number of classes
            # missing_classes = set(all_classes) - set(present_classes)
            #
            # if missing_classes:
            #     print(f"Batch {start_idx // self.batch_size} is missing classes {missing_classes}")
            ############################################################
            yield data, labels


class LogitsToPredicate(torch.nn.Module):
    """
    This model has inside a logits model, that is a model which compute logits for the classes given an input example x.
    The idea of this model is to keep logits and probabilities separated. The logits model returns the logits for an example,
    while this model returns the probabilities given the logits model.

    In particular, it takes as input an example x and a class label l. It applies the logits model to x to get the logits.
    Then, it applies a softmax function to get the probabilities per classes. Finally, it returns only the probability related
    to the given class l.
    """

    def __init__(self, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l, training=False):
        logits = self.logits_model(x, training=training)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)  # 计算并返回与给定类标签l对应的概率值
        return out


class MLP(torch.nn.Module):
    """
    This model returns the logits for the classes given an input example. It does not compute the softmax, so the output
    are not normalized.
    This is done to separate the accuracy computation from the satisfaction level computation. Go through the example
    to understand it.
    """

    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        self.elu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.2)
        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                                  for i in range(1, len(layer_sizes))])

    def forward(self, x, training=False):
        """
        Method which defines the forward phase of the neural network for our multi class classification task.
        In particular, it returns the logits for the classes given an input example.

        :param x: the features of the example
        :param training: whether the network is in training mode (dropout applied) or validation mode (dropout not applied)
        :return: logits for example x
        """
        for layer in self.linear_layers[:-1]:
            x = self.elu(layer(x))
            if training:
                x = self.dropout(x)
        logits = self.linear_layers[-1](x)
        return logits


class MultiKANModel(torch.nn.Module):
    def __init__(self, kan):
        """
        Wrap an already built MultKAN instance.
        Args:
            kan: a MultKAN model (which has attributes such as act_fun, symbolic_fun, node_bias, node_scale,
                 subnode_bias, subnode_scale, depth, width, mult_homo, mult_arity, input_id, symbolic_enabled, etc.)
        """
        super(MultiKANModel, self).__init__()
        self.kan = kan

    def forward(self, x, training=False, singularity_avoiding=False, y_th=10.):
        # Select input features according to input_id
        x = x[:, self.kan.input_id.long()]
        # Loop through each layer
        for l in range(self.kan.depth):
            # Get outputs from the numerical branch (KANLayer) of current layer
            x_numerical, preacts, postacts_numerical, postspline = self.kan.act_fun[l](x)
            # Get output from the symbolic branch if enabled
            if self.kan.symbolic_enabled:
                x_symbolic, postacts_symbolic = self.kan.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
            # Sum the numerical and symbolic outputs
            x = x_numerical + x_symbolic

            # Subnode affine transformation
            x = self.kan.subnode_scale[l][None, :] * x + self.kan.subnode_bias[l][None, :]

            # Process multiplication nodes
            dim_sum = self.kan.width[l+1][0]
            dim_mult = self.kan.width[l+1][1]
            if dim_mult > 0:
                if self.kan.mult_homo:
                    for i in range(self.kan.mult_arity-1):
                        if i == 0:
                            x_mult = x[:, dim_sum::self.kan.mult_arity] * x[:, dim_sum+1::self.kan.mult_arity]
                        else:
                            x_mult = x_mult * x[:, dim_sum+i+1::self.kan.mult_arity]
                else:
                    for j in range(dim_mult):
                        acml_id = dim_sum + int(np.sum(self.kan.mult_arity[l+1][:j]))
                        for i in range(self.kan.mult_arity[l+1][j]-1):
                            if i == 0:
                                x_mult_j = x[:, [acml_id]] * x[:, [acml_id+1]]
                            else:
                                x_mult_j = x_mult_j * x[:, [acml_id+i+1]]
                        if j == 0:
                            x_mult = x_mult_j
                        else:
                            x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                # Concatenate sum and mult parts
                x = torch.cat([x[:, :dim_sum], x_mult], dim=1)

            # Node affine transformation
            x = self.kan.node_scale[l][None, :] * x + self.kan.node_bias[l][None, :]

        # Final x corresponds to the logits output of the whole model
        return x


def save_model(model, model_save_folder, model_name):
    """
    Save the model to disk.
    """
    torch.save(model.state_dict(), os.path.join(model_save_folder, model_name))

    print(f"Model saved to {os.path.join(model_save_folder, model_name)}")


def load_model_state(infer_model, model_save_folder, model_name):
    """
    Load the model from disk.
    """
    checkpoint = torch.load(
        os.path.join(model_save_folder, model_name),
        map_location=device,
        weights_only=True     # <-- only load tensor weights, no pickle objects
    )
    infer_model.load_state_dict(checkpoint)
    infer_model.eval()
    return infer_model


def compute_accuracy(loader, model):
    total_correct = 0
    total_samples = 0
    for data, labels in loader:
        logits = model(data)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum()
        total_samples += labels.numel()
    return total_correct.float() / total_samples


def compute_sat_levels(loader, P):
	sat_level  = 0
	for data, labels in loader:
		x = ltn.Variable("x", data)
		x_MQTT_DDoS_Connect_Flood = ltn.Variable("x_MQTT_DDoS_Connect_Flood", data[labels == 0])
		x_MQTT_DDoS_Publish_Flood = ltn.Variable("x_MQTT_DDoS_Publish_Flood", data[labels == 1])
		x_MQTT_DoS_Connect_Flood = ltn.Variable("x_MQTT_DoS_Connect_Flood", data[labels == 2])
		x_MQTT_DoS_Publish_Flood = ltn.Variable("x_MQTT_DoS_Publish_Flood", data[labels == 3])
		x_MQTT_Malformed_Data = ltn.Variable("x_MQTT_Malformed_Data", data[labels == 4])
		x_Benign = ltn.Variable("x_Benign", data[labels == 5])

		sat_level = SatAgg(
			Forall(x_MQTT_DDoS_Connect_Flood, P(x_MQTT_DDoS_Connect_Flood, l_MQTT_DDoS_Connect_Flood)),
			Forall(x_MQTT_DDoS_Publish_Flood, P(x_MQTT_DDoS_Publish_Flood, l_MQTT_DDoS_Publish_Flood)),
			Forall(x_MQTT_DoS_Connect_Flood, P(x_MQTT_DoS_Connect_Flood, l_MQTT_DoS_Connect_Flood)),
			Forall(x_MQTT_DoS_Publish_Flood, P(x_MQTT_DoS_Publish_Flood, l_MQTT_DoS_Publish_Flood)),
			Forall(x_MQTT_Malformed_Data, P(x_MQTT_Malformed_Data, l_MQTT_Malformed_Data)),
			Forall(x_Benign, P(x_Benign, l_Benign))
		)
	return sat_level


##############################Load data######################################
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # Use CPU for this example
print(f"\n Using device: {device} \n")

# Load data
input_folder = '/home/zyang44/Github/baseline_cicIOT/P1_structurelevel/efficiency/input_files'
train_fname = 'logiKNet_train_35945.csv'
test_fname = 'logiKNet_test_3994.csv'

train_df, test_df = load_csv_data(input_folder, train_fname, test_fname)
# Extract features and labels   
X_train, y_train = extract_features_labels(train_df)
X_test, y_test = extract_features_labels(test_df)

dataset_numeric = {
    'train_input': torch.tensor(X_train, dtype=torch.float32, device=device),
    'train_label': torch.tensor(y_train, dtype=torch.long, device=device),
    'test_input': torch.tensor(X_test, dtype=torch.float32, device=device),
    'test_label': torch.tensor(y_test, dtype=torch.long, device=device)
}

train_loader = DataLoader(
    dataset_numeric['train_input'],
    dataset_numeric['train_label'], 
    batch_size=len(X_train), 
    shuffle=True
    )
test_loader = DataLoader(
    dataset_numeric['test_input'],
    dataset_numeric['test_label'],
    # batch_size=len(X_test),
    shuffle=False
    )


###############################load model and testing########################################
model_state_folder = '/home/zyang44/Github/baseline_cicIOT/P1_structurelevel/efficiency/model_weights'

# load all four models
mlp_infer = MLP(layer_sizes=(18, 10, 6)).to(device)
mlp_infer = load_model_state(mlp_infer, model_state_folder, 'mlp.pt')

logicmlp_infer = MLP(layer_sizes=(18, 10, 6)).to(device)
logicmlp_infer = load_model_state(logicmlp_infer, model_state_folder, 'logic_mlp.pt')

logiKNet_infer = KAN(width=[18, 10, 6], grid=5, k=3, seed=42, device=device)
logiKNet_infer = load_model_state(logiKNet_infer, model_state_folder, 'logiKNet.pt')

hierarchical_logiKNet_infer = KAN(width=[18, 10, 6], grid=5, k=3, seed=42, device=device)
hierarchical_logiKNet_infer = load_model_state(hierarchical_logiKNet_infer, model_state_folder, 'hierarchical_logiKNet.pt')

model_list = {
    'mlp': mlp_infer,
    'logic_mlp': logicmlp_infer,
    'logiKNet': logiKNet_infer,
    'hierarchical_logiKNet': hierarchical_logiKNet_infer
}

# test the models 
def test_model(model, loader, model_name=""):
    start_time = time.perf_counter()

    model.eval()
    with torch.no_grad():
        for data, labels in loader:
            logits = model(data)
            preds = torch.argmax(logits, dim=1)

    end_time = time.perf_counter()
    print(f"[{model_name}] Inference time: {end_time - start_time:.4f} seconds")


for model_name, model in model_list.items():
    test_model(model, test_loader, model_name)