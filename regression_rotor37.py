import numpy as np
from plaid import Sample
from plaid.bridges import huggingface_bridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

dataset = huggingface_bridge.load_dataset_from_hub(
    "PLAID-datasets/Rotor37", split="all_samples"
)

dataset, pb_def = huggingface_bridge.huggingface_dataset_to_plaid(
    dataset, processes_number=2, verbose=False
)

input_scalars: np.ndarray = np.empty((1200, 2))
output_scalars: np.ndarray = np.empty((1200,))
nodes: np.ndarray = np.empty((1200, 29773, 3))
field: np.ndarray = np.empty((1200, 29773, 1))

for i, sample in enumerate(dataset):
    s: Sample = sample
    scalar = np.array([s.get_scalar("Omega"), s.get_scalar("P")])
    input_scalars[i] = scalar
    output_scalars[i] = s.get_scalar("Efficiency")
    nodes[i] = s.get_nodes()
    field[i] = np.expand_dims(s.get_field("Temperature"), axis=-1)
input_scalars = np.array(input_scalars)
output_scalars = np.array(output_scalars)
nodes = np.array(nodes)
field = np.array(field)

# train-test split
idx_train = np.random.choice(1000, size=800, replace=False)
idx_test = np.arange(start=400, stop=600)
idx_train = np.setdiff1d(np.arange(1000), idx_test)


nodes_train = nodes[idx_train]
nodes_test = nodes[idx_test]

input_scalars_train = input_scalars[idx_train]
input_scalars_test = input_scalars[idx_test]

output_scalars_train = output_scalars[idx_train]
output_scalars_test = output_scalars[idx_test]

# Dimensionality reduction with PCA
pca = PCA(n_components=32)
nodes_train_reduced = pca.fit_transform(nodes_train.reshape(800, -1))
nodes_test_reduced = pca.transform(nodes_test.reshape(200, -1))

pca_field = PCA(n_components=32)
field_train_reduced = pca_field.fit_transform(field[idx_train].reshape(800, -1))
field_test_reduced = pca_field.transform(field[idx_test].reshape(200, -1))

x_train = np.hstack((input_scalars_train, nodes_train_reduced))
x_test = np.hstack((input_scalars_test, nodes_test_reduced))

regressor = TabPFNRegressor(n_estimators=32)
regressor.fit(x_train, output_scalars_train)
preds = regressor.predict(x_test)

mse = mean_squared_error(output_scalars_test, preds)
r2 = r2_score(output_scalars_test, preds)

print(f"Test MSE: {mse}, R2: {r2}")

for i in range(32):
    regressor_field = TabPFNRegressor()
    regressor_field.fit(x_train, field_train_reduced[:, i])
    preds_field = regressor_field.predict(x_test)

    r2 = r2_score(field_test_reduced[:, i], preds_field)
    print(f"Field PCA Component {i+1} R2: {r2}")