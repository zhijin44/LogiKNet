from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
model = KAN(width=[2,6,1], grid=3, k=3, seed=42, device=device)


from kan.utils import create_dataset
import matplotlib.pyplot as plt 
# create dataset f(x,y) = exp(sin(pi*x)+y^2)
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2, device=device)
print(dataset['train_input'].shape, dataset['train_label'].shape)

# plot KAN at initialization
model(dataset['train_input'])
plt.figure()
model.plot()
plt.savefig('kan_initialization.png')


# train the model
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
plot_output = model.plot()
plt.savefig('kan_sparsity_reg.png')

model = model.prune()
plt.close()

