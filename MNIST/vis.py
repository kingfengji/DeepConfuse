import numpy as np
import matplotlib.pyplot as plt
import torch
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
origindata = torch.load('data/MNIST/processed/training.pt')
advdata = torch.load('training_adv.pt')
for i in range(1, 11):
    idx = np.random.randint(10000)
    fig.add_subplot(rows, columns, 2 * i - 1)
    plt.imshow(origindata[0][idx])
    fig.add_subplot(rows, columns, 2 * i)
    plt.imshow(advdata[idx])
plt.savefig('mnist.png')