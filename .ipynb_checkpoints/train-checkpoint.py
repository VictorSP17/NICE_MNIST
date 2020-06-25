import os
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
from torchvision.utils import save_image
import sys
from config import cfg
from nice import NICE
import matplotlib.pyplot as plt
import pylab
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Hyper-parameters
NN = True

# Data preprocessing:
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(dataset=test, batch_size=cfg['TRAIN_BATCH_SIZE'],
                                         shuffle=True, pin_memory=True)

# Model and criterion:
model = NICE(data_dim=784, num_coupling_layers=cfg['NUM_COUPLING_LAYERS'])
if cfg['USE_CUDA']:
  device = torch.device('cuda')
  model = model.to(device)

opt = optim.Adam(model.parameters())

# Load model
state_dict = torch.load('saved_models/59.pt')
model.load_state_dict(state_dict)

# TRAINING:
testL = np.zeros(cfg['TRAIN_EPOCHS'])
trainL = np.zeros(cfg['TRAIN_EPOCHS'])

for epoch in range(cfg['TRAIN_EPOCHS']):
  mean_likelihood = 0.0
  num_minibatches = 0

  model.train()

  for batch_id, (x, _) in enumerate(dataloader):
      x = x.view(-1, 784) + torch.rand(784) / 256. # Uniform noise between 0 and 1/256 at most!
      if cfg['USE_CUDA']:
        x = x.cuda()

      x = torch.clamp(x, 0, 1) # Make sure values between 0 and 1.

      z, likelihood = model(x)
      loss = -torch.mean(likelihood)   # NLL

      loss.backward()
      opt.step()
      model.zero_grad()

      mean_likelihood -= loss
      num_minibatches += 1

      #print(mean_likelihood / num_minibatches)
      #print(batch_id)

  mean_likelihood /= num_minibatches
  trainL[epoch] = mean_likelihood
  print('Epoch {} completed. Log Likelihood: {}'.format(epoch, mean_likelihood))
  
  # TEST EVALUATION ########
  ml = 0
  num_minibatches = 0
  for batch_id, (x, _) in enumerate(testloader):
      x = x.view(-1, 784)
      z, likelihood = model(x)
      l = -torch.mean(likelihood)

      ml -= l
      num_minibatches += 1

  ml /= num_minibatches
  testL[epoch] = ml
  print('Test Log Likelihood: {}'.format(ml))
  ############

  plt.figure()
  pylab.xlim(0, cfg['TRAIN_EPOCHS'] + 1)
  plt.plot(range(1, cfg['TRAIN_EPOCHS'] + 1), testL, label='test loss')
  plt.plot(range(1, cfg['TRAIN_EPOCHS'] + 1), trainL, label='train loss')
  plt.legend()
  plt.savefig(os.path.join('save', 'loss.pdf'))
  plt.close()

  if epoch % 5 == 0:
    save_path = os.path.join(cfg['MODEL_SAVE_PATH'], '{}.pt'.format(epoch))
    torch.save(model.state_dict(), save_path)

    # SOME SAMPLING:
    model.train(False)

    fake_images = model.sample(20).round() # round to 0, 1
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    sample_dir = 'samples'
    save_image(fake_images.data, os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch)))



# NEAREST NEIGHBOURS:
# Images 28x28, search the closest one.
# function(generated_image) --> closest training_image
if NN == True:
  aux_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=1,
                                                shuffle=False)

  def nearest_gt(generated_image):
      min_d = 0
      closest = False
      for i, (image, _) in enumerate(aux_data_loader):
          image = image.view(1, 28, 28).round() # all distances in binary...
          d = torch.dist(generated_image,image) # must be torch tensors (1,28,28)
          if i == 0 or d < min_d:
              min_d = d
              closest = image

      return closest

  fake_images = model.sample(24).round() # round to 0, 1
  fake_images = fake_images.view(24, 1, 28, 28)
  save_image(fake_images, './samples/f24.png')
  NN = torch.zeros(24, 1, 28, 28)
  for i in range(0,24):
        NN[i] = nearest_gt(fake_images[i])
        print(i)
  save_image(NN.data, './samples/NN24.png')





