#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    # Take the mean of the token embeddings for simplicity
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


# In[2]:


import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# In[3]:


import torch.optim as optim

# Hyperparameters
latent_dim = 768  # Size of BERT embeddings
image_dim = 784  # Size of generated image (28x28 for simplicity)
epochs = 10000
batch_size = 32

# Instantiate models
generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
for epoch in range(epochs):
    for i, (text, real_images) in enumerate(dataloader):
        # Prepare inputs
        text_embeddings = text_to_embedding(text).detach()
        real_images = real_images.view(batch_size, -1)
        
        # Train Discriminator
        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        
        real_loss = criterion(discriminator(real_images), real_labels)
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(text_embeddings)
        fake_loss = criterion(discriminator(fake_images), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_images = generator(text_embeddings)
        g_loss = criterion(discriminator(fake_images), real_labels)
        
        g_loss.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")


# In[6]:


from pycocotools.coco import COCO
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class COCODataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return caption, img

# Define the paths to the COCO dataset
root_dir = 'path/to/coco/train2017'
annotation_file = 'path/to/coco/annotations/captions_train2017.json'

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128 for this example
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset and dataloader
coco_dataset = COCODataset(root_dir, annotation_file, transform=transform)
dataloader = DataLoader(coco_dataset, batch_size=32, shuffle=True)

# Example of using the dataloader
for epoch in range(epochs):
    for i, (text, real_images) in enumerate(dataloader):
        text_embeddings = text_to_embedding(text).detach()
        real_images = real_images.view(batch_size, -1)

        optimizer_D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        real_loss = criterion(discriminator(real_images), real_labels)
        z = torch.randn(batch_size, latent_dim)
        fake_images = generator(text_embeddings)
        fake_loss = criterion(discriminator(fake_images), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        fake_images = generator(text_embeddings)
        g_loss = criterion(discriminator(fake_images), real_labels)

        g_loss.backward()
        optimizer_G.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")


# In[7]:


import matplotlib.pyplot as plt

def generate_image_from_text(text):
    generator.eval()
    with torch.no_grad():
        text_embedding = text_to_embedding(text)
        generated_image = generator(text_embedding).view(28, 28).cpu().numpy()
        plt.imshow(generated_image, cmap='gray')
        plt.show()

# Example usage
generate_image_from_text("A beautiful sunny day in the park.")


# In[ ]:




