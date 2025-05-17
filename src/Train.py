import os
import time

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import src.GAN as GAN
import src.Dataloader as Dataloader

def trainGAN(epochs=10, lr=0.001, latent_size=128, output_dir="weights", checkpoint=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Dataloader.ImageDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = GAN.Generator().to(DEVICE)
    discriminator = GAN.Discriminator().to(DEVICE)
    if checkpoint:
        generator.load_state_dict(torch.load("weights/generator_epoch_10.pth"))
        discriminator.load_state_dict(torch.load("weights/generator_epoch_10.pth"))
    optimizerGen = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerDisc = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    loss_fn = BCEWithLogitsLoss()

    generator.train()
    discriminator.train()

    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    start_time = time.time()
    for epoch in range(epochs):
        last_d_loss = None
        last_g_loss = None

        for i, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(DEVICE)


            # ==========================
            # Training the Discriminator
            # ==========================

            # Generate fake images
            z = torch.randn(batch_size, latent_size, 1, 1).to(DEVICE)
            fake_images = generator(z)

            # Defining the labels
            real_labels = torch.ones(batch_size, device=DEVICE)
            fake_labels = torch.zeros(batch_size, device=DEVICE)

            # Reset gradients
            optimizerDisc.zero_grad()

            # Variables for how real the discriminator thinks the images are
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())

            # Loss scores
            d_loss_real = loss_fn(real_output, real_labels)
            d_loss_fake = loss_fn(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            # Backpropagate and update the discriminators weights
            d_loss.backward()
            optimizerDisc.step()


            # =======================
            # Training the Generator
            # =======================

            # Generate new fake images
            z = torch.randn(batch_size, latent_size, 1, 1).to(DEVICE)
            fake_images = generator(z)

            # Trying to trick the discriminator
            optimizerGen.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss = loss_fn(fake_output, real_labels)

            # Backpropagate and update the generators weights
            g_loss.backward()
            optimizerGen.step()

            # Store the latest loss values
            last_d_loss = d_loss.item()
            last_g_loss = g_loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}]  D_loss: {last_d_loss:.4f}  G_loss: {last_g_loss:.4f}")
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        if (epoch + 1) == epochs:
            generator.eval()
            with torch.no_grad():
                z = torch.randn(16, latent_size, 1, 1).to(DEVICE)
                samples = generator(z)
                samples = (samples + 1) / 2  # Rescale from [-1, 1] to [0, 1]

                # Save image output
                save_image(samples, os.path.join(output_dir, f"epoch_{epoch+1}.png"), nrow=4)

            # Save model weights
            torch.save(generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch+1}.pth"))

            generator.train()