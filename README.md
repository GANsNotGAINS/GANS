# Audio GAN
The name is a little misleading as we want to train a GAN on an image representation of midi. The issue with this is that GANs take fixed size inputs and generate fixed size images, whereas audio data can be of varying lengths. Therefore, we must clip the midi file or the resulting midi2pix image.

Currently, I just added code for a basic GAN on the MNIST dataset. I used tensorflow and the code contains generator and discriminator model code (components of the GAN), loss calculations, optimizers, and the training setup. All we need to do is use images from our dataset and tune some hyperparameters including the GAN architecture (i.e. what layers to use, etc.).

## Packages
- Tensorflow
- Tensorboard


## Training the GAN
This will generate `gan_models/` directory which will store the checkpoint every 10k iterations and the `gan_tensorboard/` directory which can be used to visualize tensorboard.
```bash
python gan.py
```

## Running Tensorboard
You can run tensorboard by specifying the log directory and navigate to `localhost:6006` in browser to view the scalar summaries and images
Tensorboard provides scalar summaries of the generator and discriminator losses as well as an image tab which displays sample generated and ground truth images.
```bash
tensorboard --logdir=gan_tensorboard
```


## TODO
- Fix our audio input (trim to a certain image size) and adapt the conv layers to fit our midi2pix image
- Update dataset of images that the GAN is trained on

A simple GAN structure with the discriminator composed of 2 fully connected layers to predict whether an image is real or fake (generated). The generator has 2 fc layers as well. We can arbitrarily change this architecture depending on what works (add dropout, change activation functions, add/remove layers, etc).

Currently, I used the MNIST dataset just to test out the GAN code. The items in the `TODO` are primarily to just use whatever midi2pix images we have.