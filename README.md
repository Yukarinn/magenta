## Introduction

This is forked from the [Magenta](https://github.com/magenta/magenta) repository and we have modified the [sketch_rnn model](/magenta/models/sketch_rnn) for our CS 486 group research project. We would like to investigate in the impact of the randomness of the latent vector, and added a hyperparameter, `scale`, to the model which lets us specify the level of noise being sent to the latent vector. In our research paper, we are experimenting with values `0.5, 1.0, 2.0, 4.0`.

Below is the introduction copied from the Magenta repository.

<img src="magenta-logo-bg.png" height="75">

**Magenta** is a research project exploring the role of machine learning
in the process of creating art and music.  Primarily this
involves developing new deep learning and reinforcement learning
algorithms for generating songs, images, drawings, and other materials. But it's also
an exploration in building smart tools and interfaces that allow
artists and musicians to extend (not replace!) their processes using
these models.  Magenta was started by some researchers and engineers
from the [Google Brain team](https://research.google.com/teams/brain/),
but many others have contributed significantly to the project. We use
[TensorFlow](https://www.tensorflow.org) and release our models and
tools in open source on this GitHub.  If you’d like to learn more
about Magenta, check out our [blog](https://magenta.tensorflow.org),
where we post technical details.  You can also join our [discussion
group](https://groups.google.com/a/tensorflow.org/forum/#!forum/magenta-discuss).

This is the home for our Python TensorFlow library. To use our models in the browser with [TensorFlow.js](https://js.tensorflow.org/), head to the [Magenta.js](https://github.com/tensorflow/magenta-js) repository.

## To use the modified version of sketch_rnn

First, clone this repository:

```bash
git clone https://github.com/Yukarinn/magenta.git
```

Next, install the dependencies by changing to the base directory and executing the setup command:

```bash
pip install -e .
```

You can now run the model with this code:
```bash
python magenta/magenta/models/sketch_rnn/sketch_rnn_train.py
```

Here's an example of the command using the configs and hyperparameters:
```bash
python magenta/magenta/models/sketch_rnn/sketch_rnn_train.py --log_root=checkpoints --data_dir=datasets --hparams="data_set=[apple.npz, donut.npz, bus.npz, table.npz, calculator.npz, power_outlet.npz],scale=1.0, num_steps=10000,save_every=100,use_recurrent_dropout=0,kl_decay_rate=0.9999,min_learning_rate=0.0001"

```

Here is a list of full options for the model, along with the default settings:

```python
data_set=['aaron_sheep.npz'],  # Our dataset. Can be list of multiple .npz sets.
num_steps=10000000,            # Total number of training set. Keep large.
save_every=500,                # Number of batches per checkpoint creation.
dec_rnn_size=512,              # Size of decoder.
dec_model='lstm',              # Decoder: lstm, layer_norm or hyper.
enc_rnn_size=256,              # Size of encoder.
enc_model='lstm',              # Encoder: lstm, layer_norm or hyper.
z_size=128,                    # Size of latent vector z. Recommend 32, 64 or 128.
kl_weight=0.5,                 # KL weight of loss equation. Recommend 0.5 or 1.0.
kl_weight_start=0.01,          # KL start weight when annealing.
kl_tolerance=0.2,              # Level of KL loss at which to stop optimizing for KL.
batch_size=100,                # Minibatch size. Recommend leaving at 100.
grad_clip=1.0,                 # Gradient clipping. Recommend leaving at 1.0.
num_mixture=20,                # Number of mixtures in Gaussian mixture model.
learning_rate=0.001,           # Learning rate.
decay_rate=0.9999,             # Learning rate decay per minibatch.
kl_decay_rate=0.99995,         # KL annealing decay rate per minibatch.
min_learning_rate=0.00001,     # Minimum learning rate.
use_recurrent_dropout=True,    # Recurrent Dropout without Memory Loss. Recomended.
recurrent_dropout_prob=0.90,   # Probability of recurrent dropout keep.
use_input_dropout=False,       # Input dropout. Recommend leaving False.
input_dropout_prob=0.90,       # Probability of input dropout keep.
use_output_dropout=False,      # Output droput. Recommend leaving False.
output_dropout_prob=0.90,      # Probability of output dropout keep.
random_scale_factor=0.15,      # Random scaling data augmention proportion.
augment_stroke_prob=0.10,      # Point dropping augmentation proportion.
conditional=True,              # If False, use decoder-only model.
scale = 1.0                    # Added for CS486 research project to specify the scale of noise sent to the latent vector.
```
