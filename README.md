# IFT6266 project

To help you get started with the IFT6266 class projects, we'll provide a snippets of code to read and process the datasets that you are asked to work with.

## Dogs vs. Cats

The Dogs vs. Cats dataset was originally part of [a Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats). The competition page and [forums](https://www.kaggle.com/c/dogs-vs-cats/forums) will have more details on the dataset and approaches that people have tried. Also have a look at [the leaderboard](https://github.com/vdumoulin/ift6266h15/wiki/Class-Project-Leaderboard) from last year's class.

[Fuel]() is a data-processing framework for machine learning that helps you download, read and process data for the training of machine learning algorithms. Fuel contains [a wrapper](https://github.com/mila-udem/fuel/pull/285) for the Dogs vs. Cats dataset. To get started with it, begin by following the [installation instructions](http://fuel.readthedocs.org/en/latest/setup.html). Afterwards, follow [the instructions](http://fuel.readthedocs.org/en/latest/built_in_datasets.html) for downloading a built-in dataset but use `dogs_vs_cats` instead of `mnist`. This will download the Dogs vs. Cats dataset for you and convert the JPEG images into a numerical HDF5 file (which can easily be viewed as NumPy arrays). The following commands should get you started:

```bash
cd $HOME
mkdir fuel_data  # Create a directory in which Fuel can store its data
echo "data_path: \"$HOME/fuel_data\"" > ~/.fuelrc  # Create the Fuel configuration file
cd fuel_data  # Go to the data directory
fuel-download dogs_vs_cats  # Download the original dataset into the current directory
fuel-convert dogs_vs_cats  # Convert the raw images into an HDF5 (numerical) dataset
```

Fuel was written to deal with out-of-memory datasets, streaming data, parallel on-the-fly preprocessing of data, etc. in mind, so take some time to read [the overview](fuel.readthedocs.org/en/latest/overview.html) in order to understand the basic terminology. As a quick pointer, consider the following example:

```python
# Let's load and process the dataset
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers.image import RandomFixedSizeCrop
from fuel.transformers import Flatten

# Load the training set
train = DogsVsCats(('train',), subset=slice(0, 20000))

# We now create a "stream" over the dataset which will return shuffled batches
# of size 128. Using the DataStream.default_stream constructor will turn our
# 8-bit images into floating-point decimals in [0, 1].
stream = DataStream.default_stream(
    train,
    iteration_scheme=ShuffledScheme(train.num_examples, 128)
)

# Our images are of different sizes, so we'll use a Fuel transformer
# to take random crops of size (32 x 32) from each image
cropped_stream = RandomFixedSizeCrop(
    stream, (32, 32), which_sources=('image_features',))
    
# We'll use a simple MLP, so we need to flatten the images
# from (channel, width, height) to simply (features,)
flattened_stream = Flatten(
    cropped_stream, which_sources=('image_features',))
```

Note that the Dogs vs. Cats dataset only has a training set and a test set; you'll need to create your own validation set! This is why we selected a subset of 20,000 images of the 25,000 as our training set.

You'll need to extend this example a bit for it to work well; 32 x 32 crops are too small, but some of the images aren't any bigger. To deal with this you'll need to upscale the smaller images (see e.g. `fuel.transformers.image.MinimumImageDimensions`), but you might also want to downscale the bigger ones.

You can download the original datasets from Kaggle or from here: [train](https://www.dropbox.com/s/s3u30quvpxqdbz6/train.zip?dl=1), [test](https://www.dropbox.com/s/21rwu6drnplsbkb/test1.zip?dl=1).

### Quick example with Blocks

[Blocks](https://blocks.readthedocs.org/en/latest/) is a framework for training neural networks that is used a lot at MILA. It helps you build neural network models, apply optimization algorithms, monitor validation sets, plot your results, serialize your models, etc.

Blocks is a framework made for research, and isn't as plug-and-play as some other frameworks (e.g. [Keras](https://github.com/fchollet/keras)). As such, we don't expect you to use it. However, some parts such as the optimization algorithms and monitoring can save you some time. Have a look at [Blocks' introduction tutorial](https://blocks.readthedocs.org/en/latest/tutorial.html) to get you started and consider the following example which shows you how to optimize a Theano expression of a cost using Blocks.

```python
# Create the Theano MLP
import theano
from theano import tensor
import numpy

X = tensor.matrix('image_features')
T = tensor.lmatrix('targets')

W = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(3072, 500)), 'W')
b = theano.shared(numpy.zeros(500))
V = theano.shared(
    numpy.random.uniform(low=-0.01, high=0.01, size=(500, 2)), 'V')
c = theano.shared(numpy.zeros(2))
params = [W, b, V, c]

H = tensor.nnet.sigmoid(tensor.dot(X, W) + b)
Y = tensor.nnet.softmax(tensor.dot(H, V) + c)

loss = tensor.nnet.categorical_crossentropy(Y, T.flatten()).mean()

# Use Blocks to train this network
from blocks.algorithms import GradientDescent, Scale
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

algorithm = GradientDescent(cost=loss, parameters=params,
                            step_rule=Scale(learning_rate=0.1))

# We want to monitor the cost as we train
loss.name = 'loss'
extensions = [TrainingDataMonitoring([loss], every_n_batches=1),
              Printing(every_n_batches=1)]

main_loop = MainLoop(data_stream=flattened_stream, algorithm=algorithm,
                     extensions=extensions)
main_loop.run() 
```

### First things to try

* You'll want to use convolutional nets.
* There is currently no transformer in Fuel that resizes images to a fixed size (i.e. make sure that the shortest size is N pixels). This might be a good idea to implement.
* Resizing and cropping images can be CPU-intensive. Moreover, it is a good idea to to data augmentation on this dataset e.g. add rotations, distortions, etc. to make sure the network doesn't overfit. By default these operations are done in the same process that controls the training on the GPU (or CPU) which means that during the data processing no training is happening and vice versa. This isn't very efficient! Fuel has a "data server" which allows you to run training and data preprocessing/augmentation in parallel. Have a look at the tutorial [here](https://github.com/vdumoulin/fuel/blob/server_doc/docs/server.rst).

## Spiritual Ascension Music

The second dataset is a 3 hour long audio track from [a YouTube video](https://www.youtube.com/watch?v=XqaJ2Ol5cC4). For this too a Fuel wrapper is available, but you'll need to make sure to install the Python module `pafy` (use `pip install pafy`) and the `ffmpeg` package.

On Linux you can probably use e.g. `sudo apt-get install ffmpeg` or `yum install ffmpeg` depending on your platform, while on OS X it's probably easiest to use [Homebrew](http://brew.sh/) (`brew install ffmpeg`). There are also [Windows builds](http://ffmpeg.zeranoe.com/builds/) available.

```bash
fuel-download youtube_audio --youtube-id XqaJ2Ol5cC4
fuel-convert youtube_audio --youtube-id XqaJ2Ol5cC4
```

If you can't manage to install `ffmpeg`, you can also download the HDF5 file [directly from Dropbox](https://www.dropbox.com/s/9jljjz2t21a70sz/XqaJ2Ol5cC4.hdf5?dl=1) and simply place it in your Fuel data path. (The [WAVE file](https://www.dropbox.com/s/ytohwf0l0xulrxg/XqaJ2Ol5cC4.wav?dl=1) is also available.)

```python
from fuel.datasets.youtube_audio import YouTubeAudio
data = YouTubeAudio('XqaJ2Ol5cC4')
stream = data.get_example_stream()
it = stream.get_epoch_iterator()
sequence = next(it)
```

Note that this gives you the entire sequence as one batch. During training, you probably want to split up the sequence in smaller subsequences e.g. of length `N`. So you would have sequences `[T, T + N]` while having `[T + 1, T + N + 1]` as targets. To do this you will need to implement a Fuel transformer.

### Tips

* Even if you don't use Blocks in order to construct your model, be sure to look at its implementations of e.g. the [LSTM](https://github.com/mila-udem/blocks/blob/master/blocks/bricks/recurrent.py#L419-L477) and [GRU](https://github.com/mila-udem/blocks/blob/master/blocks/bricks/recurrent.py#L568-L604) units; it's easy to get wrong!
* If you want to get started on a Fuel transformer that produces sets of subsequences, have a look at the [`NGrams` transformer](https://github.com/mila-udem/fuel/blob/master/fuel/transformers/text.py).
