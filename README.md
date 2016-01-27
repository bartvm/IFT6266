# IFT6266 project

To help you get started with the IFT6266 class projects, we'll provide a snippets of code to read and process the datasets that you are asked to work with.

## Dogs vs. Cats

The Dogs vs. Cats dataset was originally part of [a Kaggle competition](https://www.kaggle.com/c/dogs-vs-cats). The competition page and [forums](https://www.kaggle.com/c/dogs-vs-cats/forums) will have more details on the dataset and approaches that people have tried.

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
from fuel.datasets.dogs_vs_cats import DogsVsCats
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme

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

# To train your network for 10 epochs, you can now do something like
for epoch in range(10):
    epoch_iterator = stream.get_epoch_iterator()
    for batch in epoch_iterator:
        train_network(batch)
```

Note that the Dogs vs. Cats dataset only has a training set and a test set; you'll need to create your own validation set! This is why we selected a subset of 20,000 images of the 25,000 as our training set.

You'll need to extend this example a bit for it to work well; 32 x 32 crops are too small, but some of the images aren't any bigger. To deal with this you'll need to upscale the smaller images (see e.g. `fuel.transformers.image.MinimumImageDimensions`), but you might also want to downscale the bigger ones.

You can download the original datasets from Kaggle or from here: [train](https://www.dropbox.com/s/s3u30quvpxqdbz6/train.zip?dl=1), [test](https://www.dropbox.com/s/21rwu6drnplsbkb/test1.zip?dl=1).
