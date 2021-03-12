Practice folder for GANs.
Implemented with tensorflow and keras.

Learned/did so far:
- Subclassed keras model
- Save model architecture as well as weights
- Continue training from previous training
- Distributed training with TPU in colab
- Updated to work with either gcs bucket or local filesystem
- Minibatch discrimination implemention

To learn/do:
- Profile performance in tensorboard
- Better input pipeline
- Hyperparameter search
- Continue implementing other techniques from `Improved Techniques for Training GANs` paper

### Setup gcloud to have access to gcs buckets from jupyter notebook locally:
These steps worked as of Jan 9 2021 so things might change later...
https://googleapis.dev/python/google-api-core/latest/auth.html

1. Install google cloud sdk (gcloud):
    Instructions: https://cloud.google.com/sdk/docs/install
    After running `gcloud init`, `~/.config/gcloud` should now be setup.
    (To place the config files in a different directory run `export CLOUDSDK_CONFIG=path/you/want` before running `gcloud init`)
2. Run `gcloud auth application-default login`
3. Make sure it works locally:
    Start a python shell and try to access your gcs bucket
    ```bash
    $ python
    >>> import tensorflow as tf
    >>> tf.io.gfile.stat('gs://your_bucket')
    ```
    If everything went well this should work.
    But at this point accessing the gcs bucket wouldn't work yet in the jupyter notebook.
4. Create a new service account `https://console.cloud.google.com/identity/serviceaccounts`
    Give the necessary permissions like `Service Account Admin` and add your user as member.
5. Download the service account key file.
6. Add a new environment variable `GOOGLE_APPLICATION_CREDENTIALS=path/to/your/downloaded/keyfile.json`
7. Start jupyter notebook, run `%env` to list all environment variables and make sure `GOOGLE_APPLICATION_CREDENTIALS` is in there.
8. Try accessing your gcs bucket:
    ```python
    import tensorflow as tf
    tf.io.gfile.stat('gs://your_bucket')
    ```
Should be good to go now.

### Serialize/Deserialize GAN model

Since my gan model is a subclassed keras model instead of 2 stacked Sequentials, I had issues trying to use `gan.save()` directly on my model to save both the discriminator and
the generator. So instead I modified it to compile the gen and dis separately and call `.save()` on each. To save the architecture of the gan model I overrode `get_config` and `from_config`. Though by default `model_from_json` returns a non-compiled model so to ease the process of restarting the training from a checkpoint, in `compile()` I save the compile arguments and in `from_config` I use the respective `deserialize` functions to compile the model so I can directly call `fit` when loading.
ex:
```python
gan = GAN.load('logs', latest=True)
gan.fit(...)
```
Ah also, had to switch to `tf.io.gfile.GFile('path', 'r').read()` instead of `open('path', 'r').read()` so that it works for both a gcs bucket or a local path. 