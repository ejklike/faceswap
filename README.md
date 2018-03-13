# faceswap
experimental faceswap repository

## Dependencies
this code is tested on python>=3.5 and CUDA 8.0 environment with additional packages below:

```
pathlib==1.0.1
scandir==1.6
h5py==2.7.1
Keras==2.1.2
opencv-python==3.3.0.10
tensorflow-gpu==1.4.0
scikit-image
dlib # need some attention to install!
face_recognition
tqdm
```

## How to

### Download celebA dataset

```python
import celeba_helper as helper

data_dir = '../celebA' # as you want
helper.download_extract('celeba', data_dir)
```

### Train model

```bash
$ python train.py
```


## References
- [deepfakes](https://github.com/deepfakes/faceswap): faceswap model code
- [face_gen](https://github.com/bestkao/face_gen): celebA dataset download helper
