# Speech Recognition Project: Deep Denoising Autoencoder Private
#### ELEC-E5510

This repo is based on course at Aalto University.
30.10.2019-13.12.2019

It includes the final project for the course.

[Link to the course](https://mycourses.aalto.fi/course/view.php?id=24700).

# Instructions

1. To run training of the Deep Auto Encoder, change directory to `./python`:
```
$ cd ./dda/python
```

2. Run the training
```
$ pyhton ./main.py
```

Alternatively you can run only the test
```
$ pyhton ./main.py  --train false
```

You can also change all these parameters:
```python
--training_data_dir  # default='./data/train_data'
--testing_data_dir   # default='./data/test_data'
--enhanced_dir       # default= './data/enhanced'
--h5_dir_name        # default='./data/train_data/training_files_h5'
--h5_file_name       # default='file'
--model_dir          # default='./trained_model'
--feat               # default='spec'
--num_cpu            # default=5
--epochs             # default=50
--batch_size         # default=32
--split_number       # default=100
--lr                 # default=1e-3
--train              # default=True
--test               # default=True
```
