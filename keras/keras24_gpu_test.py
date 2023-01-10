import tensorflow as tf
print(tf.__version__)           #2.7.4

#gpu 확인법

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)      #[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


if(gpus):
    print(1)
else:
    print(0)
