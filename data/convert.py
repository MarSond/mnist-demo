import os
import struct
import numpy as np
from PIL import Image

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def save_images(images, labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (img, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(img, 'L')
        img.save(os.path.join(output_dir, f'{label}_{i}.png'))

def main():
    train_images_file = 'train-images.idx3-ubyte'
    train_labels_file = 'train-labels.idx1-ubyte'
    test_images_file = 't10k-images.idx3-ubyte'
    test_labels_file = 't10k-labels.idx1-ubyte'

    train_images = read_idx(train_images_file)
    train_labels = read_idx(train_labels_file)
    test_images = read_idx(test_images_file)
    test_labels = read_idx(test_labels_file)

    save_images(train_images, train_labels, 'mnist_train')
    save_images(test_images, test_labels, 'mnist_test')

if __name__ == '__main__':
    main()
