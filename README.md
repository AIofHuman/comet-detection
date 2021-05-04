# comet-detection
The goal of this hobby project was to show the possibility of detecting comets in astro images using modern neural network architectures.

Project steps:
1. With the help of Google, 150 astrographs of comets were found.
2. Using preprocessing procedures (prepare_image_files_to_labels.ipynb), the images were renamed to "image_n.jpg" and convert to the format (3, 500, 500) (channel, width, height).
3. With the help of cvat.org, the bounding boxes were marked. The label was done in 1.5 hours.
4. Using augmentation (albumentations https://github.com/albumentations-team/albumentations), increased the dataset to 600 samples.
4. I tested two architectures of the network architecture: 'Resnet-18' and 'Efficientnet'. The best results were shown by the 'Efficientnet'.
5. Expert knowledge was added to the training process: usually the comment is bright and extended in the image, 
so the average signal level in the bounding box will differ from the average level of the image. 
I create function to calc ratio: avg_signal_bb/ (avg_signal_bb + avg_signal_all_image). And predict ratio in train process.
6. The final training of the neural network took 2.5 hours on my NVideo GTX 2060 video card.

Recap:
- 
