import torch
import numpy as np
import matplotlib.pyplot as plt


def print_image(sample):
    """ Print one sample of СometDetectionDataset class.

    For example:  print_image(train[2])

    Args:
        sample (object): item of СometDetectionDataset class.

    Returns:
        Plot sample with bounding box.
    """

    fig, ax = plt.subplots(figsize = (9,9))

    image = np.moveaxis(sample['image'].numpy(), 0, 2)
    ax.imshow(image)

    print(sample['file_name'])

    x1_y1 = sample['x1_y1'].numpy()
    x2_y2 = sample['x2_y2'].numpy()

    w = abs(x1_y1[0] - x2_y2[0]) * image.shape[0]
    h = abs(x1_y1[1] - x2_y2[1]) * image.shape[1]
    x = min(x1_y1[0], x2_y2[0]) * image.shape[0]
    y = min(x1_y1[1], x2_y2[1]) * image.shape[1]

    ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', lw=2))
    plt.show()


def show_images_batch(batch):
    """ Print one batch of samples.

    For example:  show_images_batch(next(iter(train_loader)))

    Args:
        batch (objects): item of DataLoader based on СometDetectionDataset class.

    Returns:
        Plot samples with bounding boxes.
    """

    img_list = batch['image']
    x1_y1_list = batch['x1_y1']
    x2_y2_list = batch['x2_y2']
    ratio = batch['ratio']
    ratio = torch.unsqueeze(ratio, 1)

    label = batch['label']
    files_name = batch['file_name']

    show_list_images(img_list, x1_y1_list, x2_y2_list, ratio, files_name)


def show_list_images(img_list, x1_y1_list, x2_y2_list, ratio, files_name):
    """ Utility function for ploting image with bounding box
    """

    fig = plt.figure(figsize = (20,20))
    for i in range(len(img_list)):
        ax = fig.add_subplot(4, 4, i+1)
        image = np.moveaxis(img_list[i].cpu().detach().numpy(), 0, 2)
        ax.imshow(image)
        x1_y1 = x1_y1_list[i].cpu().detach().numpy()
        x2_y2 = x2_y2_list[i].cpu().detach().numpy()
        ratio_str = ratio[i].cpu().detach().numpy()[0]
        w = abs(x1_y1[0] - x2_y2[0]) * image.shape[0]
        h = abs(x1_y1[1] - x2_y2[1]) * image.shape[1]
        x = min(x1_y1[0], x2_y2[0]) * image.shape[0]
        y = min(x1_y1[1], x2_y2[1]) * image.shape[1]

        ax.add_patch(patches.Rectangle((x,y), w, h, fill=False, edgecolor='red', lw=2))
        file_path, file_name = os.path.split(files_name[i])

        ax.set_title('File {0} ratio {1:.5f}'.format(file_name,ratio_str))

    plt.show()


def plot_training_curves(training, title):
    """ Plot curves of training process
    """

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(training['n_epoch'],training['loss'])
    ax.grid()
    ax.set_title(title)
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plt.show()


