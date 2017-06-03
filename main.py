import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data_loader import iCIFAR10, iCIFAR100
from model import iCaRLNet

def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()


# Hyper Parameters
total_classes = 10
num_classes = 10


transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Initialize CNN
K = 2000 # total number of exemplars
icarl = iCaRLNet(2048, 1)
icarl.cuda()


for s in range(0, total_classes, num_classes):
    # Load Datasets
    print "Loading training examples for classes", range(s, s+num_classes)
    train_set = iCIFAR10(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         download=True,
                         transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
                                               shuffle=True, num_workers=2)

    test_set = iCIFAR10(root='./data',
                         train=False,
                         classes=range(num_classes),
                         download=True,
                         transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                               shuffle=True, num_workers=2)



    # Update representation via BackProp
    icarl.update_representation(train_set)
    m = K / icarl.n_classes

    # Reduce exemplar sets for known classes
    icarl.reduce_exemplar_sets(m)

    # Construct exemplar sets for new classes
    for y in xrange(icarl.n_known, icarl.n_classes):
        print "Constructing exemplar set for class-%d..." %(y),
        images = train_set.get_image_class(y)
        icarl.construct_exemplar_set(images, m, transform_test)
        print "Done"

    for y, P_y in enumerate(icarl.exemplar_sets):
        print "Exemplar set for class-%d:" % (y), P_y.shape
        #show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print "iCaRL classes: %d" % icarl.n_known

    total = 0.0
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Train Accuracy: %d %%' % (100 * correct / total))

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))


