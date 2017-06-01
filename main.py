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
num_epochs = 20
batch_size = 100

total_classes = 10
num_classes = 2


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
K = 1000 # total number of exemplars
icarl = iCaRLNet(2048, 1)
icarl.cuda()


for s in range(0, total_classes, num_classes):
    # Load Datasets
    print "Loading training examples for classes", range(s, s+num_classes)
    train_set = iCIFAR10(root='./data',
                         train=True,
                         classes=range(s,s+num_classes),
                         download=True,
                         transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = iCIFAR10(root='./data',
                         train=False,
                         classes=range(num_classes),
                         download=True,
                         transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
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
        icarl.construct_exemplar_set(images, m, transform)
        print "Done"

    for y, P_y in enumerate(icarl.exemplar_sets):
        print "Exemplar set for class-%d:" % (y), P_y.shape
        #show_images(P_y[:10])

    icarl.n_known = icarl.n_classes
    print "iCaRL classes: %d" % icarl.n_known

    total = 0.0
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    print('Test Accuracy: %d %%' % (100 * correct / total))


"""
# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=learnig_rate)

for epoch in range(num_epochs):
    for i, (indices, images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        g = net(images)

        loss = sum(criterion(g[:,y], (labels==y).type(torch.cuda.FloatTensor))\
                for y in xrange(2))
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_set)//batch_size, loss.data[0]))
"""


