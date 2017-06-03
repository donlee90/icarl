import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image

from resnet import resnet18

# Hyper Parameters
num_epochs = 50
batch_size = 100
learning_rate = 0.002

class iCaRLNet(nn.Module):
    def __init__(self, feature_size, n_classes):
        # Network architecture
        super(iCaRLNet, self).__init__()
        self.feature_extractor = resnet18()
        self.feature_extractor.fc =\
            nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)

        self.n_classes = n_classes
        self.n_known = 0

        # List containing exemplar_sets
        # Each exemplar_set is a np.array of N images
        # with shape (N, C, H, W)
        self.exemplar_sets = []

        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        #self.optimizer = optim.SGD(self.parameters(), lr=2.0,
        #                           weight_decay=0.00001)

        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x, transform):
        """Classify images by neares-means-of-exemplars

        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """
        batch_size = x.size(0)

        if self.compute_means:
            print "Computing mean of exemplars...",
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # Extract feature for each exemplar in P_y
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print "Done"

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x) # (batch_size, feature_size)
        for i in xrange(feature.size(0)): # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        _, preds = dists.min(1)

        return preds
        

    def construct_exemplar_set(self, images, m, transform):
        """Construct an exemplar set for image set

        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        features = []
        for img in images:
            x = Variable(transform(Image.fromarray(img)), volatile=True).cuda()
            feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature) # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

        exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in xrange(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0/(k+1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
            """
            print "Selected example", i
            print "|exemplar_mean - class_mean|:",
            print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
            #features = np.delete(features, i, axis=0)
            """
        
        self.exemplar_sets.append(np.array(exemplar_set))
                

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]


    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)


    def update_representation(self, dataset):

        self.compute_means = True

        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print "%d new classes" % (len(new_classes))

        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

        # Store network outputs with pre-update parameters
        q = torch.zeros(len(dataset), self.n_classes).cuda()
        for indices, images, labels in loader:
            images = Variable(images).cuda()
            indices = indices.cuda()
            g = F.sigmoid(self.forward(images))
            q[indices] = g.data
        q = Variable(q).cuda()

        # Run network training
        optimizer = self.optimizer

        for epoch in xrange(num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                g = self.forward(images)
                
                # Classification loss for new classes
                loss = self.cls_loss(g, labels)
                #loss = loss / len(range(self.n_known, self.n_classes))

                # Distilation loss for old classes
                if self.n_known > 0:
                    g = F.sigmoid(g)
                    q_i = q[indices]
                    dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y])\
                            for y in xrange(self.n_known))
                    #dist_loss = dist_loss / self.n_known
                    loss += dist_loss

                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                           %(epoch+1, num_epochs, i+1, len(dataset)//batch_size, loss.data[0]))
