import numpy as np
import torch
import torch.nn as nn
from distance_matching_network import *
from tqdm import tqdm
import argparse
from torch.optim import Adam
import torch.nn.functional as F
import warnings
import data as dataset
from tensorboard.summary import writer
from torch.autograd import Variable

warnings.filterwarnings('ignore')

def train(train_batches, data, model, optimizer, device):
    total_c_loss = 0.
    total_accuracy = 0.

    for i in range(train_batches):
        x_support_set, y_support_set, x_target, y_target = data.get_train_batch(augment=True)
        x_support_set = Variable(x_support_set).to(device)
        y_support_set = Variable(y_support_set).to(device)
        x_target = Variable(x_target).to(device)
        y_target = Variable(y_target).to(device)
        preds, target_label = model(x_support_set, y_support_set, x_target, y_target)

        #produce predictions for target probablities
        correct_prediction = torch.argmax(preds, 1) == target_label
        accuracy = torch.mean(correct_prediction)
        targets = target_label.scatter_(1, target_label, num_classes_per_set)

        optimizer.zero_grad()
        loss = F.cross_entropy(targets, preds)
        loss.backward()
        optimizer.step()

        total_c_loss += loss.data[0]
        total_c_accuracy += accuracy
        self.iteration +=1
        if self.iteration %1500:
            self.lr*=0.5
            print ('LR changed to: ', self.lr)
    
    total_c_loss /= train_batches
    total_accuracy /=train_batches
    return total_c_loss, total_accuracy, loss

def validation(val_batches, data, model, optimizer, device):
    total_val_loss = 0.
    total_val_accuracy = 0.

    for i in range(val_batches):
        x_support_set, y_support_set, x_target, y_target = data.get_val_batch(val_batches)
        x_support_set = Variable(x_support_set).to(device)
        y_support_set = Variable(y_support_set).to(device)
        x_target = Variable(x_target).to(device)
        y_target = Variable(y_target).to(device)
        preds, target_label = model(x_support_set, y_support_set, x_target, y_target)

        #produce predictions for target probablities
        correct_prediction = torch.argmax(preds, 1) == target_label
        accuracy = torch.mean(correct_prediction)
        targets = target_label.scatter_(1, target_label, num_classes_per_set)

        loss = F.cross_entropy(targets, preds)
        val_loss += loss.data[0]
        val_acc += accuracy

    total_val_loss /= val_batches
    total_val_accuracy /= val_batches

    return total_val_loss, total_val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type= int, default=5)
    parser.add_argument('--shot', type= int, default=1)
    parser.add_argument('--is_test', action= 'store_true')
    parser.add_argument('--lr', type=float, default= 1e-3)
    parser.add_argument('--epochs','-e', type=int, default= 100)
    parser.add_argument('--ckp', type=int, default= -1)
    args=parser.parse_args()
    print (args)

    #split
    sp = 1
    lr = args.lr
    total_epochs = args.epochs
    batch_size = int(32//sp)
    classes_per_set = args.ways
    samples_per_class = args.shot

    continue_from_epoch = args.ckp
    logs_path = 'one_shot_outputs/'
    experiment_name = 'LGM_{classes_per_set}way{samples_per_class}shot'

    data = dataset.MiniImageNetDataSet(batch_size, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    one_shot_learner = MetaMatchingNetwork(num_classes_per_set=classes_per_set, num_samples_per_class=samples_per_class)

    total_train_batches = 1000
    total_val_batches = int(250 * sp)
    total_test_batches = int(250 * sp)

    optimizer = Adam(one_shot_learner.parameters(), lr= args.lr )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    one_shot_learner.to(device)
    with tqdm(total=total_epochs) as pbar:
        for epoch in range(total_epochs):
            #Train 
            
            epoch_loss, epoch_acc, lr = train(total_train_batches, data, one_shot_learner, optimizer, device)
            iter_out = "Epoch: {}/{}, Loss: {:0.5f}, Accuracy: {:0.5f}, Lr: {:0.5f}".format(epoch, total_epochs, epoch_loss, epoch_acc, lr)

            pbar.set_description(iter_out)
            pbar.update(1)

            #Validate
            epoch_loss, epoch_acc =validation(total_val_batches, data, model, device)
            iter_out = "Validation Epoch: {} --- Loss: {:0.5f}, Accuracy: {:0.5f}".format(epoch, epoch_loss, epoch_acc)
            pbar.set_description(iter_out)
            pbar.update(1)
            


