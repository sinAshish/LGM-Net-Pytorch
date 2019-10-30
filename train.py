import os
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
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from utils import save_statistics
warnings.filterwarnings('ignore')

def train(train_batches, data, model, optimizer, device):
    total_c_loss = 0.
    total_c_accuracy = 0.
    correct = 0
    
    with tqdm(total=train_batches) as pbar:
        for i in range(train_batches):
            x_support_set, y_support_set, x_target, y_target = data.get_train_batch(augment=True)
            x_support_set = Variable(x_support_set).to(device)
            y_support_set = Variable(y_support_set).to(device)
            x_target = Variable(x_target).to(device)
            y_target = Variable(y_target).to(device)
            preds, target_label = model(x_support_set, y_support_set, x_target, y_target)

            #produce predictions for target probablities
            #import pdb; pdb.set_trace()
            target_label  = torch.tensor(target_label, dtype= torch.long)
            #correct_prediction = (torch.argmax(preds, 1) == target_label)
            _, predicted = torch.max(preds.data, 1)
            total_train = target_label.size(0)
            correct += predicted.eq(target_label.data).sum().item()

            accuracy = 100 * correct / total_train
            #targets = target_label.scatter(1, target_label, classes_per_set)
            targets = F.one_hot(target_label)

            optimizer.zero_grad()
            loss = F.cross_entropy(preds, torch.max(targets.float(), 1)[1])
            loss.backward()
            optimizer.step()

            total_c_loss += loss.item()
            total_c_accuracy += accuracy
            
            iter_out = "train_loss: {:0.5f}, train_accuracy: {:0.5f}".format(loss.item(), accuracy)
            pbar.set_description(iter_out)
            pbar.update(1)
            
        total_c_loss /= train_batches
        total_c_accuracy /=train_batches

    return total_c_loss, total_c_accuracy

def testing(total_test_batches, data, model, device):
    total_test_loss = 0.
    total_test_accuracy = 0.
    correct = 0.
    with torch.no_grad():
        with tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = data.get_test_batch(total_test_batches)
                x_support_set = Variable(x_support_set).to(device)
                y_support_set = Variable(y_support_set).to(device)
                x_target = Variable(x_target).to(device)
                y_target = Variable(y_target).to(device)
                preds, target_label = model(x_support_set, y_support_set, x_target, y_target)

                target_label  = torch.tensor(target_label, dtype= torch.long)
                #correct_prediction = (torch.argmax(preds, 1) == target_label)
                _, predicted = torch.max(preds.data, 1)
                total_val = target_label.size(0)
                correct += predicted.eq(target_label.data).sum().item()

                
                accuracy = 100 * correct / total_val
                #targets = target_label.scatter(1, target_label, classes_per_set)
                targets = F.one_hot(target_label)

                loss = F.cross_entropy(preds, torch.max(targets.float(), 1)[1])
                total_test_loss += loss.item()
                total_test_accuracy += accuracy

                iter_out = "test_loss: {:0.5f}, test_accuracy: {:0.5f}".format(loss.item(), accuracy)
                pbar.set_description(iter_out)
                pbar.update(1)

            total_test_accuracy/= total_test_batches
            total_test_loss/= total_test_batches

    return total_test_loss, total_test_accuracy

def validation(val_batches, data, model, device):
    total_val_loss = 0.
    total_val_accuracy = 0.
    correct=0.
    with torch.no_grad():
        with tqdm(total=val_batches) as pbar:
            for i in range(val_batches):
                x_support_set, y_support_set, x_target, y_target = data.get_val_batch(val_batches)
                x_support_set = Variable(x_support_set).to(device)
                y_support_set = Variable(y_support_set).to(device)
                x_target = Variable(x_target).to(device)
                y_target = Variable(y_target).to(device)
                preds, target_label = model(x_support_set, y_support_set, x_target, y_target)

                target_label  = torch.tensor(target_label, dtype= torch.long)
                #correct_prediction = (torch.argmax(preds, 1) == target_label)
                _, predicted = torch.max(preds.data, 1)
                total_val = target_label.size(0)
                correct += predicted.eq(target_label.data).sum().item()

                accuracy = 100 * correct / total_val
                #targets = target_label.scatter(1, target_label, classes_per_set)
                targets = F.one_hot(target_label)


                loss = F.cross_entropy(preds, torch.max(targets.float(), 1)[1])
                total_val_loss += loss.item()
                total_val_accuracy += accuracy

                iter_out = "val_loss: {:0.5f}, val_accuracy: {:0.5f}".format(loss.item(), accuracy)
                pbar.set_description(iter_out)
                pbar.update(1)

            total_val_loss /= val_batches
            total_val_accuracy /= val_batches

    return total_val_loss, total_val_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ways','-w', type= int, default=5)
    parser.add_argument('--shot','-s', type= int, default=1)
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
    experiment_name = f'LGM_{classes_per_set}way_{samples_per_class}shot'
    logs="{}way{}shot learning problems, with {} tasks per task batch".format(classes_per_set, samples_per_class, batch_size)
    save_statistics(experiment_name, ["Experimental details: {}".format(logs)])
    save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy",
                                      "test_c_loss", "test_c_accuracy"])

    data = dataset.MiniImageNetDataSet(batch_size, classes_per_set=classes_per_set, samples_per_class=samples_per_class)
    one_shot_learner = MetaMatchingNetwork(num_classes_per_set=classes_per_set, num_samples_per_class=samples_per_class)

    total_train_batches = 1
    total_val_batches = int(2 * sp)
    total_test_batches = int(2 * sp)

    optimizer = Adam(one_shot_learner.parameters(), lr= args.lr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    one_shot_learner.to(device)

    writer = SummaryWriter(log_dir=logs_path)
    with tqdm(total=total_epochs) as pbar:
        for epoch in range(total_epochs):
            #Train 
            
            train_epoch_loss, train_epoch_acc = train(total_train_batches, data, one_shot_learner, optimizer, device)
            iter_out = "Training Epoch: {}/{}, Loss: {:0.5f}, Accuracy: {:0.5f}".format(epoch, total_epochs, train_epoch_loss, train_epoch_acc)
            writer.add_scalar('Loss/train', train_epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', train_epoch_acc, epoch)
            pbar.set_description(iter_out)
            pbar.update(1)

            #Validate
            val_epoch_loss, val_epoch_acc =validation(total_val_batches, data, one_shot_learner, device)
            iter_out = "Validation Epoch: {} --- Loss: {:0.5f}, Accuracy: {:0.5f}".format(epoch, val_epoch_loss, val_epoch_acc)
            writer.add_scalar('Loss/val', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/val', val_epoch_acc, epoch)
            pbar.set_description(iter_out)
            pbar.update(1)

            #Test
            test_epoch_loss, test_epoch_acc =testing(total_test_batches, data, one_shot_learner, device)
            iter_out = "Testing Epoch: {} --- Loss: {:0.5f}, Accuracy: {:0.5f}".format(epoch, test_epoch_loss, test_epoch_acc)
            writer.add_scalar('Loss/test', test_epoch_loss, epoch)
            writer.add_scalar('Accuracy/test', test_epoch_acc, epoch)
            pbar.set_description(iter_out)

            save_statistics(experiment_name, [epoch, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc,
                                                      test_epoch_loss, test_epoch_acc])
            if not os.path.exists('saved_models/'):
                os.makedirs('saved_models',exist_ok=False)
            save_path = "saved_models/{}_{}.pth".format(experiment_name, epoch)
            
            torch.save(one_shot_learner.state_dict(), save_path)
            pbar.update(1)
            


