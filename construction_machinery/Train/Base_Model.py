# Custom
from Utils import *

import torch
import torch.nn as nn

import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')

# Knowledge distillation Loss
def distillation(student_logits, labels, teacher_logits, alpha):
    distillation_loss = nn.BCELoss()(student_logits, teacher_logits)
    student_loss = nn.BCELoss()(student_logits, labels.reshape(-1, 1))
    return alpha * student_loss + (1-alpha) * distillation_loss

def distill_loss(output, target, teacher_output, loss_fn, opt):
    loss_b = loss_fn(output, target, teacher_output, alpha=0.1)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item()

##################### Teacher Model #####################
# Train Function
def train(model, optimizer, train_loader, val_loader, scheduler, device, CFG):
    model.to(device)

    best_score = 0
    best_model = None
    criterion = nn.BCELoss().to(device)

    for epoch in range(CFG["EPOCHS"]):
        train_loss = []

        model.train()
        for X, _, y in train_loader:
            X = X.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            y_pred = model(X)

            loss = criterion(y_pred, y.reshape(-1, 1))
            loss.backward()

            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_score = validation_teacher(model, val_loader, criterion, device)
        if epoch % CFG['Print_epoch'] == 0:
            print(
                f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)

        if best_score < val_score:
            best_model = model
            best_score = val_score

    print('Best Score: {:.5f}'.format(best_score))

    return best_model, best_score

# Validation Function
def validation_teacher(model, val_loader, criterion, device):
    model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.5

    with torch.no_grad():
        for X, _, y in val_loader:
            X = X.float().to(device)
            y = y.float().to(device)

            model_pred = model(X.to(device))

            loss = criterion(model_pred, y.reshape(-1, 1))
            val_loss.append(loss.item())

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1

##################### Student Model #####################
# Train
def student_train(s_model, t_model, optimizer, train_loader, val_loader, scheduler, device, CFG):
    s_model.to(device)
    t_model.to(device)

    best_score = 0
    best_model = None

    for epoch in range(CFG["EPOCHS"]):
        train_loss = []
        s_model.train()
        t_model.eval()

        for X_t, X_s, y in train_loader:
            X_t = X_t.float().to(device)
            X_s = X_s.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            output = s_model(X_s)
            with torch.no_grad():
                teacher_output = t_model(X_t)

            loss_b = distill_loss(output, y, teacher_output, loss_fn=distillation, opt=optimizer)

            train_loss.append(loss_b)

        val_loss, val_score = validation_student(s_model, t_model, val_loader, distill_loss, device)
        if epoch % CFG['Print_epoch'] == 0:
            print(
                f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss) :.5f}] Val Loss : [{np.mean(val_loss) :.5f}] Val F1 Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)

        if best_score < val_score:
            best_model = s_model
            best_score = val_score

    print('Best Score: {:.5f}'.format(best_score))

    return best_model, best_score

# Validation
def validation_student(s_model, t_model, val_loader, criterion, device):
    s_model.eval()
    t_model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.5

    with torch.no_grad():
        for X_t, X_s, y in val_loader:
            X_t = X_t.float().to(device)
            X_s = X_s.float().to(device)
            y = y.float().to(device)

            model_pred = s_model(X_s)
            teacher_output = t_model(X_t)

            loss_b = criterion(model_pred, y, teacher_output, loss_fn=distillation, opt=None)
            val_loss.append(loss_b)

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1

##################### Inference #####################
# Threshold
def choose_threshold(model, val_loader, device):
    model.to(device)
    model.eval()

    thresholds = np.arange(0.1, 1, 0.1)
    pred_labels = []
    true_labels = []

    best_score = 0
    best_thr = None
    with torch.no_grad():
        for _, x_s, y in val_loader:
            x_s = x_s.float().to(device)
            y = y.float().to(device)

            model_pred = model(x_s)

            model_pred = model_pred.squeeze(1).to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        for threshold in thresholds:
            pred_labels_thr = np.where(np.array(pred_labels) > threshold, 1, 0)
            score_thr = competition_metric(true_labels, pred_labels_thr)
            if best_score < score_thr:
                best_score = score_thr
                best_thr = threshold
    return best_thr, best_score

# Test
def test(model, test_x, threshold, save_path, device, file_path):
    # Test
    x = torch.Tensor(test_x).float().to(device)
    model_pred = model(x)
    model_pred = model_pred.squeeze(1).to('cpu').detach().numpy()
    test_predict = np.where(np.array(model_pred) > threshold, 1, 0)

    submit = pd.read_csv('/data/jyhwang/construction_machinery/Data/Raw/sample_submission.csv')
    submit['Y_LABEL'] = test_predict
    submit.to_csv(os.path.join(save_path, file_path), index=False)