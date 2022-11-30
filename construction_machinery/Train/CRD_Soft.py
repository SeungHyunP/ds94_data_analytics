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
    student_logits = student_logits.reshape(-1, 1)
    teacher_logits = teacher_logits.reshape(-1, 1)

    distillation_loss = nn.BCELoss()(student_logits, teacher_logits)
    student_loss = nn.BCELoss()(student_logits, labels.reshape(-1, 1))
    return alpha * student_loss + (1 - alpha) * distillation_loss


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

            loss = model(X, y)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        val_loss, val_score = validation_teacher(model, val_loader, device)
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
def validation_teacher(model, val_loader, device):
    model.eval()

    val_loss = []
    pred_labels = []
    true_labels = []
    threshold = 0.5

    with torch.no_grad():
        for X, _, y in val_loader:
            X = X.float().to(device)
            y = y.float().to(device)

            loss = model(X, y)
            model_pred, cluster_0, cluster_1 = model.predict(X.to(device))

            val_loss.append(loss.item())

            model_pred = model_pred.to('cpu')
            pred_labels += model_pred.tolist()
            true_labels += y.tolist()

        pred_labels = np.where(np.array(pred_labels) > threshold, 1, 0)
        val_f1 = competition_metric(true_labels, pred_labels)
    return val_loss, val_f1

##################### Student Model #####################
# Train
def student_train(train_loader, val_loader, module_list, criterion_list, optimizer, opt, scheduler, device, CFG):
    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    s_model = module_list[0]
    t_model = module_list[-1]

    best_score = 0
    best_model = None

    for epoch in range(CFG["EPOCHS"]):
        train_loss = []
        s_model.train()
        t_model.eval()

        for idx, (X_t, X_s, y, index, contrast_idx) in enumerate(train_loader):
            X_t = X_t.float().to(device)
            X_s = X_s.float().to(device)
            y = y.float().to(device)
            index = torch.tensor(index).to(device)
            contrast_idx = torch.tensor(contrast_idx).cuda()

            optimizer.zero_grad()

            feat_s, loss_cls = s_model(X_s, y, is_feat=True)
            with torch.no_grad():
                feat_t, _ = t_model(X_t, y, is_feat=True)
                feat_t = [f.detach() for f in feat_t]

            # cls + kl div
            # loss_cls = criterion_cls(logit_s, y.reshape(-1, 1))
            logit_s, _, _ = s_model.predict(X_s)
            logit_t, _, _ = t_model.predict(X_t)

            loss_div = criterion_div(logit_s.reshape(-1, 1), logit_t.reshape(-1, 1))
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)

            loss = opt['gamma'] * loss_cls + opt['alpha'] * loss_div + opt['beta'] * loss_kd
            loss.backward()
            optimizer.step()

            train_loss.append(loss.to('cpu').detach().numpy())

        val_loss, val_score = validation_student(s_model, t_model, val_loader, device)
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
def validation_student(s_model, t_model, val_loader, device):
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

            model_pred, cluster_0, cluster_1 = s_model.predict(X_s)
            teacher_output, cluster_0, cluster_1 = t_model.predict(X_t)

            loss_b = distill_loss(model_pred, y, teacher_output, loss_fn=distillation, opt=None)
            val_loss.append(loss_b)

            model_pred = model_pred.to('cpu')
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

            model_pred, cluster_0, cluster_1 = model.predict(x_s)
            model_pred = model_pred.to('cpu')

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
    model_pred, _, _ = model.predict(x)
    model_pred = model_pred.to('cpu').detach().numpy()
    test_predict = np.where(np.array(model_pred) > threshold, 1, 0)

    submit = pd.read_csv('/data/jyhwang/construction_machinery/Data/Raw/sample_submission.csv')
    submit['Y_LABEL'] = test_predict
    submit.to_csv(os.path.join(save_path, file_path), index=False)