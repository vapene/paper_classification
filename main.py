import argparse
import numpy as np
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
from param_parser import parameter_parser, tab_printer
from utils import RandomRotation, PaperDataset, EMA
from models import M3, M5, M7, resnet
from tensorboardX import SummaryWriter
writer = SummaryWriter()

def train(score_dict, MODEL, SEED, EPOCHS):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    ### create file & path
    if not os.path.exists(f"./outcome2"):
        os.makedirs(f"./outcome2")
    MODEL_FILE = str(f"./outcome2/{MODEL}_{SEED}.pt")

    ### transform
    transform = transforms.Compose([
        RandomRotation(20, seed=SEED),
        transforms.RandomAffine(0, translate=(0.2, 0.2)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)])
    ### loader
    train_dataset = PaperDataset(training=True, transform=transform)
    test_dataset = PaperDataset(training=False, transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=80, shuffle=True, drop_last=True)  # 800:6,
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=80, shuffle=False, drop_last=True)
    ### hyper-parameter
    model = eval(f"{MODEL}().to(device)")
    ema = EMA(model, decay=0.999)  # Exponential Weighted Moving Average
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    g_step = 0
    ### training
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_idx, (data,target,idx) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output,target)
            train_pred = output.argmax(dim = 1, keepdim = True)

            train_correct += train_pred.eq(target.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            loss.backward()
            optimizer.step()
            g_step +=1
            ema(model, g_step)

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * train_correct / len(train_loader.dataset)
        print('model',MODEL,'epoch',epoch,'train_loss',train_loss,'train_accuracy',train_accuracy)
        writer.add_scalar(f"loss/train_loss", train_loss, epoch)
        writer.add_scalar(f"acc/train_accuracy", train_accuracy, epoch)
        ### test
        model.eval()
        ema.assign(model)
        test_loss = 0
        test_correct = 0
        max_correct = 0
        with torch.no_grad():
            for data, target,idx in test_loader:
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
            if (max_correct < test_correct):
                torch.save(model.state_dict(), MODEL_FILE)
                max_correct = test_correct
                print(f"Best accuracy! correct images: {test_correct}")
        ema.resume(model)

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * test_correct / len(test_loader.dataset)
        best_test_accuracy = 100 * max_correct / len(test_loader.dataset)
        lr_scheduler.step()
        if epoch%20==0:
            print('model',MODEL,'epoch',epoch,'test_loss',test_loss,'test_accuracy',test_accuracy,'best_test_accuracy',best_test_accuracy)
        writer.add_scalar(f"loss/test_loss", test_loss, epoch)
        writer.add_scalar(f"acc/test_accuracy", test_accuracy, epoch)
        lr_scheduler.step()
    score_dict[MODEL] = score_dict[MODEL]+torch.Tensor([best_test_accuracy/100])
    return score_dict

def evaluate(SCORE_DICT, args):
    prediction_dict = {"M3": torch.zeros((1000,5)), "M5": torch.zeros((1000,5)), "M7": torch.zeros((1000,5)), "resnet": torch.zeros((1000,5))}
    target_dict = {"M3": torch.zeros(args.batch_size*10), "M5": torch.zeros(args.batch_size*10), "M7": torch.zeros(args.batch_size*10), "resnet": torch.zeros(args.batch_size*10)}
    test_dataset = PaperDataset(training=False, transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=80, shuffle=False, drop_last=True)

    start = 0
    end = 0
    for batch_idx, (data, target, idx) in enumerate(test_loader):
        end += args.batch_size
        for model in args.models:
            best_model = eval(f"{model}().to(device)")
            best_model.load_state_dict(torch.load(f"/home/soonwook34/jongwon/pretrain-gnns/MnistSimpleCNN/outcome2/{model}_{args.seed}.pt"))
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                output = best_model(data)
                output = torch.nn.functional.normalize(output.detach().cpu()) #* SCORE_DICT[model]
                prediction_dict[model][start:end,:] = prediction_dict[model][start:end,:] + output.detach().cpu()
                target_dict[model][start:end] = target_dict[model][start:end] + target.detach().cpu()
        start += args.batch_size
    return prediction_dict, target_dict, end

def ensemble(PREDICTION_DICT, TARGET_DICT, end):
    final_correct = 0
    summation = torch.zeros((end,5))
    for model in args.models:
        summation = np.add(summation,PREDICTION_DICT[model][:end])

    final_pred = summation.argmax(dim=1, keepdim=True)
    final_correct += final_pred.eq(TARGET_DICT[model][:end].view_as(final_pred)).sum().item()

    final_accuracy = 100 * final_correct / end
    print('final accuracy',final_accuracy)

    ### wrong txt to check ensemble power.
    # best_model = eval(f"{MODEL}().to(device)")
    # best_model.load_state_dict(torch.load(f"./outcome2/{MODEL}_{SEED}.pt"))
    # wrong_images = []
    # with torch.no_grad():
    #     for batch_idx, (data, target, idx) in enumerate(test_loader):
    #         data, target = data.to(device), target.to(device)
    #         output = best_model(data)
    #         best_loss += F.nll_loss(output, target, reduction='sum').item()
    #         best_pred = output.argmax(dim=1, keepdim=True)
    #         best_correct += best_pred.eq(target.view_as(best_pred)).sum().item()
    #         wrong_images.extend(np.nonzero(~best_pred.eq(target.view_as(best_pred)).cpu().numpy())[0]+(100*batch_idx))
    #         # array([7, 8, 9])
    # np.savetxt(f"./outcome2/{MODEL}_{SEED}_wrong.txt", wrong_images)


if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    device = torch.device("cuda:" + args.device)
    score_dict = {"M3":torch.Tensor([1e-10]), "M5":torch.Tensor([1e-10]), "M7":torch.Tensor([1e-10]), "resnet":torch.Tensor([1e-10])}
    for model in args.models:
        score_dict = train(score_dict, model, args.seed, args.epochs)
    print('157 score dict',score_dict)
    # score_dict =  {'M3': [58.02469135802469, 45.67901234567901, 52.46913580246913],
    #  'M5': [49.382716049382715, 51.23456790123457, 57.407407407407405],
    #  'M7': [58.02469135802469, 61.111111111111114, 50.0],
    #  'resnet': [74.07407407407408, 66.66666666666667, 71.60493827160494]}

    prediction_dict, target_dict, end = evaluate(score_dict, args)
    ensemble(prediction_dict, target_dict, end)
    print('scoredict',score_dict)
    writer.close()