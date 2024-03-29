import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim import lr_scheduler
from model import MainNet
from dataloader import UAVDatasetTuple
from utils import draw_roc_curve, calculate_precision_recall, visualize_sum_testing_result, visualize_lstm_testing_result
from correlation import Correlation


os.environ["CUDA_VISIBLE_DEVICES"]="2"

def train(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    sum_running_loss = 0.0
    loss_mse = 0.0
    num_images = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        image = data['image'].to(device).float()
        init = data['init'].to(device).float()
        label = data['label'].to(device).float()
        #model prediction
        prediction = model(subx=image, mainx=init)
        #loss
        loss_mse = criterion(prediction, label.data)

        # update the weights within the model
        loss_mse.backward()
        optimizer.step()

        # accumulate loss
        if loss_mse != 0.0:
            sum_running_loss += loss_mse * image.size(0)
        num_images += image.size(0)

        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            sum_epoch_loss = sum_running_loss / num_images
            print('\nTraining phase: epoch: {} batch:{} Loss: {:.4f}\n'.format(epoch, batch_idx, sum_epoch_loss))


def val(model, test_loader, device, criterion, epoch, batch_size):
    model.eval()
    sum_running_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            image = data['image'].to(device).float()
            init = data['init'].to(device).float()
            label = data['label'].to(device).float()

            prediction = model(subx=image, mainx=init)
            # loss
            loss_mse = criterion(prediction, label.data)

            # accumulate loss
            sum_running_loss += loss_mse.item() * image.size(0)

            # visualize the sum testing result
            visualize_sum_testing_result(init, prediction, label.data, batch_idx, epoch, batch_size)

    sum_running_loss = sum_running_loss / len(test_loader.dataset)

    prediction_output = prediction.cpu().detach().numpy()
    label_output = label.cpu().detach().numpy()

    print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(epoch, sum_running_loss))
    return sum_running_loss, prediction_output, label_output

def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data path", required=True, type=str)
    parser.add_argument("--init_path", help="init path", required=True, type=str)
    parser.add_argument("--label_path", help="label path", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--momentum", help="momentum", required=True, type=float)
    parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
    parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
    parser.add_argument("--load_from_checkpoint", dest='load_from_checkpoint', action='store_true')
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    device = torch.device("cuda")

    all_dataset = UAVDatasetTuple(image_path=args.data_path, init_path=args.init_path, label_path=args.label_path)
    # positive_ratio, negative_ratio = all_dataset.get_class_count()
    # weight = torch.FloatTensor((positive_ratio, negative_ratio))
    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    print("Total image tuples for train: ", len(train_dataset))
    print("Total image tuples for test: ", len(test_dataset))

    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")
    model_ft = MainNet()
    model_ft = nn.DataParallel(model_ft)

    criterion  = nn.MSELoss(reduction='sum')

    if args.load_from_checkpoint:
        chkpt_model_path = os.path.join(args.checkpoint_dir, args.model_checkpoint_name)
        print("Loading ", chkpt_model_path)
        chkpt_model = torch.load(chkpt_model_path, map_location=device)
        model_ft.load_state_dict(chkpt_model)
        model_ft.eval()

    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=30,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30, drop_last=True)

    cor = Correlation()

    if args.eval_only:
        loss, prediction_output, label_output = val(model_ft, test_loader, device, criterion, 0)
        print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(0, loss))
        cor.corrcoef(prediction_output, label_output, ".", "correlation_test.png")
        return True

    best_loss = np.inf
    coef = 0.0
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)
        exp_lr_scheduler.step()
        train(model_ft, train_loader, device, optimizer_ft, criterion, epoch)
        loss, prediction_output, label_output = val(model_ft, test_loader, device, criterion, epoch, args.batch_size)
        if loss < best_loss:
            save_model(checkpoint_dir=args.checkpoint_dir,
                       model_checkpoint_name=args.model_checkpoint_name + '_' + str(loss),
                       model=model_ft)
            best_loss = loss
        cor_path = args.checkpoint_dir.replace("check_point", "testing_result")
        cor_path = os.path.join(cor_path, "epoch_" + str(epoch))
        coef = cor.corrcoef(prediction_output, label_output, cor_path, "correlation_{0}.png".format(epoch))
        print('correlation coefficient : {0}\n'.format(coef))

if __name__ == '__main__':
    main()