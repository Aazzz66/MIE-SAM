import numpy as np
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import time
import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torchvision import transforms
from data import SalObjDataset ,test_dataset
from utils import *
from torch.autograd import Variable
from IPython.display import clear_output
from MIESAM import Net as MIENet
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener


net = MIENet(num_classes=N_CLASSES).cuda()

params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('All Params:   ', params)

params1 = 0
params2 = 0
for name, param in net.image_encoder.named_parameters():
    if "lora_" not in name:
        params1 += param.nelement()
    else:
        params2 += param.nelement()
print('ImgEncoder:   ', params1)
print('Lora: ', params2)
print('Others: ', params-params1-params2)

#train_set = SalObjDataset(train_root=train_root, trainsize=trainsize,max_samples=100)
train_set = SalObjDataset(train_root=train_root, trainsize=trainsize)
train_loader = get_loader(train_root=train_root, batchsize=BATCH_SIZE, trainsize=trainsize)
test_set = test_dataset(test_root, testsize=256)



base_lr = 0.001 # 设置基础学习率
params_dict = dict(net.named_parameters())
params = []

#optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0005)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs, eta_min=1e-6)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35,55,75], gamma=0.1)
def test(net, test_dataset, all=False, stride=WINDOW_SIZE[0], batch_size=1, window_size=WINDOW_SIZE,visualize=False,
         save_dir='./results/ts/VT1000'):

    net.eval()
    all_preds = []
    all_gts = []
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    if visualize:
        os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for image, t, gt, shape, name in test_loader:
            image = image.cuda()
            t = t.cuda()
            gt = gt.cuda()


            output = net(image, t, mode='Test')
            pred = output.cpu().numpy()
            gt = gt.cpu().numpy()
            gt = np.resize(gt, pred.shape)

            if visualize:
                pred1 = (pred * 255).astype(int)
                gt1 = (gt * 255).astype(int)
                pred_color = convert_to_grayscale(pred1)
                pred_color_squeezed = np.squeeze(pred_color)
                pred_color=pred_color_squeezed
                if isinstance(name, (list, tuple)):
                    filename = name[0]
                else:
                    filename = name

                if filename.endswith('.jpg'):
                    filename = filename.replace('.jpg', '.png')
                elif not filename.endswith('.png'):
                    filename = filename + '.png'
                pred_path = os.path.join(save_dir, filename)


                cv2.imwrite(pred_path, pred_color)
        all_preds.append(pred)
        all_gts.append(gt)
        pred_array = np.stack(all_preds)
        gt_array = np.stack(all_gts)

        pred_tensor = torch.from_numpy(pred_array).float().unsqueeze(1)
        gt_tensor = torch.from_numpy(gt_array).float()

        # 计算指标
        bce, aiou1, mae = LossFunc(pred_tensor, gt_tensor)
        if all:
            return bce, aiou1, mae, all_preds, all_gts
        else:
            return bce, aiou1, mae


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    # Initialize loss tracking

    train_loss_history = []
    val_loss_history = []
    val_metrics_history = {
        'bce': [],
        'aiou1': [],
        'mae': []
    }
    train_metrics_history = {
        'bce': [],
        'aiou1': [],
        'mae': []
    }
    lr_history = []
    iter_ = 0
    best_mae = float('inf')
    for e in range(1, epochs + 1):
        net.train()
        start_time = time.time()
        epoch_loss = 0
        epoch_bce = 0
        epoch_aiou1 = 0
        epoch_mae = 0
        batch_count = 0

        for batch_idx, (data, t , target) in enumerate(train_loader):
            data, t, target = Variable(data.cuda()), Variable(t.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data, t , mode='Train')
            target = target.long()
            bce, aiou1, mae = LossFunc(output, target)
            loss = bce + aiou1 + mae


            epoch_loss += loss.detach().item()
            epoch_bce += bce.item()
            epoch_aiou1 += aiou1.item()
            epoch_mae += mae.item()
            batch_count += 1

            loss.backward()
            optimizer.step()

            if iter_ % 1 == 0:
                clear_output()
                print(
                    f'Train (epoch {e}/{epochs}) [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')


            iter_ += 1

            del (data, t, target, loss)
            torch.cuda.empty_cache()
        epoch_loss/=(batch_idx+1)
        epoch_bce/= (batch_idx+1)
        epoch_aiou1 /= (batch_idx + 1)
        epoch_mae /= (batch_idx + 1)


        train_loss_history.append(epoch_loss)
        train_metrics_history['bce'].append(epoch_bce)
        train_metrics_history['aiou1'].append(epoch_aiou1)
        train_metrics_history['mae'].append(epoch_mae)
        print(f"Train Metrics - BCE: {epoch_bce:.4f} | aIoU: {epoch_aiou1:.4f} | MAE: {epoch_mae:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        net.eval()
        val_bce, val_aiou1, val_mae = 0.0, 0.0, 0.0
        val_batch_count = 0

        bce, aiou1, mae = test(net, test_set, all=False, visualize=True,
                                           save_dir='./results/2026.04',
                                           stride=Stride_Size)
        val_bce += bce.item()
        val_aiou1 += aiou1.item()
        val_mae += mae.item()
        val_batch_count += 1

        val_bce /= val_batch_count
        val_aiou1 /= val_batch_count
        val_mae /= val_batch_count
        val_total_loss = val_bce + val_aiou1 + val_mae
        val_loss_history.append(val_total_loss)
        val_metrics_history['bce'].append(val_bce)
        val_metrics_history['aiou1'].append(val_aiou1)
        val_metrics_history['mae'].append(val_mae)
        print(f"Validation Metrics - BCE: {val_bce:.4f} | aIoU: {val_aiou1:.4f} | MAE: {val_mae:.4f}")


        if e % save_epoch == 0:
            train_time = time.time()
            print("Training time: {:.3f} seconds".format(train_time - start_time))

            if val_mae < best_mae:
                best_mae = val_mae
                #model_path = f"./resultsv/256*256_Adam_best_model_epoch{e}_mae{val_mae:.4f}.pth"
                model_path = f"./resultsv/2026.4.19.pth"
                torch.save(net.state_dict(), model_path)

                best_save_dir = './results/best_model_images'
                test(net, test_set, all=False, visualize=True, save_dir=best_save_dir, stride=Stride_Size)

                # print(f"New best model saved: {model_path}")
                print(f"Model saved to {model_path} (LR={current_lr:.6f})")
            if scheduler is not None:
                 scheduler.step()

        print(f'Training completed. Best validation MAE: {best_mae:.4f}')
        plot_loss_curves(train_loss_history,train_metrics_history,
                         val_loss_history, val_metrics_history,
                         lr_history,
                         save_path="./results/T/2026.4.png")




if MODE == 'Train':
    train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)

elif MODE == 'Test':

    test_set = test_dataset(test_root, testsize=256)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    model_path = "./resultsv/.."
    net.load_state_dict(torch.load(model_path), strict=False)
    net.eval()
    save_dir = './results/ts/VT1000'
    bce, aiou1, mae, all_preds, all_gts = test(net, test_set, all=True, stride=32, visualize=True,save_dir=save_dir)

    print(f"Test Results - BCE: {bce:.4f} | aIoU: {aiou1:.4f} | MAE: {mae:.4f}")





