import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from dumb_3d_dataset import Dumb3DDataset
from probabilistic_unet_3d import ProbabilisticUnet
from utils import l2_regularisation


def calculate_iou(predictions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    pred_flat = predictions.view(-1)
    masks_flat = masks.view(-1)
    intersection = torch.mean(pred_flat*masks_flat)
    union = torch.mean(pred_flat + masks_flat) - intersection
    return (intersection + 1e-6)/(union + 1e-6)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Dumb3DDataset(length=1024, num_classes=2)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    # train_indices, val_indices = indices[split:], indices[:split]
    train_indices, val_indices = indices[:4], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
    print("Number of training/test patches:",
          (len(train_indices), len(val_indices)))

    net = ProbabilisticUnet(input_channels=1, num_classes=2, num_filters=[
                            32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4, beta=1e-2)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 10
    for _ in tqdm(range(epochs)):
        for _, (patch, mask) in enumerate(tqdm(train_loader)):
            patch = patch.to(device)
            mask = mask.to(device)
            print(patch.shape, mask.shape)
            net.forward(patch, mask, training=True)
            elbo = net.elbo(mask)
            reg_loss = l2_regularisation(
                net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss
            print(f'{patch.shape=}, {mask.shape=}, {elbo=}, {loss=}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iou_score = []
        for _, (patch, mask) in enumerate(tqdm(val_loader)):
            mask = torch.squeeze(mask, 0).to(device)
            patch = patch.to(device)
            net.forward(patch, segm=None, training=False)
            num_preds = 4
            predictions = []
            for _ in range(num_preds):
                mask_pred = net.sample(testing=True)
                mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                predictions.append(mask_pred)
            predictions = torch.cat(predictions, 0)
            predictions = predictions.mean(0).nan_to_num()
            iou_score_iter = calculate_iou(predictions, mask)
            iou_score.append(iou_score_iter.cpu().numpy())

        iou_val = np.mean(iou_score)
        print(f'validation iou = {iou_val}')


if __name__ == '__main__':
    main()
