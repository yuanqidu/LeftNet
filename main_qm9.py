### Based on the code in https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph

from qm9_dataset import QM93D
from model import LEFTNet

import argparse
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import time

        
def run(device, train_dataset, valid_dataset, test_dataset, model, scheduler_name, loss_func, epochs=800, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
    save_dir='models/', log_dir='', disable_tqdm=False):     

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == 'steplr':
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
    elif scheduler_name == 'onecyclelr':
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs) 

    best_valid = float('inf')
    test_valid = float('inf')
        
    if save_dir != '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if log_dir != '':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    start_epoch = 1
    
    for epoch in range(start_epoch, epochs + 1):
        print("=====Epoch {}".format(epoch), flush=True)
        t_start = time.perf_counter()
        
        train_mae = train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm)
        valid_mae = val(model, valid_loader, device, disable_tqdm)
        test_mae = val(model, test_loader, device, disable_tqdm)


        if log_dir != '':
            writer.add_scalar('train_mae', train_mae, epoch)
            writer.add_scalar('valid_mae', valid_mae, epoch)
            writer.add_scalar('test_mae', test_mae, epoch)
        
        if valid_mae < best_valid:
            best_valid = valid_mae
            test_valid = test_mae
            if save_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

        t_end = time.perf_counter()
        print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae, 'Best valid': best_valid, 'Test@ best valid': test_valid, 'Duration': t_end-t_start})


        if scheduler_name == 'steplr':
            scheduler.step()

    print(f'Best validation MAE so far: {best_valid}')
    print(f'Test MAE when got best validation result: {test_valid}')
    
    if log_dir != '':
        writer.close()

def train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm):  
    model.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader, disable=disable_tqdm)):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        loss = loss_func(out, batch_data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if scheduler_name == 'onecyclelr':
            scheduler.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)

def val(model, data_loader, device, disable_tqdm):   
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    
    for step, batch_data in enumerate(tqdm(data_loader, disable=disable_tqdm)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        preds = torch.cat([preds, out.detach_()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

    return torch.mean(torch.abs(preds - targets)).cpu().item()


parser = argparse.ArgumentParser(description='QM9')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--target', type=str, default='U0')

parser.add_argument('--train_size', type=int, default=110000)
parser.add_argument('--valid_size', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--cutoff', type=float, default=5.0)
parser.add_argument('--num_radial', type=int, default=32)
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=4)

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--vt_batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save_dir', type=str, default='models/')
parser.add_argument('--disable_tqdm', default=False, action='store_true')
parser.add_argument('--scheduler', type=str, default='steplr')
parser.add_argument('--norm_label', default=False, action='store_true')

args = parser.parse_args()

print(args)
print(args.save_dir)

dataset = QM93D(root='dataset/')

target = args.target
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=args.train_size, valid_size=args.valid_size, seed=args.seed)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

if args.norm_label:
    y_mean = torch.mean(train_dataset.data.y).item()
    y_std = torch.std(train_dataset.data.y).item()
    print('y_mean, y_std:', y_mean, y_std)
else:
    y_mean = None
    y_std = None

model = LEFTNet(pos_require_grad=False, cutoff=args.cutoff, num_layers=args.num_layers,
            hidden_channels=args.hidden_channels, num_radial=args.num_radial, y_mean=y_mean, y_std=y_std)

loss_func = torch.nn.L1Loss()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print('device',device)

model.to(device)

run(device=device, 
    train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, 
    model=model, scheduler_name=args.scheduler, loss_func=loss_func, 
    epochs=args.epochs, batch_size=args.batch_size, vt_batch_size=args.batch_size, 
    lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, 
    weight_decay=args.weight_decay, 
    save_dir=args.save_dir, log_dir=args.save_dir, disable_tqdm=args.disable_tqdm)