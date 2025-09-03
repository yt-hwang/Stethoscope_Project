import os, argparse, random, numpy as np, torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .config import NUM_CLASSES, SEED
from .dataset import RespiratoryDataset
from .models.cnn_small import SmallCNN

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0; correct = 0; loss_sum = 0.0
    for x, y in tqdm(loader, desc="train"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total = 0; correct = 0; loss_sum = 0.0
    ys = []; ps = []
    for x, y in tqdm(loader, desc="eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        ys.extend(y.cpu().numpy().tolist())
        ps.extend(pred.cpu().numpy().tolist())
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total, ys, ps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = RespiratoryDataset(args.metadata)
    n_train = int(0.8*len(ds))
    n_val = len(ds) - n_train
    tr, va = random_split(ds, [n_train, n_val])

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SmallCNN(num_classes=3).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, opt, loss_fn, device)
        va_loss, va_acc, ys, ps = eval_epoch(model, dl_va, loss_fn, device)
        print(f"EP{ep}: train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best.pt'))
    # final report
    from sklearn.metrics import classification_report
    print(classification_report(ys, ps, digits=3))

if __name__ == '__main__':
    main()
