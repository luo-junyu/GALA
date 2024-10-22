import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from arguments import arg_parse
from utils.utils_data import get_dataset, split_confident_data
from utils.graph_aug import AugTransform
from utils.utils import *
from models.model import GNN
from tqdm import tqdm
import copy

def pseudo_consistency_loss(logits_s, logits_w, class_acc, T=1.0, p_cutoff=0.0, use_hard_labels=True):
    logits_w = logits_w.detach()
    pesudo_label = torch.softmax(logits_w, dim=-1).detach()
    max_probs, max_idx = torch.max(pesudo_label, dim=-1)
    # mask = max_probs.ge(p_cutoff).float().cuda()
    mask = max_probs.ge(p_cutoff * (class_acc[max_idx] / (2 - class_acc[max_idx]))).float().cuda()
    select = max_probs.ge(p_cutoff).long()
    

    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
    return masked_loss.mean(), mask.mean(), select, max_idx.long()

def graph_mixup(confident_data, inconfident_data):
    for idx in range(min(len(confident_data), len(inconfident_data))):
        conf_g = confident_data[idx]
        inconf_g = inconfident_data[idx]
        labeled_graph_cluster = conf_g.cluster.unique()
        unlabeled_graph_cluster = inconf_g.cluster.unique()
        if labeled_graph_cluster.shape[0] < 2 or unlabeled_graph_cluster.shape[0] < 2:
            continue
        # random choice a cluster to exchange
        label_subcluster = labeled_graph_cluster[torch.randint(labeled_graph_cluster.shape[0], (1,))]
        unlabeled_subcluster = unlabeled_graph_cluster[torch.randint(unlabeled_graph_cluster.shape[0], (1,))]
        # exchange the subgraph
        min_len = min((conf_g.cluster == label_subcluster).sum(), (inconf_g.cluster == unlabeled_subcluster).sum())

        if min_len <= 2:
            continue

        conf_g_idx = (conf_g.cluster == label_subcluster).nonzero().squeeze()[:min_len]
        inconf_g_idx = (inconf_g.cluster == unlabeled_subcluster).nonzero().squeeze()[:min_len]
        # copy to temp
        temp = copy.deepcopy(conf_g.x[conf_g_idx])
        conf_g = copy.deepcopy(conf_g)
        inconf_g = copy.deepcopy(inconf_g)
        conf_g.x[conf_g_idx] = inconf_g.x[inconf_g_idx]
        inconf_g.x[inconf_g_idx] = temp
    
    return confident_data, inconfident_data


@torch.no_grad()
def test(loader, model):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train(loader, model):
    model.eval()

    total_correct = 0
    for data_dict in loader:
        data = data_dict.to(device)
        _, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def run(seed=0):
    epochs = args.epochs
    tta_epoch = args.tta_epoch
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'ckpt')

    dataset, split_dataset = get_dataset(DS, path, args)
    (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset) = split_dataset

    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    train_transforms = AugTransform(args.aug)

    criterion = nn.CrossEntropyLoss()
    dataset_num_features = source_train_dataset[0].x.shape[1]
    print(f'num_features: {dataset_num_features}')

    setup_seed(seed)
    model = GNN(dataset_num_features, args.hidden_dim, args.num_gc_layers, dataset.num_classes, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)

    print('================')
    print(f'[{DS}] {args.source_index}->{args.target_index}')
    print('================')

    best_val_acc = 0.0
    final_test_acc = 0.0
    raw_target_acc = 0.0
    model_for_tta = None

    """
    Train on Source Domain 
    """

    for epoch in tqdm(range(1, epochs + 1)):
        loss_all = 0
        model.train()
        for data_dict in source_train_loader:
            data = data_dict.to(device)
            optimizer.zero_grad()
            x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
            pred_w = model.classifier(x_proj)
            loss = criterion(pred_w, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

    """
    Source Free Domain Adaptation
    """

    model_for_tta = copy.deepcopy(model)

    with torch.no_grad():
        confident_dataset, inconfident_dataset = \
            split_confident_data(model,target_train_dataset,device,args)

    confident_dataloader = DataLoader(confident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    inconfident_dataloader = DataLoader(inconfident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    best_val_acc = 0.0
    final_test_acc = 0.0

    def kldev_consistency_loss(pred_s, pred_w, epsilon=1e-7):
        log_probs = F.log_softmax(pred_s, dim=-1)
        pred_w = torch.clamp(F.softmax(pred_w, dim=-1), epsilon, 1.0)
        kl_div = F.kl_div(log_probs, pred_w, reduction='batchmean')
        return kl_div

    # Pseudo Label Learning on Target Domain
    for epoch in tqdm(range(1, tta_epoch + 1)):
        loss_all = 0
        model.train()

        for data_con, data_incon in zip(confident_dataloader, inconfident_dataloader):

            data_con = data_con.to(device)
            data_incon = data_incon.to(device)
            optimizer.zero_grad()
            loss = 0
            consistency_loss = 0
            pseudo_loss = 0

            _, x_proj = model(data_con.x, data_con.edge_index, data_con.batch, data_con.num_graphs)
            pred = model.classifier(x_proj)

            model_for_tta.eval()
            with torch.no_grad():
                _, x_p = model_for_tta(data_con.x, data_con.edge_index, data_con.batch, data_con.num_graphs)
                pred_p = model_for_tta.classifier(x_p)
                pseudo_label = pred_p.argmax(dim=-1)
            
            pseudo_label.detach()
            pseudo_loss = criterion(pred, pseudo_label) * 0.1
            loss += pseudo_loss

            if args.jigsaw:
                data_con_aug, _ = graph_mixup(data_con, data_incon)
                _, x_proj_con = model(data_con_aug.x, data_con_aug.edge_index, data_con_aug.batch, data_con_aug.num_graphs)
                pred_con = model.classifier(x_proj_con)
                consistency_loss = kldev_consistency_loss(pred_con, pred)
                loss = pseudo_loss + consistency_loss
            
            loss *= 0.1
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            test_acc = test(target_test_loader, model)
            tqdm.write(f'Loss: {loss_all / len(source_train_loader):.2f}, Test: {test_acc:.4f}')
    logger.info(f'{args.DS} {args.source_index}->{args.target_index}, {final_test_acc}')


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(args)
    run()
    