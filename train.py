import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import *
from model.gat import *
from util.misc import CSVLogger

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    type=int,
                    default=32,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs',
                    type=int,
                    default=80,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--in_dim',
                    type=int,
                    default=47 + 657,
                    help='dim of atom feature')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--seed',
                    type=int,
                    default=123,
                    help='random seed (default: 123)')
parser.add_argument('--logdir', type=str, default='logs', help='logdir name')
parser.add_argument('--dataset', type=str, default='USPTO50K', help='dataset')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
parser.add_argument('--heads', type=int, default=4, help='number of heads')
parser.add_argument('--gat_layers',
                    type=int,
                    default=3,
                    help='number of gat layers')
parser.add_argument('--valid_only',
                    action='store_true',
                    default=False,
                    help='valid_only')
parser.add_argument('--test_only',
                    action='store_true',
                    default=False,
                    help='test_only')
parser.add_argument('--test_on_train',
                    action='store_true',
                    default=False,
                    help='run testing on training data')
parser.add_argument('--typed',
                    action='store_true',
                    default=False,
                    help='if given reaction types')
parser.add_argument('--use_cpu',
                    action='store_true',
                    default=False,
                    help='use gpu or cpu')
parser.add_argument('--load',
                    action='store_true',
                    default=False,
                    help='load model checkpoint.')

args = parser.parse_args()


def collate(data):
    return map(list, zip(*data))


def test(GAT_model, test_dataloader, data_split='test', save_pred=False):
    GAT_model.eval()
    correct = 0.
    total = 0.
    epoch_loss = 0.
    # Bond disconnection probability
    pred_true_list = []
    pred_logits_mol_list = []
    # Bond disconnection number gt and prediction
    bond_change_gt_list = []
    bond_change_pred_list = []
    for i, data in enumerate(tqdm(test_dataloader)):
        rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

        x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
        x_pattern_feat = list(
            map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
        x_atom = list(
            map(lambda x, y: torch.cat([x, y], dim=1), x_atom, x_pattern_feat))

        if args.typed:
            rxn_class = list(
                map(lambda x: torch.from_numpy(x).float(), rxn_class))
            x_atom = list(
                map(lambda x, y: torch.cat([x, y], dim=1), x_atom, rxn_class))

        x_atom = torch.cat(x_atom, dim=0)
        disconnection_num = torch.LongTensor(disconnection_num)
        if not args.use_cpu:
            x_atom = x_atom.cuda()
            disconnection_num = disconnection_num.cuda()

        x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
        y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
        if not args.use_cpu:
            x_adj = [xa.cuda() for xa in x_adj]
            y_adj = [ye.cuda() for ye in y_adj]

        mask = list(map(lambda x: x.view(-1, 1).bool(), x_adj))
        bond_disconnections = list(
            map(lambda x, y: torch.masked_select(x.view(-1, 1), y), y_adj,
                mask))
        bond_labels = torch.cat(bond_disconnections, dim=0).float()

        # batch graph
        g_dgl = dgl.batch(x_graph)
        h_pred, e_pred = GAT_model(g_dgl, x_atom)
        e_pred = e_pred.squeeze()
        loss_h = nn.CrossEntropyLoss(reduction='sum')(h_pred,
                                                      disconnection_num)
        loss_ce = nn.BCEWithLogitsLoss(reduction='sum')(e_pred, bond_labels)
        loss = loss_ce + loss_h
        epoch_loss += loss.item()

        h_pred = torch.argmax(h_pred, dim=1)
        bond_change_pred_list.extend(h_pred.cpu().tolist())
        bond_change_gt_list.extend(disconnection_num.cpu().tolist())

        start = end = 0
        pred = torch.sigmoid(e_pred)
        edge_lens = list(map(lambda x: x.shape[0], bond_disconnections))
        cur_batch_size = len(edge_lens)
        bond_labels = bond_labels.long()
        for j in range(cur_batch_size):
            start = end
            end += edge_lens[j]
            label_mol = bond_labels[start:end]
            pred_proab = pred[start:end]
            mask_pos = torch.nonzero(x_adj[j]).tolist()
            assert len(mask_pos) == len(pred_proab)

            pred_disconnection_adj = torch.zeros_like(x_adj[j], dtype=torch.float32)
            for idx, pos in enumerate(mask_pos):
                pred_disconnection_adj[pos[0], pos[1]] = pred_proab[idx]
            for idx, pos in enumerate(mask_pos):
                pred_proab[idx] = (pred_disconnection_adj[pos[0], pos[1]] + pred_disconnection_adj[pos[1], pos[0]]) / 2

            pred_mol = pred_proab.round().long()
            if torch.equal(pred_mol, label_mol):
                correct += 1
                pred_true_list.append(True)
                pred_logits_mol_list.append([
                    True,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])
            else:
                pred_true_list.append(False)
                pred_logits_mol_list.append([
                    False,
                    label_mol.tolist(),
                    pred_proab.tolist(),
                ])
            total += 1

    pred_lens_true_list = list(
        map(lambda x, y: x == y, bond_change_gt_list, bond_change_pred_list))
    bond_change_pred_list = list(
        map(lambda x, y: [x, y], bond_change_gt_list, bond_change_pred_list))
    if save_pred:
        print('pred_true_list size:', len(pred_true_list))
        np.savetxt('logs/{}_disconnection_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(bond_change_pred_list),
                   fmt='%d')
        np.savetxt('logs/{}_result_{}.txt'.format(data_split, args.exp_name),
                   np.asarray(pred_true_list),
                   fmt='%d')
        with open('logs/{}_result_mol_{}.txt'.format(data_split, args.exp_name),
                  'w') as f:
            for idx, line in enumerate(pred_logits_mol_list):
                f.write('{} {}\n'.format(idx, line[0]))
                f.write(' '.join([str(i) for i in line[1]]) + '\n')
                f.write(' '.join([str(i) for i in line[2]]) + '\n')

    print('Bond disconnection number prediction acc: {:.6f}'.format(
        np.mean(pred_lens_true_list)))
    print('Loss: ', epoch_loss / total)
    acc = correct / total
    print('Bond disconnection acc (without auxiliary task): {:.6f}'.format(acc))
    return acc


if __name__ == '__main__':
    batch_size = args.batch_size
    epochs = args.epochs
    data_root = os.path.join('data', args.dataset)
    args.exp_name = args.dataset
    if args.typed:
        args.in_dim += 10
        args.exp_name += '_typed'
    else:
        args.exp_name += '_untyped'
    print(args)

    test_id = '{}'.format(args.logdir)
    filename = 'logs/' + test_id + '.csv'
    csv_logger = CSVLogger(
        args=args,
        fieldnames=['epoch', 'train_acc', 'valid_acc', 'train_loss'],
        filename=filename,
    )

    GAT_model = GATNet(
        in_dim=args.in_dim,
        num_layers=args.gat_layers,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        use_gpu=(args.use_cpu == False),
    )

    if args.use_cpu:
        device = 'cpu'
    else:
        GAT_model = GAT_model.cuda()
        device = 'cuda:0'

    if args.load:
        GAT_model.load_state_dict(
            torch.load('checkpoints/{}_checkpoint.pt'.format(args.exp_name),
                       map_location=torch.device(device)), )
        args.lr *= 0.2
        milestones = []
    else:
        milestones = [20, 40, 60, 80]

    optimizer = torch.optim.Adam([{
        'params': GAT_model.parameters()
    }],
                                 lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.2)

    if args.test_only:
        test_data = RetroCenterDatasets(root=data_root, data_split='test')
        test_dataloader = DataLoader(test_data,
                                     batch_size=4 * batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=collate)
        test(GAT_model, test_dataloader, data_split='test', save_pred=True)
        exit(0)

    valid_data = RetroCenterDatasets(root=data_root, data_split='valid')
    valid_dataloader = DataLoader(valid_data,
                                  batch_size=4 * batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  collate_fn=collate)
    if args.valid_only:
        test(GAT_model, valid_dataloader)
        exit(0)

    train_data = RetroCenterDatasets(root=data_root, data_split='train')
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=collate)
    if args.test_on_train:
        test_train_dataloader = DataLoader(train_data,
                                           batch_size=8 * batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           collate_fn=collate)
        test(GAT_model, test_train_dataloader, data_split='train', save_pred=True)
        exit(0)

    # Record epoch start time
    for epoch in range(1, 1 + epochs):
        total = 0.
        correct = 0.
        epoch_loss = 0.
        epoch_loss_ce = 0.
        epoch_loss_h = 0.
        GAT_model.train()
        progress_bar = tqdm(train_dataloader)
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))
            rxn_class, x_pattern_feat, x_atom, x_adj, x_graph, y_adj, disconnection_num = data

            x_atom = list(map(lambda x: torch.from_numpy(x).float(), x_atom))
            x_pattern_feat = list(
                map(lambda x: torch.from_numpy(x).float(), x_pattern_feat))
            x_atom = list(
                map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                    x_pattern_feat))

            if args.typed:
                rxn_class = list(
                    map(lambda x: torch.from_numpy(x).float(), rxn_class))
                x_atom = list(
                    map(lambda x, y: torch.cat([x, y], dim=1), x_atom,
                        rxn_class))

            x_atom = torch.cat(x_atom, dim=0)
            disconnection_num = torch.LongTensor(disconnection_num)
            if not args.use_cpu:
                x_atom = x_atom.cuda()
                disconnection_num = disconnection_num.cuda()

            x_adj = list(map(lambda x: torch.from_numpy(np.array(x)), x_adj))
            y_adj = list(map(lambda x: torch.from_numpy(np.array(x)), y_adj))
            if not args.use_cpu:
                x_adj = [xa.cuda() for xa in x_adj]
                y_adj = [ye.cuda() for ye in y_adj]

            mask = list(map(lambda x: x.view(-1, 1).bool(), x_adj))
            bond_connections = list(
                map(lambda x, y: torch.masked_select(x.view(-1, 1), y), y_adj,
                    mask))
            bond_labels = torch.cat(bond_connections, dim=0).float()

            GAT_model.zero_grad()
            # batch graph
            g_dgl = dgl.batch(x_graph)
            h_pred, e_pred = GAT_model(g_dgl, x_atom)
            e_pred = e_pred.squeeze()
            loss_h = nn.CrossEntropyLoss(reduction='sum')(h_pred,
                                                          disconnection_num)
            loss_ce = nn.BCEWithLogitsLoss(reduction='sum')(e_pred,
                                                            bond_labels)
            loss = loss_ce + loss_h
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_ce += loss_ce.item()
            epoch_loss_h += loss_h.item()

            start = end = 0
            pred = torch.round(torch.sigmoid(e_pred)).long()
            edge_lens = list(map(lambda x: x.shape[0], bond_connections))
            cur_batch_size = len(edge_lens)
            bond_labels = bond_labels.long()
            for j in range(cur_batch_size):
                start = end
                end += edge_lens[j]
                if torch.equal(pred[start:end], bond_labels[start:end]):
                    correct += 1
            assert end == len(pred)

            total += cur_batch_size
            progress_bar.set_postfix(
                loss='%.5f' % (epoch_loss / total),
                acc='%.5f' % (correct / total),
                loss_ce='%.5f' % (epoch_loss_ce / total),
                loss_h='%.5f' % (epoch_loss_h / total),
            )

        scheduler.step(epoch)
        train_acc = correct / total
        train_loss = epoch_loss / total
        print('Train Loss: {:.5f}'.format(train_loss))
        print('Train Bond Disconnection Acc: {:.5f}'.format(train_acc))

        if epoch % 5 == 0:
            valid_acc = test(GAT_model, valid_dataloader)
            row = {
                'epoch': str(epoch),
                'train_acc': str(train_acc),
                'valid_acc': str(valid_acc),
                'train_loss': str(train_loss),
            }
            csv_logger.writerow(row)

    csv_logger.close()
    torch.save(GAT_model.state_dict(),
               'checkpoints/{}_checkpoint.pt'.format(args.exp_name))
