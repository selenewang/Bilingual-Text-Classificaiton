import argparse
import os
import pickle
import time

import numpy as np
import torch.nn.parallel
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch import optim

from .model import MLP





def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)







def train(lr=args.lr,
          n_emb=args.n_emb,
          n_hidden=args.n_hidden,
          batch_size=args.batch_size,
          valid_freq=3000,
          disp_freq=1000,
          save_freq=100000,
          max_epochs=args.n_epoch,
          patience=10,
          save_name=args.save_name,
          save_dir=args.save_dir,
          device=args.device):
    # Load train and valid dataset
    print('loading train')
    with open(args.en_train_path, 'rb') as f:
        en_y = np.array(pickle.load(f))
        en_x = pickle.load(f)
    print('loading english test')
    with open(args.en_test_path, 'rb') as f:
        en_test_y = np.array(pickle.load(f))
        en_test_x = pickle.load(f)
    print('loading french test')
    with open(args.fr_test_path, 'rb') as f:
        fr_test_y = np.array(pickle.load(f))
        fr_test_x = pickle.load(f)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1125)
    # _fr_test_y, _fr_y, _fr_test_x, _fr_x = train_test_split(fr_test_y, fr_test_x, test_size=0.2)
    # _fr_test_y = np.array(fr_test_y)[train_index]
    # _fr_test_x = fr_test_x[train_index, :]
    # _fr_y = np.array(fr_test_y)[test_index]
    # _fr_x = fr_test_x[test_index, :]
    # en_x = np.concatenate((en_x, _fr_x), axis=0)
    # print(en_y.shape, _fr_y.shape)
    # en_y = np.concatenate((en_y, _fr_y), axis=0)
    for train_index, test_index in sss.split(en_x, en_y):
        en_train_y = np.array(en_y)[train_index]
        en_train_x = en_x[train_index, :]
        en_valid_y = np.array(en_y)[test_index]
        en_valid_x = en_x[test_index, :]
    print('Number of training sample: %d' % en_train_x.shape[0])
    print('Number of validation sample: %d' % en_valid_x.shape[0])
    print('Number of english testing sample: %d' % en_test_x.shape[0])
    print('Number of french testing sample: %d' % fr_test_x.shape[0])
    print('-' * 100)

    kf_valid = get_minibatches_idx(len(en_valid_y), batch_size)
    kf_en_test = get_minibatches_idx(len(en_test_y), batch_size)

    # kf_fr_train = get_minibatches_idx(len(_fr_test_y), batch_size)
    kf_fr_test = get_minibatches_idx(len(fr_test_y), batch_size)
    # Loader parameter: use CUDA pinned memory for faster data loading
    pin_memory = (device == args.device)
    # Test set

    n_class = len(set(en_train_y))
    best_valid_err = None
    bad_counter = 0

    uidx = 0  # the number of update done
    estop = False  # early stop switch
    net = MLP(n_mlp_layer=args.n_mlp_layers,
              n_hidden=n_hidden,
              n_class=n_class,
              n_emb=args.n_emb,
              device=args.device)

    print('-' * 100)
    if args.load_net != '':
        assert os.path.exists(args.load_net), 'Path to pretrained net does not exist'
        net.load_state_dict(torch.load(args.load_net))
        print('Load exists model stored at: ', args.load_net)

    if args.device == 'gpu':
        net = net.cuda()

    # Begin Training
    net.train()
    print('-' * 100)
    print('Model structure: ')
    print('MLP baseline')
    print(net.main)
    print('-' * 100)
    print('Parameters for tuning: ')
    print(net.state_dict().keys())
    print('-' * 100)

    # Define optimizer
    assert args.optimizer in ['SGD', 'Adam'], 'Please choose either SGD or Adam'
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(lr=lr, params=filter(lambda p: p.requires_grad, net.parameters()), momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(params=filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    else:
        print('Optimizer cannot be defined')
    try:
        for eidx in range(max_epochs):
            # print('Training mode on: ' ,net.training)
            start_time = time.time()
            n_samples = 0
            # Get new shuffled index for the training set.
            # kf = get_minibatches_idx(len(en_train_y), batch_size, shuffle=True)
            kf = get_minibatches_idx(len(en_train_y), batch_size, shuffle=True)

            for _, train_index in kf:
                # Remove gradient from previous batch
                net.zero_grad()
                uidx += 1
                y_batch = torch.autograd.Variable(torch.from_numpy(en_train_y[train_index]).long())
                # y_batch = torch.autograd.Variable(torch.from_numpy(_fr_test_y[train_index]).long())
                x_batch = torch.autograd.Variable(torch.from_numpy(en_train_x[train_index]).float())
                # x_batch = torch.autograd.Variable(torch.from_numpy(_fr_test_x[train_index]).float())
                if net.device == 'gpu':
                    y_batch = y_batch.cuda()
                scores = net.forward(x_batch)
                loss = net.loss(scores, y_batch)

                loss.backward()
                optimizer.step()
                n_samples += len(x_batch)
                gradient = 0

                # For logging gradient information
                for name, w in net.named_parameters():
                    if w.grad is not None:
                        w_grad = torch.norm(w.grad.data, 2) ** 2
                        gradient += w_grad
                gradient = gradient ** 0.5
                if np.mod(uidx, disp_freq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', loss.data[0], 'Gradient ', gradient)

                if save_name and np.mod(uidx, save_freq) == 0:
                    print('Saving...')
                    torch.save(net.state_dict(), '%s/%s_epoch%d_update%d.net' % (save_dir, save_name, eidx, uidx))

                if np.mod(uidx, valid_freq) == 0:
                    print('Evaluation on validation set: ')
                    kf_valid = get_minibatches_idx(len(en_valid_y), batch_size)
                    top_1_err, top_n_err = eval.net_evaluation(net, kf_valid, en_valid_x, en_valid_y)
                    # Save best performance state_dict for testing
                    if best_valid_err is None:
                        best_valid_err = top_1_err
                        best_state_dict = net.state_dict()
                        torch.save(best_state_dict, '%s/%s_best.net' % (save_dir, save_name))
                    else:
                        if top_1_err < best_valid_err:
                            print('Best validation performance so far, saving model parameters')
                            bad_counter = 0  # reset counter
                            best_valid_err = top_1_err
                            best_state_dict = net.state_dict()
                            torch.save(best_state_dict, '%s/%s_best.net' % (save_dir, save_name))
                        else:
                            bad_counter += 1
                            print('-' * 100)
                            print('Validation error: ', top_1_err)
                            print('Getting worse, patience left: ', patience - bad_counter)
                            print('Best validation error now: ', best_valid_err)
                            # Learning rate annealing
                            lr /= args.lr_anneal
                            print('Learning rate annealed to: ', lr)
                            print('*' * 100)
                            if args.optimizer == 'SGD':
                                optimizer = optim.SGD(lr=lr, params=filter(lambda p: p.requires_grad, net.parameters()),
                                                      momentum=0.9)
                            elif args.optimizer == 'Adam':
                                optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
                            if bad_counter > patience:
                                print('-' * 100)
                                print('Early Stop!')
                                estop = True
                                break

            epoch_time = time.time() - start_time
            print('Epoch processing time: %.2f s' % epoch_time)
            print('Seen %d samples' % n_samples)
            if estop:
                break
        print('-' * 100)
        print('Training finish')
        best_state_dict = torch.load('%s/%s_best.net' % (save_dir, save_name))
        torch.save(net.state_dict(), '%s/%s_final.net' % (save_dir, save_name))
        net.load_state_dict(best_state_dict)

        # add self connection
        print('Evaluation on validation set: ')
        kf_valid = get_minibatches_idx(len(en_valid_y), batch_size)
        # kf_valid = get_minibatches_idx(len(_fr_y), batch_size)
        eval.net_evaluation(net, kf_valid, en_valid_x, en_valid_y)
        # eval.net_evaluation(net, kf_fr_test, _fr_x, _fr_y)

        # Evaluate model on test set
        print('Evaluation on test set: ')
        eval.net_evaluation(net, kf_en_test, en_test_x, en_test_y)
        eval.net_evaluation(net, kf_fr_test, fr_test_x, fr_test_y)
    except KeyboardInterrupt:
        print('-' * 100)
        print("Training interrupted, saving final model...")
        best_state_dict = torch.load('%s/%s_best.net' % (save_dir, save_name))
        torch.save(net.state_dict(), '%s/%s_final.net' % (save_dir, save_name))
        net.load_state_dict(best_state_dict)
        print('Evaluation on validation set: ')
        eval.net_evaluation(net, kf_valid, en_valid_x, en_valid_y)

        # Evaluate model on test set
        print('Evaluation on validation set: ')
        eval.net_evaluation(net, kf_en_test, en_test_x, en_test_y)
        eval.net_evaluation(net, kf_fr_test, fr_test_x, fr_test_y)


if __name__ == '__main__':
    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass

    print(args.save_name)
    print('Arguments:   ')
    print(args)
    print('-' * 100)


    train()
