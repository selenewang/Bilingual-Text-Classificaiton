import argparse


class DefaultsAndTypesHelpFormatter(argparse.HelpFormatter):
    def _get_help_string(self, action):
        help = action.help
        if action.type:
            action.metavar = action.type.__name__
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help


def parse_args():

    parser = argparse.ArgumentParser(formatter_class=DefaultsAndTypesHelpFormatter)

    # Everything that loaded from disk
    parser.add_argument('--source_path', type=str, default='./eddy_data/',
                        help='Path to raw data')
    parser.add_argument('--embedding_path', type=str, default='./embedding/',
                        help='Path to embedding')

    parser.add_argument('--en_train_path', type=str, default='processed_data/en_train.dat',
                        help='Path to english train document')
    parser.add_argument('--en_test_path', type=str, default='processed_data/en_test.dat',
                        help='Path to english test document')
    parser.add_argument('--fr_train_path', type=str, default='processed_data/fr_train.dat',
                        help='Path to french train document')
    parser.add_argument('--fr_test_path', type=str, default='processed_data/fr_test.dat',
                        help='Path to french test document')
    parser.add_argument('--load_net', type=str, default='', help='Path to trained model (For continue Training)')

    # Models Parameters
    parser.add_argument('--n_emb', type=int, default=300, help='Document dimension')
    parser.add_argument('--n_hidden', type=int, default=1024, help='Feature extraction layer dimension')
    parser.add_argument('--n_mlp_layers', type=int, default=1, help='Number of MLP layers')
    parser.add_argument('--batch_size', type=int, default=124, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=150, help='number of training epochs')

    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Optimizer type: must choose SGD or Adam. Default Adam')
    parser.add_argument('--lr', type=float, help='Learning rate, default 0.0001', default=0.0001)
    parser.add_argument('--lr_anneal', type=float, help='Parameter for learning rate annealing', default=2)
    parser.add_argument('--save_dir', type=str, default='saved_nets', help='Model saving directory')
    parser.add_argument('--save_name', type=str, default='model', help='Model saving prefix name')

    # Hardware
    parser.add_argument('--device', type=str, default='gpu', help='Device used for training (cpu or gpu)')
    # parser.add_argument('--ngpu', type=int, default=1, help='Number of GPU used if you choose --device gpu')


    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    args.save_name = '%s_%d_%d' % (
        args.save_name,
        int(args.n_mlp_layers),
        int(args.n_hidden))



    return args