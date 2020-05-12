import argparse
import constants


def get_args():
    """Parse script arguments.

    :return: object with all argument values
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, required=True, help=constants.LOSS_OPTIONS_STR)
    parser.add_argument('--distance', type=str, default='cosine',
                        help='Loss distance (if applicable): euclidean / cosine')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Steps (in percentage) to show epoch progress. Default value: 10')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training and testing. Default value: 32')
    parser.add_argument('--save', dest='save', action='store_true', help='Save best model on dev')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do NOT save best model on dev')
    parser.set_defaults(save=True)
    parser.add_argument('--recover', type=str, default=None,
                        help='The path to the saved model to recover for training')
    parser.add_argument('-m', '--margin', type=float, default=0.2,
                        help='Loss margin (if applicable). Default value: 0.2')
    parser.add_argument('-s', '--loss-scale', type=float, default=10,
                        help='Loss scaling factor or lambda for Center loss (if applicable). Default value: 10')
    parser.add_argument('--exp-path', type=str, required=True, help='Experiment directory for logs')
    parser.add_argument('--lr', type=float, default=0.001, help='Encoder learning rate. Default value: 0.001')
    parser.add_argument('--seed', type=int, default=constants.SEED, help='Random seed')
    parser.add_argument('--triplet-strategy', type=str, default='all',
                        help=F'Triplet sampling strategy: {constants.TRIPLET_SAMPLING_OPTIONS_STR}. Default value: `all`')
    parser.add_argument('--semihard-negatives', type=int, default=10,
                        help='Negatives to keep in semi-hard negative triplet sampling strategy (if applicable). Default value: 10')
    parser.add_argument('--recover-optim', dest='recover_optim', action='store_true',
                        help='Recover optimizer state from model checkpoint')
    parser.add_argument('--no-recover-optim', dest='recover_optim', action='store_false',
                        help='Do NOT recover optimizer state from model checkpoint')
    parser.set_defaults(recover_optim=True)
    parser.add_argument('--size-average', dest='size_average', action='store_true', help='Average batch loss')
    parser.add_argument('--no-size-average', dest='size_average', action='store_false', help='Sum batch loss')
    parser.set_defaults(size_average=True)
    parser.add_argument('--path', type=str, required=True, help='Path to AMI dataset')
    parser.add_argument('--model', type=str, required=True, help='lstm / bert')
    parser.add_argument('--vocab', type=str, required=False, help='Path to vocabulary file (LSTM only)')
    parser.add_argument('--word2vec-model', type=str, required=False, default=None,
                        help='Path to GENSIM Word2Vec model (LSTM only)')
    return parser.parse_args()