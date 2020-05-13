from gensim.models import KeyedVectors
from transformers import BertTokenizer
from models import MetricNet, AMILSTM, AMIBert
from datasets.ami import AMI
import losses.config as cf
import losses.triplet as tri
from distances import Distance
import constants


def build_model(args, nfeat: int, clf):
    """Build a `MetricNet` model with the appropriate encoder
    and classification layer according to the loss.
    Also returns a loader of the AMI dataset with the appropriate
    format for the model.

    :param args: script arguments object
    :param nfeat: number of features
    :param clf: `nn.Module` a classifier module
    :return:
        - model: `MetricNet` with appropriate encoder and classifier
        - loader: `SimDataset` formatted to the encoder
    """
    # Instantiate AMI dataset
    dataset = AMI(args.path)
    if args.model == 'lstm':
        # Create vocabulary
        vocab_vec = KeyedVectors.load(args.word2vec_model, mmap='r')
        vocab = [line.strip() for line in open(args.vocab, 'r')]
        # Build LSTM encoder and its loader
        encoder = AMILSTM(nfeat_word=300, nfeat_sent=nfeat,
                          word_list=vocab, vec_vocab=vocab_vec, dropout=0.2)
        loader = dataset.word2vec_loader(args.batch_size)
    elif args.model == 'bert':
        # Load pretrained tokenizer
        pretrained_weights = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
        # Build Bert encoder and its loader
        encoder = AMIBert(pretrained_weights, freeze=False)
        loader = dataset.bert_loader(tokenizer, args.batch_size)
    else:
        raise ValueError(f"Unknown model '{args.model}'. Only 'lstm' and 'bert' are accepted")

    # Build joint model
    return MetricNet(encoder=encoder, classifier=clf), loader


def build_config(args, nfeat: int, nclass: int) -> cf.LossConfig:
    """Create a configuration object for a given loss with
    a given configuration of hyper-parameters.

    :param args: script argument object
    :param nfeat: number of features
    :param nclass: number of classes
    :return: a `LossConfig` for the appropriate loss and hyper-parameters
    """
    distance = Distance.from_name(args.distance)
    if args.loss == 'softmax':
        return cf.SoftmaxConfig(nfeat, nclass)
    elif args.loss == 'contrastive':
        print(f"[Margin: {args.margin}]")
        return cf.ContrastiveConfig(margin=args.margin,
                                    distance=distance,
                                    size_average=args.size_average,
                                    online=True)
    elif args.loss == 'triplet':
        sampling = build_triplet_strategy(args.triplet_strategy, args.semihard_negatives)
        return cf.TripletConfig(scaling=args.loss_scale,
                                distance=distance,
                                size_average=args.size_average,
                                online=True,
                                sampling=sampling)
    elif args.loss == 'arcface':
        print(f"[Margin: {args.margin}]")
        return cf.ArcFaceConfig(nfeat, nclass, margin=args.margin, s=args.loss_scale)
    elif args.loss == 'center':
        return cf.CenterConfig(nfeat, nclass, lweight=args.loss_scale, distance=distance)
    elif args.loss == 'coco':
        return cf.CocoConfig(nfeat, nclass, alpha=args.loss_scale)
    else:
        raise ValueError(f"Loss function should be one of: {constants.LOSS_OPTIONS_STR}")


def build_triplet_strategy(strategy: str, semihard_n: int) -> tri.TripletSamplingStrategy:
    """
    Create a triplet sampling strategy object based on a script argument
    :param strategy: the name of the strategy received as a script argument
    :param semihard_n: the number of negatives to keep when using a semi-hard negative strategy
    :return: a `TripletSamplingStrategy` object
    """
    if strategy == 'all':
        return tri.BatchAll()
    elif strategy == 'semihard-neg':
        return tri.SemiHardNegative(semihard_n)
    elif strategy == 'hardest-neg':
        return tri.HardestNegative()
    elif strategy == 'hardest-pos-neg':
        return tri.HardestPositiveNegative()
    else:
        raise ValueError(f"Triplet strategy should be one of: {constants.TRIPLET_SAMPLING_OPTIONS_STR}")


def build_essential_plots(lr: float, batch_size: int, eval_metric: str, eval_metric_color: str) -> list:
    """Create a list of plot definitions for the trainer to build from dumped logs.
    The plots included here correspond to the training loss and the performance on dev.

    :param lr: learning rate value
    :param batch_size: number of sentences in a batch
    :param eval_metric: the evaluation metric name
    :param eval_metric_color: the matplotlib color with which to draw the plot
    :return: a list of dictionaries describing each plot for a `Trainer`
    """
    return [
        {
            'log_file': 'loss.log',
            'metric': 'Loss',
            'bottom': None, 'top': None,
            'color': 'blue',
            'title': f'Train Loss - lr={lr} - batch_size={batch_size}',
            'filename': 'loss-plot'
        },
        {
            'log_file': 'metric.log',
            'metric': eval_metric,
            'bottom': 0.,
            'top': 1.,
            'color': eval_metric_color,
            'title': f'Dev {eval_metric} - lr={lr} - batch_size={batch_size}',
            'filename': f"dev-{eval_metric.lower().replace(' ', '-')}-plot"
        }
    ]