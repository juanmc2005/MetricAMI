from os.path import join
import utils
import builders
import args
from core.base import Trainer
from core.plugins.logging import TrainLogger, MetricFileLogger, HeaderPrinter
from core.plugins.storage import BestModelSaver, ModelLoader
from metrics import KNNF1ScoreMetric, Evaluator

# Script arguments
args = args.get_args()

# Create directory to save plots, models, results, etc
log_path = utils.create_log_dir(args.exp_path, args)
print(f"Logging to {log_path}")

# Dump all script arguments
utils.dump_params(join(log_path, 'config.cfg'), args)

# Set custom seed
utils.set_custom_seed(args.seed)

# Load dataset and create model
print(f"[Model: {args.model.upper()}]")
print(f"[Loss: {args.loss.upper()}]")
print('[Loading Dataset and Model...]')

# Embedding dim is 768 based on BERT, we use the same for LSTM to be fair
# Classes are 6 because we include a 'non-misogyny class'
nfeat, nclass = 768, 6
config = builders.build_config(args, nfeat, nclass)
model, loader = builders.build_model(args, nfeat, config.clf)
dev = loader.dev_partition()
test = loader.test_partition()
train = loader.training_partition()
print('[Dataset and Model Loaded]')

print(f"[LR: {args.lr}]")
print(f"[Batch Size: {args.batch_size}]")
print(f"[Epochs: {args.epochs}]")
print(f"[Model Size: {utils.model_size(model):.1f}m]")

# Train and evaluation plugins
test_callbacks = []
train_callbacks: list = [HeaderPrinter()]

# Logging configuration
if args.log_interval in range(1, 101):
    print(f"[Logging: every {args.log_interval}%]")
    # Save per epoch dev performance on metric.log
    test_callbacks.append(MetricFileLogger(log_path=join(log_path, f"metric.log")))
    # Save per epoch loss on loss.log
    train_callbacks.append(TrainLogger(args.log_interval, train.nbatches(),
                                       loss_log_path=join(log_path, f"loss.log")))
else:
    print(f"[Logging: None]")

# Model saving configuration
print(f"[Model Saving: {args.save}]")
if args.save:
    test_callbacks.append(BestModelSaver(log_path, args.loss))

# Evaluation configuration
metric = KNNF1ScoreMetric(config.test_distance, neighbors=10)
dev_evaluator = Evaluator(dev, metric, 'dev', test_callbacks)
test_evaluator = Evaluator(test, metric, 'test',
                           callbacks=[MetricFileLogger(log_path=join(log_path, 'test-metric.log'))])
train_callbacks.extend([dev_evaluator, test_evaluator])

# Training configuration
optim = config.optimizer(model, lr=args.lr)
model_loader = ModelLoader(args.recover, args.recover_optim) if args.recover is not None else None
trainer = Trainer(args.loss, model, config.loss, train, optim,
                  model_loader=model_loader,
                  callbacks=train_callbacks,
                  last_metric_fn=lambda: dev_evaluator.last_metric)

print()

# Start training
plots = builders.build_essential_plots(args.lr, args.batch_size, 'Macro F1', 'green')
trainer.train(args.epochs, log_path, plots)

# Dump and print results
best_epoch = dev_evaluator.best_epoch
best_dev = dev_evaluator.best_metric
best_test = test_evaluator.results[dev_evaluator.best_epoch-1]

with open(join(log_path, 'results.txt'), 'w') as out:
    out.write(f"epoch: {best_epoch}\n")
    out.write(f"dev: {best_dev}\n")
    out.write(f"test: {best_test}\n")

print(f"Best result at epoch {best_epoch}:")
print(f"Dev: {best_dev}")
print(f"Test: {best_test}")
