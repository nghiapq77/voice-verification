import sys
import time
import os
import argparse
import glob

from model import SpeakerNet
from utils import get_data_loader, tuneThresholdfromScore

parser = argparse.ArgumentParser(description="SpeakerNet")

# Data loader
parser.add_argument('--max_frames',
                    type=int,
                    default=100,
                    help='Input length to the network for training')
parser.add_argument(
    '--eval_frames',
    type=int,
    default=100,
    help='Input length to the network for testing; 0 for whole files')
parser.add_argument('--batch_size',
                    type=int,
                    default=320,
                    help='Batch size, number of speakers per batch')
parser.add_argument('--max_seg_per_spk',
                    type=int,
                    default=100,
                    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--nDataLoaderThread',
                    type=int,
                    default=8,
                    help='Number of loader threads')
parser.add_argument('--augment',
                    action='store_true',
                    default=False,
                    help='Augment input')

# Training details
parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
parser.add_argument('--test_interval',
                    type=int,
                    default=10,
                    help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',
                    type=int,
                    default=500,
                    help='Maximum number of epochs')
parser.add_argument('--trainfunc',
                    type=str,
                    default="softmaxproto",
                    help='Loss function')

# Optimizer
parser.add_argument('--optimizer',
                    type=str,
                    default="adam",
                    help='sgd or adam')
parser.add_argument('--scheduler',
                    type=str,
                    default="steplr",
                    help='Learning rate scheduler')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument("--lr_decay",
                    type=float,
                    default=0.95,
                    help='Learning rate decay every [test_interval] epochs')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0,
                    help='Weight decay in the optimizer')

# Loss functions
parser.add_argument(
    "--hard_prob",
    type=float,
    default=0.5,
    help='Hard negative mining probability, otherwise random, only for some loss functions'
)
parser.add_argument(
    "--hard_rank",
    type=int,
    default=10,
    help='Hard negative mining rank in the batch, only for some loss functions'
)
parser.add_argument('--margin',
                    type=float,
                    default=1,
                    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',
                    type=float,
                    default=15,
                    help='Loss scale, only for some loss functions')
parser.add_argument(
    '--nPerSpeaker',
    type=int,
    default=2,
    help='Number of utterances per speaker per batch, only for metric learning based losses'
)
parser.add_argument(
    '--nClasses',
    type=int,
    default=400,
    help='Number of speakers in the softmax layer, only for softmax-based losses')

# Load and save
parser.add_argument('--initial_model',
                    type=str,
                    default="checkpoints/baseline_v2_ap.model",
                    help='Initial model weights')
parser.add_argument('--save_path',
                    type=str,
                    default="exp",
                    help='Path for model and logs')

# Training and test data
parser.add_argument('--train_list',
                    type=str,
                    default="dataset/train.def.txt",
                    help='Train list')
parser.add_argument('--test_list',
                    type=str,
                    default="dataset/val.def.txt",
                    help='Evaluation list')
parser.add_argument('--musan_path',
                    type=str,
                    default="dataset/musan_split",
                    help='Absolute path to the test set')
parser.add_argument('--rir_path',
                    type=str,
                    default="dataset/RIRS_NOISES/simulated_rirs",
                    help='Absolute path to the test set')

# Model definition
parser.add_argument('--n_mels',
                    type=int,
                    default=64,
                    help='Number of mel filterbanks')
parser.add_argument('--log_input',
                    type=bool,
                    default=True,
                    help='Log input features')
parser.add_argument('--model',
                    type=str,
                    default="ResNetSE34V2",
                    help='Name of model definition')
parser.add_argument('--encoder_type',
                    type=str,
                    default="ASP",
                    help='Type of encoder')
parser.add_argument('--nOut',
                    type=int,
                    default=512,
                    help='Embedding size in the last FC layer')

# For test only
parser.add_argument('--eval',
                    dest='eval',
                    action='store_true',
                    help='Eval only')
parser.add_argument('--test',
                    dest='test',
                    action='store_true',
                    help='Test only')

args = parser.parse_args()

# Initialise directories
model_save_path = args.save_path + "/model"
result_save_path = args.save_path + "/result"

if not (os.path.exists(model_save_path)):
    os.makedirs(model_save_path)

if not (os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

# Load models
s = SpeakerNet(**vars(args))

it = 1
prevloss = float("inf")
sumloss = 0
min_eer = [100]

# Load model weights
modelfiles = glob.glob('%s/model0*.model' % model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1])
    print("Model %s loaded from previous state!" % modelfiles[-1])
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif (args.initial_model != ""):
    s.loadParameters(args.initial_model)
    print("Model %s loaded!" % args.initial_model)

for ii in range(0, it - 1):
    s.__scheduler__.step()

# Write args to scorefile
scorefile = open(result_save_path + "/scores.txt", "a+")

for items in vars(args):
    print(items, vars(args)[items])
    scorefile.write('%s %s\n' % (items, vars(args)[items]))
scorefile.flush()

# Initialise data loader
trainLoader = get_data_loader(args.train_list, **vars(args))

while (1):
    clr = [x['lr'] for x in s.__optimizer__.param_groups]

    print(time.strftime("%Y-%m-%d %H:%M:%S"), it,
          "Training %s with LR %f..." % (args.model, max(clr)))

    # Train network
    loss, traineer = s.train_network(loader=trainLoader)

    # Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...")

        sc, lab, _ = s.evaluateFromList(args.test_list,
                                        cohorts_path=None,
                                        eval_frames=args.eval_frames)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1])

        min_eer.append(result[1])

        print(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f" %
            (max(clr), traineer, loss, result[1], min(min_eer)))
        scorefile.write(
            "IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"
            % (it, max(clr), traineer, loss, result[1], min(min_eer)))

        scorefile.flush()

        s.saveParameters(model_save_path + "/model%09d.model" % it)

        with open(model_save_path + "/model%09d.eer" % it, 'w') as eerfile:
            eerfile.write('%.4f' % result[1])

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"),
              "LR %f, TEER/TAcc %2.2f, TLOSS %f" % (max(clr), traineer, loss))
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n" %
                        (it, max(clr), traineer, loss))

        scorefile.flush()

    if it >= args.max_epoch:
        sys.exit(1)

    it += 1
    print("")

scorefile.close()
