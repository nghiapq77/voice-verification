import argparse
import numpy as np
from pathlib import Path
import sys
import random
from tqdm import tqdm
from math import fabs
import torch
import torch.nn.functional as F

from model import SpeakerNet
from utils import read_config, tuneThresholdfromScore

parser = argparse.ArgumentParser(description="SpeakerNet")

# YAML config
parser.add_argument('--config', type=str, default=None)

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
                    default="checkpoints/final_500.model",
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
parser.add_argument('--test_path',
                    type=str,
                    default="dataset/",
                    help='Absolute path to the test set')
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
parser.add_argument('--prepare',
                    dest='prepare',
                    action='store_true',
                    help='Prepare embeddings')
parser.add_argument('-t',
                    '--prepare_type',
                    type=str,
                    default='cohorts',
                    help='embed / cohorts')
parser.add_argument('--predict',
                    dest='predict',
                    action='store_true',
                    help='Predict')
parser.add_argument('--cohorts_path',
                    type=str,
                    default="checkpoints/cohorts_final_500_f100.npy",
                    help='Cohorts path')
parser.add_argument('--test_threshold',
                    type=float,
                    default=1.7206447124481201,
                    help='Test threshold')

args = parser.parse_args()
if args.config is not None:
    args = read_config(args.config, args)

# Load models
model = SpeakerNet(**vars(args))
model.loadParameters(args.initial_model)
model.eval()
cohorts = np.load('checkpoints/cohorts_final_500_f100.npy')
top_cohorts = 200
threshold = 1.7206447124481201
eval_frames = 100
num_eval = 10

if __name__ == '__main__':
    # Evaluation code
    if args.eval is True:
        sc, lab, trials = model.evaluateFromList(
            args.test_list,
            cohorts_path=args.cohorts_path,
            print_interval=100,
            eval_frames=args.eval_frames)
        target_fa = np.linspace(10, 0, num=50)
        result = tuneThresholdfromScore(sc, lab, target_fa)
        print('tfa [thre, fpr, fnr]')
        best_sum_rate = 999
        best_tfa = None
        for i, tfa in enumerate(target_fa):
            print(tfa, result[0][i])
            sum_rate = result[0][i][1] + result[0][i][2]
            if sum_rate < best_sum_rate:
                best_sum_rate = sum_rate
                best_tfa = result[0][i]
        print(f'Best sum rate {best_sum_rate} at {best_tfa}')
        print(f'EER {result[1]} at threshold {result[2]}')
        print(f'AUC {result[3]}')
        sys.exit(1)

    # Test code
    if args.test is True:
        model.testFromList(args.test_path,
                           cohorts_path=args.cohorts_path,
                           thre_score=args.test_threshold,
                           print_interval=100,
                           eval_frames=args.eval_frames)
        sys.exit(1)

    # Prepare embeddings for cohorts/verification
    if args.prepare is True:
        model.prepare(eval_frames=args.eval_frames,
                      from_path=args.test_list,
                      save_path=args.save_path,
                      num_eval=num_eval,
                      prepare_type=args.prepare_type)
        sys.exit(1)

    # Predict
    if args.predict is True:
        """
        Predict new utterance based on distance between its embedding and saved embeddings.
        """
        embeds_path = Path(args.save_path, 'embeds.pt')
        classes_path = Path(args.save_path, 'classes.npy')
        embeds = torch.load(embeds_path).to(torch.device(args.device))
        classes = np.load(classes_path, allow_pickle=True).item()
        if args.test_list.endswith('.txt'):
            files = []
            with open(args.test_list) as listfile:
                while True:
                    line = listfile.readline()
                    if (not line):
                        break
                    data = line.split()

                    # Append random label if missing
                    if len(data) == 2:
                        data = [random.randint(0, 1)] + data

                    files.append(Path(data[1]))
                    files.append(Path(data[2]))

            files = list(set(files))
        else:
            files = list(Path(args.test_list).glob('*/*.wav'))
        files.sort()

        same_smallest_score = 1
        diff_biggest_score = 0
        for f in tqdm(files):
            embed = model.embed_utterance(f,
                                          eval_frames=args.eval_frames,
                                          num_eval=num_eval,
                                          normalize=model.__L__.test_normalize)
            embed = embed.unsqueeze(-1)
            dist = F.pairwise_distance(embed, embeds).detach().cpu().numpy()
            dist = np.mean(dist, axis=0)
            score = 1 - np.min(dist)**2 / 2
            if classes[np.argmin(dist)] == f.parent.stem:
                if score < same_smallest_score:
                    same_smallest_score = score
                indexes = np.argsort(dist)[:2]
                if fabs((1 - dist[indexes[0]]**2 / 2) - (1 - dist[indexes[1]]**2 / 2)) < 0.001:
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx]**2 / 2
                        if i == 0:
                            tqdm.write(f'+ {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
                else:
                    tqdm.write(f'+ {f}, {score}', end='')
                    if score < args.test_threshold:
                        tqdm.write(' ***', end='')
                    tqdm.write('')
            else:
                if score > diff_biggest_score:
                    diff_biggest_score = score
                if score > args.test_threshold:
                    indexes = np.argsort(dist)[:3]
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx]**2 / 2
                        if i == 0:
                            tqdm.write(f'- {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
        print(f'same_smallest_score: {same_smallest_score}')
        print(f'diff_biggest_score: {diff_biggest_score}')
        sys.exit(1)
