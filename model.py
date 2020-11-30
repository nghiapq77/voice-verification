import sys
import random
import time
import importlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils import loadWAV, score_normalization


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, scheduler, trainfunc, **kwargs):
        super(SpeakerNet, self).__init__()

        SpeakerNetModel = importlib.import_module(
            'models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).cuda()

        LossFunction = importlib.import_module(
            'loss.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).cuda()

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.parameters(), **kwargs)

        Scheduler = importlib.import_module(
            'scheduler.' + scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__,
                                                     **kwargs)

        assert self.lr_step in ['epoch', 'iteration']

    def train_network(self, loader):

        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        tstart = time.time()

        for data, data_label in loader:
            data = data.transpose(0, 1)
            self.zero_grad()
            feat = []
            for inp in data:
                outp = self.__S__.forward(inp.cuda())
                feat.append(outp)

            feat = torch.stack(feat, dim=1).squeeze()
            label = torch.LongTensor(data_label).cuda()
            nloss, prec1 = self.__L__.forward(feat, label)

            loss += nloss.detach().cpu()
            top1 += prec1
            counter += 1
            index += stepsize

            nloss.backward()
            self.__optimizer__.step()

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) " % (index))
            sys.stdout.write(
                "Loss %f TEER/TAcc %2.3f%% - %.2f Hz " %
                (loss / counter, top1 / counter, stepsize / telapsed))
            sys.stdout.flush()

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        sys.stdout.write("\n")

        return (loss / counter, top1 / counter)

    def evaluateFromList(self,
                         listfilename,
                         cohorts_path='dataset/cohorts.npy',
                         print_interval=100,
                         test_path='',
                         num_eval=10,
                         eval_frames=None,
                         is_train=False):

        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = np.load(cohorts_path)

        # Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if (not line):
                    break
                data = line.split()

                # Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to file
        for idx, file in enumerate(setfiles):
            inp1 = torch.FloatTensor(
                loadWAV(file, eval_frames, evalmode=True,
                        num_eval=num_eval)).cuda()

            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()
            feats[file] = ref_feat
            telapsed = time.time() - tstart
            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading %d of %d: %.2f Hz, %.4f s, embedding size %d" %
                    (idx, len(setfiles), idx / telapsed, telapsed /
                     (idx + 1), ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        # Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # NOTE: distance for training, normalized score for evaluating and testing
            if is_train:
                dist = F.pairwise_distance(
                    ref_feat.unsqueeze(-1),
                    com_feat.unsqueeze(-1).transpose(
                        0, 2)).detach().cpu().numpy()
                score = -1 * np.mean(dist)
            else:
                score = score_normalization(ref_feat,
                                            com_feat,
                                            cohorts,
                                            top=200)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz - %.4f s" %
                                 (idx, len(lines),
                                  (idx + 1) / telapsed, telapsed / (idx + 1)))
                sys.stdout.flush()

        print('\n')

        return (all_scores, all_labels, all_trials)

    def testFromList(self,
                     root,
                     thre_score=0.5,
                     cohorts_path='data/zalo/cohorts.npy',
                     print_interval=100,
                     test_path='',
                     num_eval=10,
                     eval_frames=None):
        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = np.load(cohorts_path)

        # Read all lines
        data_root = Path(root, 'public-test')
        read_file = Path(root, 'public-test.csv')
        write_file = Path(root, 'submission.csv')
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for i, row in enumerate(tqdm(spamreader)):
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to file
        for idx, file in enumerate(setfiles):
            inp1 = torch.FloatTensor(
                # loadWAV(os.path.join(data_root, file),
                loadWAV(Path(data_root, file),
                        eval_frames,
                        evalmode=True,
                        num_eval=num_eval)).cuda()
            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()
            feats[file] = ref_feat
            telapsed = time.time() - tstart
            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading %d of %d: %.2f Hz, %.4f s, embedding size %d" %
                    (idx, len(setfiles), (idx + 1) / telapsed, telapsed /
                     (idx + 1), ref_feat.size()[1]))

        print('')
        tstart = time.time()

        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'label'])
            for idx, data in enumerate(lines):
                ref_feat = feats[data[0]].cuda()
                com_feat = feats[data[1]].cuda()

                if self.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                score = score_normalization(ref_feat,
                                            com_feat,
                                            cohorts,
                                            top=200)
                pred = '0'
                if score >= thre_score:
                    pred = '1'
                spamwriter.writerow([data[0], data[1], pred])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz, %.4f s" %
                                     (idx, len(lines),
                                      (idx + 1) / telapsed, telapsed /
                                      (idx + 1)))
                    sys.stdout.flush()

        print('\n')

    def prepare(self,
                from_path='../data/test',
                save_path='checkpoints',
                prepare_cohorts=False,
                num_eval=10,
                eval_frames=0,
                print_interval=1):
        """
        Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        """
        tstart = time.time()
        self.eval()
        if prepare_cohorts:
            # Prepare cohorts for score normalization.
            feats = []
            read_file = Path(from_path)
            files = []
            used_speakers = []
            prepare_cohorts = True
            with open(read_file) as listfile:
                while True:
                    line = listfile.readline()
                    if (not line):
                        break
                    data = line.split()

                    data_1_class = Path(data[1]).parent.stem
                    data_2_class = Path(data[2]).parent.stem
                    if data_1_class not in used_speakers:
                        used_speakers.append(data_1_class)
                        files.append(data[1])
                    if data_2_class not in used_speakers:
                        used_speakers.append(data_2_class)
                        files.append(data[2])
            setfiles = list(set(files))
            setfiles.sort()

            # Save all features to file
            for idx, f in tqdm(enumerate(setfiles)):
                inp1 = torch.FloatTensor(
                    loadWAV(f, eval_frames, evalmode=True,
                            num_eval=num_eval)).cuda()

                feat = self.__S__.forward(inp1)
                if self.__L__.test_normalize:
                    feat = F.normalize(feat, p=2,
                                       dim=1).detach().cpu().numpy().squeeze()
                else:
                    feat = feat.detach().cpu().numpy().squeeze()
                feats.append(feat)

            np.save(save_path, np.array(feats))
        else:
            # Prepare mean L2-normalized embeddings for known speakers.
            speaker_dirs = [x for x in Path(from_path).iterdir() if x.is_dir()]
            feats = None
            classes = {}
            # Save mean features
            for idx, speaker_dir in enumerate(speaker_dirs):
                classes[idx] = speaker_dir.stem
                files = list(speaker_dir.glob('*.wav'))
                feat = None
                for f in files:
                    inp = torch.FloatTensor(
                        loadWAV(f,
                                eval_frames,
                                evalmode=True,
                                num_eval=num_eval)).cuda()
                    with torch.no_grad():
                        ref_feat = self.__S__.forward(inp)
                        if feat is None:
                            feat = F.normalize(ref_feat, p=2,
                                               dim=1).unsqueeze(0)
                        else:
                            feat = torch.cat(
                                (feat, F.normalize(ref_feat, p=2,
                                                   dim=1).unsqueeze(0)), 0)
                feat = torch.mean(feat, dim=0)
                if feats is None:
                    feats = feat.unsqueeze(0)
                else:
                    feats = torch.cat((feats, feat.unsqueeze(0)), 0)
                telapsed = time.time() - tstart
                if idx % print_interval == 0:
                    sys.stdout.write(
                        "\rReading %d of %d: %.4f s, embedding size %d" %
                        (idx, len(speaker_dirs), telapsed /
                         (idx + 1), ref_feat.size()[1]))
            print('')
            feats = rearrange(feats,
                              'n_class n_sam feat -> n_sam feat n_class')
            print(feats.shape)
            torch.save(feats, Path(save_path, 'embeds.pt'))
            np.save(Path(save_path, 'classes.npy'), classes)

    def predict(self,
                from_path='../data/test_1',
                embeds_path='checkpoints/embeds.pt',
                classes_path='checkpoints/classes.npy',
                num_eval=10,
                eval_frames=0,
                print_interval=1):
        """
        Predict new utterance based on distance between its embedding and saved embeddings.
        """
        self.eval()
        feats = torch.load(embeds_path)
        classes = np.load(classes_path, allow_pickle=True).item()
        files = list(Path(from_path).glob('*/*.wav'))
        for f in tqdm(files):
            inp = torch.FloatTensor(
                loadWAV(f, eval_frames, evalmode=True,
                        num_eval=num_eval)).cuda()
            with torch.no_grad():
                feat = self.__S__.forward(inp)
                feat = F.normalize(feat, p=2, dim=1)

            # NOTE: If the embeddings are L2 normalized, the squared Euclidean distance is directly proportional to the cosine distance.
            feat = feat.unsqueeze(-1)
            dist = F.pairwise_distance(feat, feats).detach().cpu().numpy()
            dist = np.mean(dist, axis=0)
            tqdm.write(
                f'{f}, {classes[np.argmin(dist)]}, {1 - np.min(dist)**2 / 2}')

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" %
                      (origname, self_state[name].size(),
                       loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
