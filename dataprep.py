import argparse
import os
import subprocess
import hashlib
import glob
import tarfile
from zipfile import ZipFile
from scipy.io import wavfile
from pathlib import Path
import random
from tqdm import tqdm

# Parse input arguments
parser = argparse.ArgumentParser(description="VoxCeleb downloader")

parser.add_argument('--save_path',
                    type=str,
                    default="dataset/",
                    help='Target directory')
parser.add_argument('--split_ratio',
                    type=float,
                    default=0.1,
                    help='Split ratio')

parser.add_argument('--convert',
                    dest='convert',
                    action='store_true',
                    help='Enable coversion')
parser.add_argument('--generate',
                    dest='generate',
                    action='store_true',
                    help='Enable generate')
parser.add_argument('--augment',
                    dest='augment',
                    action='store_true',
                    help='Download and extract augmentation files')

args = parser.parse_args()


def md5(fname):
    """
    MD5SUM
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(args, lines):
    """
    Download with wget
    """
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split('/')[-1]

        # Download files
        out = subprocess.call('wget %s -O %s/%s' %
                              (url, args.save_path, outfile),
                              shell=True)
        if out != 0:
            raise ValueError(
                'Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'
                % url)

        # Check MD5
        md5ck = md5('%s/%s' % (args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.' % outfile)
        else:
            raise Warning('Checksum failed %s.' % outfile)


def full_extract(args, fname):
    """
    Extract zip files
    """
    print('Extracting %s' % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)


def part_extract(args, fname, target):
    """
    Partially extract zip files
    """
    print('Extracting %s' % fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)


def split_musan(args):
    """
    Split MUSAN for faster random access
    """

    files = glob.glob('%s/musan/*/*/*.wav' % args.save_path)

    audlen = 16000 * 5
    audstr = 16000 * 3

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/',
                                                 '/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + '/%05d.wav' % (st / fs), fs,
                          aud[st:st + audlen])

        print(idx, file)


def convert(args):
    files = list(Path(args.save_path).glob('*/*.wav'))
    files.sort()
    print('Converting files')
    for fpath in tqdm(files):
        fpath = str(fpath).replace('(', '\(')
        fpath = fpath.replace(')', '\)')
        outpath = fpath[:-4] + '_conv' + fpath[-4:]
        out = subprocess.call(
            'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null'
            % (fpath, outpath),
            shell=True)
        if out != 0:
            raise ValueError('Conversion failed %s.' % fpath)
        subprocess.call('rm %s' % (fpath), shell=True)
        subprocess.call('mv %s %s' % (outpath, fpath), shell=True)


def generate_lists(args):
    """
    Generate train test lists for zalo data
    """
    root = Path(args.save_path)
    train_writer = open(Path(root.parent, 'train.txt'), 'w')
    val_writer = open(Path(root.parent, 'val.txt'), 'w')
    classpaths = [d for d in root.iterdir() if d.is_dir()]
    val_filepaths_list = []
    for classpath in classpaths:
        filepaths = list(classpath.glob('*.wav'))
        val_num = 3  # 3 utterances per speaker for val
        if args.split_ratio > 0:
            val_num = int(args.split_ratio * len(filepaths))
        random.shuffle(filepaths)
        val_filepaths = filepaths[:val_num]
        train_filepaths = filepaths[val_num:]
        for train_filepath in train_filepaths:
            label = str(train_filepath.parent.stem.split('-')[0])
            train_writer.write(label + ' ' + str(train_filepath) + '\n')
        val_filepaths_list.append(val_filepaths)
    for val_filepaths in val_filepaths_list:
        for i in range(len(val_filepaths) - 1):
            for j in range(i + 1, len(val_filepaths)):
                label = '1'
                val_writer.write(label + ' ' + str(val_filepaths[i]) + ' ' +
                                 str(val_filepaths[j]) + '\n')
                label = '0'
                while True:
                    x = random.randint(0, len(val_filepaths_list) - 1)
                    if not val_filepaths_list[x]:
                        continue
                    if val_filepaths_list[x][0].parent.stem != val_filepaths[
                            i].parent.stem:
                        break
                y = random.randint(0, len(val_filepaths_list[x]) - 1)
                val_writer.write(label + ' ' + str(val_filepaths[i]) + ' ' +
                                 str(val_filepaths_list[x][y]) + '\n')

    train_writer.close()
    val_writer.close()


if __name__ == "__main__":
    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    f = open('dataset/augment.txt', 'r')
    augfiles = f.readlines()
    f.close()

    if args.augment:
        download(args, augfiles)
        part_extract(args, os.path.join(args.save_path, 'rirs_noises.zip'), [
            'RIRS_NOISES/simulated_rirs/mediumroom',
            'RIRS_NOISES/simulated_rirs/smallroom'
        ])
        full_extract(args, os.path.join(args.save_path, 'musan.tar.gz'))
        split_musan(args)

    if args.generate:
        generate_lists(args)
    if args.convert:
        convert(args)
