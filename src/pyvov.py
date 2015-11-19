import array
from collections import defaultdict
import ConfigParser
import cPickle as pickle
import numpy as np
import os

__author__ = 'Matt Pugh'
__version__ = 1.0

FS_PATHS = 'FileSystemPaths'
FS_BASE_DIR = 'base_dir'

config = ConfigParser.ConfigParser()
config.read('config.ini')

EXT_INFO = 'spr'
EXT_DATA = 'sdt'
TYPES = {
    0: 'B',     # 'unsigned char',
    2: 'i',     # 'int',
    3: 'f',     # 'float',
    5: 'd'      # 'double'
}

MODE_CHIPS = 1
MODE_IMAGES = 2
MODE_TABLES = 3

BASE_PATH = config.get(FS_PATHS, FS_BASE_DIR)
paths = {
    MODE_CHIPS: BASE_PATH + 'Chips',
    MODE_IMAGES: BASE_PATH + 'Images',
    MODE_TABLES: BASE_PATH + 'Tables'
}


def get_idx():
    if os.path.isfile('idx.p'):
        return pickle.load(open('idx.p'))
    else:
        ci = ChipsIndex()
        pickle.dump(ci, open('idx.p', 'w'))
        return ci


def normalize_chip(chip, mu, sigma):
    A = ((chip - mu) * np.ones(chip.shape[0])) / sigma

    return A


class ChipsIndex(object):

    SPLIT_TRN = 'trn'
    SPLIT_TST = 'tst'
    SPLIT_BOTH = ['trn', 'tst']

    HOM4 = ['A1', 'A2', 'A3', 'A4']
    HOM38 = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6']
    HOM56 = ['C1']
    HET36 = ['D1', 'D2', 'D3', 'D4']
    HET5 = ['E1', 'E2', 'E3', 'E4', 'E5']

    ALL = ['C1', 'D4']

    def __init__(self, exp='C', do_reshape=False):
        self.exp = exp
        self.do_reshape = do_reshape
        self.vread = vread

        self.x = None
        self.y = None
        self.i = None
        self.normalized = None

        self.__populate()

        self.scoring = {}
        self._load_scoring_table()
        self.image_stats = {}
        self._load_image_stats()

    def _load_scoring_table(self):
        with open("{0}/Experiments_Scoring_Table".format(paths[MODE_TABLES])) as fp:
            data = [x for x in fp.readlines() if not x.startswith("%")]

        for line in data:
            exp, sub_exp, img_area, n_detection_opps = line.split()

            for s in range(int(sub_exp)):
                key = "{0}{1}".format(exp, s)
                self.scoring[key] = {
                    'area': img_area,
                    'n_detections': n_detection_opps
                }

    def _load_image_stats(self):
        ndim = 1024 ** 2

        for i in range(1, 135):
            A = np.reshape(vread('img' + str(i)), [ndim, 1])
            self.image_stats[i] = (np.mean(A), np.std(A))

    def __populate(self):
        matches = defaultdict(dict)

        for chip_name in get_chip_names():
            A = vread(chip_name, MODE_CHIPS)
            parts = chip_name.split("_")
            exp_id, exp_letter, exp_split = parts[1], parts[2][0], parts[2][1:]

            if exp_letter != self.exp:  # Only interested in particular experiments
                continue

            if A.shape[0] % 15 != 0:
                raise Exception("This says it's a C experiment, but rows % 15 != 0")
                continue

            windows = []

            for i in range(A.shape[1]):
                this_window = A[:,i]

                if self.do_reshape:
                    this_window = np.reshape(this_window, [15, 15])

                windows.append(this_window)

            matches[exp_id][exp_split] = windows

        self.idx = matches

    def reshape(self, source, target_dims=[15, 15]):
        if source.shape[0] % target_dims[0] != 0:
            raise Exception("Incorrect dimensions")

        return np.reshape(source, target_dims)

    def experiments(self):
        return sorted(self.idx.keys())

    def get_all(self, normalized=False):
        if self.x is None:
            c1 = self.idx['C1']
            d4 = self.idx['D4']

            x = [z for z in c1[self.SPLIT_TRN][:]]
            x.extend([z for z in d4[self.SPLIT_TRN]])
            x.extend([z for z in d4[self.SPLIT_TST]])
            x.extend([z for z in c1[self.SPLIT_TST]])

            c1 = self.labels_for('C1')
            d4 = self.labels_for('D4')

            y = [z for z in c1[self.SPLIT_TRN][:]]
            y.extend([z for z in d4[self.SPLIT_TRN]])
            y.extend([z for z in d4[self.SPLIT_TST]])
            y.extend([z for z in c1[self.SPLIT_TST]])

            c1 = self.image_numbers_for('C1')
            d4 = self.image_numbers_for('D4')

            i = [z for z in c1[self.SPLIT_TRN][:]]
            i.extend([z for z in d4[self.SPLIT_TRN]])
            i.extend([z for z in d4[self.SPLIT_TST]])
            i.extend([z for z in c1[self.SPLIT_TST]])

            self.x = x
            self.y = y
            self.i = i

        if not normalized:
            return self.x, self.y
        else:
            if self.normalized is None:
                Xn = []
                imnums = self.i

                for i, x in enumerate(self.x):
                    Xn.append(normalize_chip(x, *self.image_stats[imnums[i]]))

                self.normalized = Xn

            return self.normalized, self.y

    def training_split_for(self, exp):
        return self.idx[exp][self.SPLIT_TRN]

    def testing_split_for(self, exp):
        return self.idx[exp][self.SPLIT_TST]

    def all_for_exp(self, exp):
        """
        For this method, exp can either be an absolute fold in the cross validation,
            i.e. C1 -> {C1}, D3 -> {D3}
        Or, it may simply be the first letter, in which case it'll return all training
        and testing splits that match the mask exp*
            i.e. C* -> {C1}, D* -> {D1, D2, D3, D4} etc.
        """
        matches = [x for x in self.experiments() if x.startswith(exp)]
        retval = dict.fromkeys(matches)

        for k in matches:
            retval[k] = {
                'trn': self.training_split_for(k),
                'tst': self.testing_split_for(k)
            }

        return retval

    def image_numbers_for(self, exp):
        retval = {}

        for k in self.SPLIT_BOTH:
            exp_name ="exp_{}_N{}".format(exp, k)
            indexes = self.vread(exp_name, mode=MODE_CHIPS)[0]
            indexes = [int(x) - 1 for x in indexes]  # MATLAB -> Python
            numbers = [IMG_NUMBERS[exp][k][i] for i in indexes]
            retval[k] = numbers

        return retval

    def labels_for(self, exp):
        retval = {}

        for k in self.SPLIT_BOTH:
            exp_name ="exp_{}_L{}".format(exp, k)
            retval[k] = self.vread(exp_name, mode=MODE_CHIPS)[0]

        return retval


def get_all_pixel_windows():
    chips = get_chip_names()
    matches = {}

    for chip_name in chips:
        A = vread(chip_name, MODE_CHIPS)

        if A.shape[0] % 15 == 0:
            windows = []

            for i in range(A.shape[0]):
                this_window = A[:, i]
                this_window = np.reshape(this_window, [15, 15])
                windows.append(this_window)

            matches[chip_name] = windows

    return matches


def get_chip_names():
    with open("{0}/newlist".format(paths[MODE_CHIPS])) as fp:
        files = set([x.split('.')[0] for x in fp])

    return sorted(files)


def vread(filename, mode=MODE_IMAGES):
    with open("{0}/{1}.{2}".format(paths[mode], filename, EXT_INFO)) as idp:
        lines = [x.strip() for x in idp]
        ndim = int(lines[0])

        if ndim != 2:
            raise TypeError("Can only read two dimensional data")

        nc = int(lines[1])
        nr = int(lines[4])
        type = int(lines[7])

    try:
        precision = TYPES[type]
    except KeyError:
        raise NotImplementedError("Unrecognized data type")

    with open("{0}/{1}.{2}".format(paths[mode], filename, EXT_DATA)) as fp:
        A = array.array(precision)
        A.fromfile(fp, nc * nr)

    A = np.array(A)
    A = A.reshape((nr, nc))

    return A.transpose()


IMG_NUMBERS = {
    'A1': {
        ChipsIndex.SPLIT_TRN: [2,3,4],
        ChipsIndex.SPLIT_TST: [1]
    },
    'A2': {
        ChipsIndex.SPLIT_TRN: [1,3,4],
        ChipsIndex.SPLIT_TST: [2]
    },

    'A3': {
        ChipsIndex.SPLIT_TRN: [1,2,4],
        ChipsIndex.SPLIT_TST: [3]
    },

    'A4': {
        ChipsIndex.SPLIT_TRN: [1,2,3],
        ChipsIndex.SPLIT_TST: [4]
    }, #B
    'B1': {
        ChipsIndex.SPLIT_TRN: [5,6,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [7, 8, 9, 10, 11, 12]
    },
    'B2': {
        ChipsIndex.SPLIT_TRN: [5,6,7,8,9,10,11,12,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [13, 14, 15, 16, 17, 18]
    },
    'B3': {
        ChipsIndex.SPLIT_TRN: [5,6,7,8,9,10,11,12,13,14,15,16,17,18,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [19, 20, 21, 22, 23, 24]
    },
    'B4': {
        ChipsIndex.SPLIT_TRN: [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,31,32,33,34,35,36,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [25, 26, 27, 28, 29, 30]
    },
    'B5': {
        ChipsIndex.SPLIT_TRN: [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [31, 32, 33, 34, 35, 36]
    },
    'B6': {
        ChipsIndex.SPLIT_TRN: [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36],
        ChipsIndex.SPLIT_TST: [37, 38, 39, 40, 41, 42]
    }, #C
    'C1': {
        ChipsIndex.SPLIT_TRN: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42],
        ChipsIndex.SPLIT_TST: [79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134]
    }, #D
    'D1': {
        ChipsIndex.SPLIT_TRN: [52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78],
        ChipsIndex.SPLIT_TST: [43,44,45,46,47,48,49,50,51]
    },
    'D2': {
        ChipsIndex.SPLIT_TRN: [43,44,45,46,47,48,49,50,51,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78],
        ChipsIndex.SPLIT_TST: [52,53,54,55,56,57,58,59,60]
    },
    'D3': {
        ChipsIndex.SPLIT_TRN: [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,70,71,72,73,74,75,76,77,78],
        ChipsIndex.SPLIT_TST: [61,62,63,64,65,66,67,68,69]
    },
    'D4': {
        ChipsIndex.SPLIT_TRN: [43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69],
        ChipsIndex.SPLIT_TST: [70,71,72,73,74,75,76,77,78]
    }, #E
    'E1': {
        ChipsIndex.SPLIT_TRN: [50,68,69,75],
        ChipsIndex.SPLIT_TST: [46]
    },
    'E2': {
        ChipsIndex.SPLIT_TRN: [46,68,69,75],
        ChipsIndex.SPLIT_TST: [50]
    },
    'E3': {
        ChipsIndex.SPLIT_TRN: [46,50,69,75],
        ChipsIndex.SPLIT_TST: [68]
    },
    'E4': {
        ChipsIndex.SPLIT_TRN: [46,50,68,75],
        ChipsIndex.SPLIT_TST: [69]
    },
    'E5': {
        ChipsIndex.SPLIT_TRN: [46,50,68,69],
        ChipsIndex.SPLIT_TST: [75]
    },
}
