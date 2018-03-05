import numpy as np
from scipy import misc
import cv2

from tensorpack import *
try:
    from .mapper import Mapper
    from .cfgs.config import cfg
except Exception:
    from mapper import Mapper
    from cfgs.config import cfg

def batch_feature(feats):
    # pad to the longest in the batch
    maxlen = max([k.shape[1] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, feats[0].shape[0], maxlen, 1))
    for idx, feat in enumerate(feats):
        ret[idx, :, :feat.shape[1]] = feat
    return ret

def sparse_label(labels):
    maxlen = max([len(k) for k in labels])
    shape = [len(labels), maxlen]   # bxt
    indices = []
    values = []
    for bid, lab in enumerate(labels):
        for tid, c in enumerate(lab):
            indices.append([bid, tid])
            values.append(c)
    indices = np.asarray(indices)
    values = np.asarray(values)
    return (indices, values, shape)

def get_imglist(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == "train" else cfg.test_list
        self.train_or_test = train_or_test
        fname_list = [fname_list] if type(fname_list) is not list else fname_list

        self.imglist = []
        for fname in fname_list:
            self.imglist.extend(get_imglist(fname))

        self.shuffle = shuffle

        self.mapper = Mapper()

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img_path = self.imglist[k]
            label_path = img_path.split('.')[0] + ".txt"
            img = misc.imread(img_path, 'L')
            if img.shape[0] != cfg.input_height:
                if cfg.input_width != None:
                    img = cv2.resize(img, (cfg.input_width, cfg.input_height))
                else:
                    scale = cfg.input_height / img.shape[0]
                    img = cv2.resize(img, None, fx=scale, fy=scale)
            feat = np.expand_dims(img, axis=2)
            with open(label_path) as f:
                content = f.readlines()
            label_cleaned = ''.join([i for i in content[0] if i in cfg.dictionary])
            if label_cleaned == "":
                continue

            word_set = label_cleaned.split(' ')
            label = self.mapper.encode_string(label_cleaned)
            yield [feat, label]

class CTCBatchData(BatchData):

    def __init__(self, ds, batch_size, remainder=False):
        super(CTCBatchData, self).__init__(ds, int(batch_size), remainder)

    def get_data(self):
        itr = self.ds.get_data()
        for _ in range(self.size()):
            feats = []
            labels = []
            for b in range(self.batch_size):
                feat, label = next(itr)
                feats.append(feat)
                labels.append(label)
            batchfeat = batch_feature(feats)
            batchlabel = sparse_label(labels)
            seqlen = np.asarray([k.shape[1] for k in feats])
            yield [batchfeat, batchlabel[0], batchlabel[1], batchlabel[2], seqlen]

            
if __name__ == '__main__':
    ds = Data('train')
    ds.reset_state()
    generator = ds.get_data()
    for i in generator:
        print(len(i[1]))
        if len(i[1]) == 0:
            print(i[2])
            input()
