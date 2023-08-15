from .ap_VOGetVideos import ap_VOGetVideos
import numpy as np
import scipy.io


def ap_VOGetTubes(class1, params, list1):
    # Given a list of tracks, returns a number of frames+BBs extracted from the tracks
    # Takes as input the "params" file which tells on which tracks it should
    # operate and what is the sampling strategy

    dataset = []
    shots = []
    tubes = []

    if 'maskfile' in params.videos.keys() or True:
        pp = params
        onlyGT = params.videos['GT']
    else:
        pp = {}
        onlyGT = False

    dirs, info = ap_VOGetVideos(class1, pp)  # Load all shots
    v = np.array([x['v'] for x in info])  # video number
    s = np.array([x['s'] for x in info])  # shot number
    l = np.array([x['l'] for x in info])  # shot length

    if onlyGT:
        gt = np.array([x['gt'] for x in info])
        gt_list = [list(x.items())[0] for x in gt]
        return dirs, gt_list

    else:
        #gt = textread(list1, '%d', 'headerlines', 1)
        #gt = reshape(gt, 3, len(gt) / 3)
        gt = []  # the result is n x 3 matrix
        with open(list1, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == '#':
                    continue
                ll = list(map(int, line.split(' ')))
                gt.append(ll)
        gt = np.array(gt)

        #for i = 1: size(gt, 1)   # 1 ~ len(gt)
        for i in range(len(gt)):  # 0 ~ len(gt)-1
            # gt : (video, shots, selected_tube_number)
            ids = np.nonzero((v == gt[i, 0]) * (s == gt[i, 1]))  # find(x) gets index of non-zeros
            if len(ids) != 1:
                print('Problem in finding the appropriate dir')
                raise NotImplementedError
            d = dirs[int(ids[0])]
            # load([d '/dataset.mat'])
            # load(d + '/' + params.trackname)
            T = scipy.io.loadmat(d + '/' + params.trackname)['T']  #load mat file

            track = gt[i, 2]

            # Get the frame image size for this video
            shots.append(d)
            tubes.append(T[track-1, :])

    return shots, tubes
