import scipy.io

from .ap_VOCategories import ap_VOCategories
from .ap_DIR import ap_Dir
import os.path
import re
import numpy as np


def ap_VOGetVideos(class1, params={}):
    # Return the directories and information of the videos avaialable for a certain
    # category "class". If "params" is provided, then the routine filters out those
    # directories that do not comply with the parameters specified in "params"

    bootstrap = False
    videos = []
    info = []

    # if isempty(params):
    is_train = True
    if isinstance(params, (list, dict)) and len(params) == 0:
        if is_train:
            minlen = -1
            onlyGT = 0
        else:
            minlen = -1
            onlyGT = 1
    elif hasattr(params, 'videos'):  # if class
        if 'minlen' in params.videos.keys():
            minlen = params.videos['minlen']
        if 'GT' in params.videos.keys():
            onlyGT = params.videos['GT']
        is_train = params.is_train

    datadir = ap_VOCategories(class1) + 'data/'  # Get the right directory

    # if isempty(params) or ~isfield(params.videos,'maskfile'):
    if hasattr(params, 'videos'):
        if 'maskfile' in params.videos.keys():
            maskfile = ap_VOCategories(class1) + 'sets/' + params.videos['maskfile']
        else:
            if not onlyGT: # get sudo-label
                # maskfile = ap_VOCategories(class1) + 'sets/train.txt'
                maskfile = ap_VOCategories(class1) + 'sets/all.txt'
            else:
                # maskfile = ap_VOCategories(class1) + 'sets/test.txt'
                maskfile = ap_VOCategories(class1) + 'sets/all2.txt'
    else:
        maskfile = ap_VOCategories(class1) + 'sets/all.txt'

    if len(maskfile) > 0:
        # -------------------------------------------
        # Read which shots we should use from a file
        # -------------------------------------------
        video = []
        with open(maskfile, 'r') as f:
            lines = f.readlines()
            c = 0
            # while ~feof(fid):
            for a in lines:
                # if isempty(a) or a(1) == '#' or ~isempty(strfind(a,'shit')):
                if len(a) == 0 or a[0] == '#' or 'shit' in a:
                    continue  # Skip comments
                c = c + 1
                # data = strsplit(' ',a)
                data = a.rstrip('\n').split(' ')

                video.append(datadir + '//' + data[0])
    else:
        video = ap_Dir(datadir + '/' + maskVideo, True)
        video.sort()

    c = 0
    s_ = []
    v = 0
    v_ = []
    n_ = 0  # Number of images
    videos = []

    # -------------------------------------------
    # Read the shots
    # -------------------------------------------
    for i in video:
        v = v + 1
        s = 0

        shots = ap_Dir(i + '/shots/*', True)
        shots.sort()

        # For each ad, loop over its shots
        for j in shots:
            s = s + 1  # Shot counter
            l = len(ap_Dir(j + '/*.jpg'))  # Get the num of images
            gt = ap_Dir(j + '/*_sticks.mat')  # Get the annotations

            if l < minlen or (onlyGT and len(gt) == 0):
                continue
            if hasattr(params, "trackname"):
                if not os.path.isfile(j + '/' + params.trackname + '.mat') and not bootstrap:
                    continue

            videos.append(j)
            info.append({'v': v, 's': s, 'l': l, 'gt': {}})

            if len(gt) > 0:
                for k in gt:
                    file = os.path.basename(k)
                    # frame = str2num(file(isstrprop(file,'digit')))
                    frame = int(re.sub(r'[^0-9]', '', file))
                    # load(k{1}, 'coor')
                    coor = scipy.io.loadmat(k)['coor'][0][0]
                    # coor[0] = [min(coor[0]([1 3])), min(coor[0]([2 4])), max(coor[0]([1 3])), max(coor[0]([2 4]))]
                    coor = np.array([min(coor[0], coor[2]), min(coor[1], coor[3]),
                                     max(coor[0], coor[2]), max(coor[1], coor[3])])
                    # info(c).gt{frame} = cellmat(coor{1});
                    info[-1]['gt'][str(frame)] = coor  # watch out we use only 1 annotation!

    return videos, info
