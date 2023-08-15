from ap_VOCategories import ap_VOCategories
from initVideoObjectOptions import initVideoObjectOptions
from ap_VOGetTubes import ap_VOGetTubes
from ap_VOGetVideos import ap_VOGetVideos
import scipy.io
import matplotlib.image as img
import matplotlib.pyplot as pp
import cv2
import numpy as np

# color 설정
blue_color = (255, 0, 0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
white_color = (255, 255, 255)
yellow_color = (255, 255, 0)

def demo():
    # This function demonstrate how to access the Youtube - Objects dataset v1.0 for
    # two common operations:
    # 1) Retrieve and visualize candidate tubes and the automatically selected tube
    # 2) Retrieve shots that contain an annotation and visualize it

    dataset = []

    classes_yot = ['__background__',  # always index 0
                'aeroplane',  'bird',  'boat',  'car',  'cat',
               'cow',  'dog',  'horse',  'motorbike',  'train']

    class1 = 'dog' #'aeroplane' # We demonstrate on the aeroplane class


    class_dir = ap_VOCategories(class1)
    tubes_selection_file = class_dir + '/selected_tubes_VID.list'

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # 1) Retrieve and visualize candidate tubes and the automatically selected
    # tube from a shot
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    params = initVideoObjectOptions()

    # The following call returns the tubes automatically selected by our
    # approach
    shots, tubes = ap_VOGetTubes(class1, params, tubes_selection_file)

    # Draw the automatically selected tube from one of the shots
    draw_shot_number = 63

    selected_tube = tubes[draw_shot_number - 1]

    # Load the list of images for this shot and the whole set of candidate tubes
    #load(shots[draw_shot_number - 1] + '/dataset.mat', 'dataset')
    #load(shots[draw_shot_number - 1] + '/' + params.trackname, 'T')
    dataset = scipy.io.loadmat(shots[draw_shot_number - 1] + '/dataset.mat')['dataset'][0]  # load mat file
    T = scipy.io.loadmat(shots[draw_shot_number - 1] + '/' + params.trackname)['T']  # load mat file
    # shots에는 gt 가 존재하는 데이터만 추출됨
    # T 는 tublet 정보이다. T[0] T[1] T[2] 각각 다른 튜블렛이다. 각 튜블렛에는 프레임 수 만큼 박스 정보가 존재한다. (없더라도 빈 어레이로 존재함)
    # selected_tube는 여러개의 튜브 중 정답에 해당하는 튜브이다.

    for i in range(len(dataset) - 1):

        # Show the video frame
        image = img.imread(shots[draw_shot_number - 1] + '/' + dataset[i][0][0])

        # Loop on all candidate tubes)
        for j in range(len(T)):
            if T[j, i].size == 0:
                continue

            # Overlat all candidate tubes
            x1, y1, x2, y2 = T[j, i][0]

            #line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]', 'color', 'yellow', 'linewidth', 2)
            top_left, bottom_right = (int(x1), int(y1)), (int(x2), int(y2))
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), yellow_color, 2)

        # Overlay the automatically selected tube( if present at this frame)
        #if size(selected_tube, 2) >= i and ~isempty(selected_tube{1, i}):
        if len(selected_tube) >= i:
            if selected_tube[i][0].size == 0:
                continue
            x1, y1, x2, y2 = selected_tube[i][0]

            #line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]', 'color', 'blue', 'linewidth', 2)
            top_left, bottom_right = (int(x1), int(y1)), (int(x2), int(y2))
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), blue_color, 2)
        pp.imshow(image)
        #pp.show()

    print('Press any key to continue')

    # --------------------------------------------------------------------
    # --------------------------------------------------------------------
    # 2) Retrieve shots that contain an annotation and visualize it
    # --------------------------------------------------------------------
    # --------------------------------------------------------------------

    only_gt = 1  # Selects only annotated shots
    min_frames = -1 #20  # Selects shots longer than 20 frames

    # Initialize shot selection parameters "params" based on simple criteria
    params = initVideoObjectOptions(min_frames, only_gt)

    # The following call selects a subset of shots matching the criteria
    # defined in "params"
    [shots, info] = ap_VOGetVideos(class1, params)

    # Choose the shot number we are going to draw
    draw_shot_number = 63
    #frame_number = find(1 - cellfun( @isempty, info[draw_shot_number - 1]['gt']))
    frame_number = [int(x) for x in info[draw_shot_number - 1]['gt']]
    assert len(frame_number) == 1
    frame_number = frame_number[-1]

    #load(shots[draw_shot_number - 1] +  '/dataset.mat', 'dataset')
    #imshow(shots[draw_shot_number - 1] + '/' +  dataset(frame_number).file)
    dataset = scipy.io.loadmat(shots[draw_shot_number - 1] + '/dataset.mat')['dataset'][0]
    image = img.imread(shots[draw_shot_number - 1] + '/' + dataset[frame_number - 1][0][0])

    # Overlay the annotation
    x1, y1, x2, y2 = info[draw_shot_number - 1]['gt'][str(frame_number)]

    #line([x1 x1 x2 x2 x1]',[y1 y2 y2 y1 y1]', 'color', 'green', 'linewidth', 2)
    top_left, bottom_right = (int(x1), int(y1)), (int(x2), int(y2))
    image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), blue_color, 2)
    pp.imshow(image)
    pp.show()

    return

demo()