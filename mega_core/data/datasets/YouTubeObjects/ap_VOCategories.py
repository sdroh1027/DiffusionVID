def ap_VOCategories(cls=None):

    dir_root = '/home/user/dataset/YouTube-Objects/vo-release'

    categories = {'aeroplane', 'bird', 'boat', 'car', 'cat', 'cow', 'dog', 'horse', 'motorbike', 'train'}
    output = categories

    if cls is not None:
        output = dir_root + '/categories/' + cls + '/'

    return output
