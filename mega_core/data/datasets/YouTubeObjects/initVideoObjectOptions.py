def initVideoObjectOptions(minlen=15, GT=False):
    # This is the "params" variable that goes to input in
    # "ap_VOTrackDescriptor"
    #  GT: When true, uses only videos with GT

    motion2boxes = 'percentile3'

    class Params:
        def __init__(self):
            # This is the method used to map motion into boxes
            self.motion2boxes = {}
            if motion2boxes == 'std':
                self.motion2boxes['method'] = 'std'
            if motion2boxes == 'percentile':
                self.motion2boxes['method'] = 'percentile'
                self.motion2boxes['percentile'] = 0.90
            if motion2boxes == 'percentile2':
                self.motion2boxes['method'] = 'percentile2'
                self.motion2boxes['percentile'] = 0.90
            if motion2boxes == 'percentile3':
                self.motion2boxes['method'] = 'percentile3'
                self.motion2boxes['percentile'] = 0.80
                self.motion2boxes['nstep'] = 4
            if motion2boxes == 'broxvoc':
                self.motion2boxes['method'] = 'broxvoc'
                self.motion2boxes['motion_thresh'] = -999.
                self.motion2boxes['min_len'] = 5

            self.trackname = 'T' + self.motion2boxes['method']

            # Criteria for which videos should be used
            self.videos = {}
            self.videos['minlen'] = minlen
            self.videos['GT'] = GT

    return Params()
