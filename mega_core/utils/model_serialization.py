# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict
import logging

import torch
import numpy as np

from mega_core.utils.imports import import_file
from mega_core.modeling.detector.diffusion_det import DiffusionDet


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, flownet=False):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained weights from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    # get a matrix of string matches, where each (i, j) entry correspond to the size of the
    # loaded_key string, if it matches
    if flownet is None:
        match_matrix = [
            len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
        ]
    elif not flownet:
        match_matrix = [
            len(j) if i.endswith(j) and ("flownet" not in i) and ("embednet" not in i) else 0 for i in current_keys for j in loaded_keys
        ]
    else:
        match_matrix = [
            len(j) if i.endswith(j) and "flownet" in i else 0 for i in current_keys for j in loaded_keys
        ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger(__name__)
    keys_not_updated = []
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            keys_not_updated.append(current_keys[idx_new])
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )
    if keys_not_updated:
        print("{} keys are not updated: {}".format(len(keys_not_updated), keys_not_updated))


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            key = key[len(prefix):]
        stripped_state_dict[key] = value
    return stripped_state_dict

import re
def remove_modules(model_keys, state_dict, skip_names):
    keys = sorted(state_dict.keys())
    stripped_state_dict = OrderedDict()

    # DiffusionDet -> DiffusionVID module name change
    names_A_to_B = ['head_series', 'head_series_cond']
    print('Name change: {} -> {}'.format(names_A_to_B[0], names_A_to_B[1]))

    num_modules = []
    for kwd in names_A_to_B:
        filtered = [key for key in model_keys if kwd + '.' in key]
        filtered_num = [int(key.split(kwd + '.')[1][0]) for key in filtered]
        num_modules.append(max(filtered_num) + 1)

    from itertools import accumulate
    range_idx = list(accumulate(num_modules))
    change_names = ['head_series.' + str(i) for i in range(range_idx[0], range_idx[1])]

    if skip_names is not None:
        # assume this is training mode
        skip_names2 = [str(i) + '.block_time_mlp.1.weight' for i in range(range_idx[0], range_idx[1])] \
                    + [str(i) + '.block_time_mlp.1.bias' for i in range(range_idx[0], range_idx[1])]
        skip_names = skip_names + skip_names2
    else:
        skip_names = []

    for key, value in state_dict.items():
        if any(name in key for name in skip_names):
            continue
        elif any(name in key for name in change_names):
            pattern = re.compile(r'\d+')
            match = pattern.search(key)
            if match:
                # If a match is found, extract the matched string and convert it to a number
                matched_string = match.group()
                number = int(matched_string)
                new_number = number - num_modules[0]

                # Replace the character at the specified position
                key_list = list(key)
                key_list[match.span()[0]] = str(new_number)
                new_string = ''.join(key_list)
                new_string = new_string.replace(names_A_to_B[0], names_A_to_B[1])
                stripped_state_dict[new_string] = value
        elif any(name in key for name in ['head_series_local']):
            # Replace the character at the specified position
            new_string = key.replace('head_series_local', 'head_series_cond')
            stripped_state_dict[new_string] = value
        else:
            stripped_state_dict[key] = value
    return stripped_state_dict

def load_state_dict(model, loaded_state_dict, flownet=False, skip_modules=None):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    if isinstance(model, DiffusionDet):
        model_state_dict_keys = list(model.state_dict().keys())
        loaded_state_dict = remove_modules(model_state_dict_keys, loaded_state_dict, skip_modules)
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, flownet=flownet)

    for name, param in model_state_dict.items():
        if isinstance(param, np.ndarray):
            model_state_dict[name] = torch.from_numpy(param)

    # use strict loading
    model.load_state_dict(model_state_dict)
