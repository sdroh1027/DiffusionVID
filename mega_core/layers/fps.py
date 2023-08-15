from mega_core import _C

from apex import amp

# Only valid with fp32 inputs - give AMP the hint
fps = amp.float_function(_C.furthest_point_sampling)