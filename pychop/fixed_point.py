import os

def to_fixed_point(x, ibits=4, fbits=4):
    print(os.environ['chop_backend'])
    if os.environ['chop_backend'] == 'torch':
        from .tch import fixed_point
        return fixed_point.to_fixed_point(x, ibits=4, fbits=4)
    else:
        from .np import fixed_point
        return fixed_point.to_fixed_point(x, ibits=4, fbits=4)