import numpy as np

def _bit_flip(y, flip, p, t):
    if not flip:
        return y
    temp = np.random.randint(0, 2, size=y.shape)  # Match your original
    k = temp <= p
    if not k.any():
        return y
    u = np.abs(y[k])
    b = np.random.randint(1, t-1, u.shape)
    y[k] = np.sign(y[k]) * np.bitwise_xor(u.astype(np.int32), 1 << (b-1))
    return y

def round_to_nearest(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.abs(x)
    inds = (y - 2 * np.floor(y / 2)) == 0.5
    y[inds] -= 1
    y = np.round(y).clip(min=0)  # Combine operations
    y *= np.sign(x)
    return _bit_flip(y, flip, p, t)

def round_towards_plus_inf(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.ceil(x)
    return _bit_flip(y, flip, p, t)

def round_towards_minus_inf(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.floor(x)
    return _bit_flip(y, flip, p, t)

def round_towards_zero(x, flip=0, p=0.5, t=24, **kwargs):
    y = np.where(x >= 0, np.floor(x), np.ceil(x))  # Simplified
    return _bit_flip(y, flip, p, t)

def stochastic_rounding(x, flip=0, p=0.5, t=24, randfunc=None):
    y = np.abs(x)
    frac = y - np.floor(y)
    if not frac.any():
        return x
    randfunc = randfunc or (lambda shape: np.random.random(shape))
    rnd = randfunc(y.shape)
    y = np.where(rnd <= frac, np.ceil(y), np.floor(y))
    y *= np.sign(x)
    return _bit_flip(y, flip, p, t)

def stochastic_rounding_equal(x, flip=0, p=0.5, t=24, randfunc=None):
    y = np.abs(x)
    frac = y - np.floor(y)
    if not frac.any():
        return x
    randfunc = randfunc or (lambda shape: np.random.random(shape))
    rnd = randfunc(y.shape)
    y = np.where(rnd <= 0.5, np.ceil(y), np.floor(y))
    y *= np.sign(x)
    return _bit_flip(y, flip, p, t)

def roundit_test(x, rmode=1, flip=0, p=0.5, t=24, randfunc=None):
    randfunc = randfunc or (lambda shape: np.random.randint(0, 2, shape))
    
    if rmode == 1:
        y = np.abs(x)
        y = np.round(y - (y % 2 == 0.5)).clip(min=0)
        y *= np.sign(x)
    elif rmode == 2:
        y = np.ceil(x)
    elif rmode == 3:
        y = np.floor(x)
    elif rmode == 4:
        y = np.where(x >= 0, np.floor(x), np.ceil(x))
    elif rmode in (5, 6):
        y = np.abs(x)
        frac = y - np.floor(y)
        if not frac.any():
            return x
        rnd = randfunc(y.shape)
        thresh = frac if rmode == 5 else 0.5
        y = np.where(rnd <= thresh, np.ceil(y), np.floor(y))
        y *= np.sign(x)
    else:
        raise ValueError('Unsupported value of rmode.')
    
    return _bit_flip(y, flip, p, t)

def return_column_order(arr):
    return arr.T.reshape(-1)