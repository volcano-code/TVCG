"""Outward-rounded binary64 intervals for the finite-operand geometry proof.

IEEE basic operations and sqrt are rounded outwards individually. Vector dot
products are explicit three-term sums. No fast-math or BLAS reductions enter
the predicate or bounds.
"""
import numpy as np


def lower(x):
    return np.nextafter(x, -np.inf)


def upper(x):
    return np.nextafter(x, np.inf)


class Interval:
    __array_priority__ = 1000

    def __init__(self, lo, hi):
        self.lo, self.hi = np.broadcast_arrays(np.asarray(lo,dtype=np.float64),
                                                np.asarray(hi,dtype=np.float64))
        if not (np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi))
                and np.all(self.lo <= self.hi)):
            raise ValueError('invalid or overflowed interval')

    @classmethod
    def point(cls, x):
        return x if isinstance(x,cls) else cls(x,x)

    @classmethod
    def ball(cls, x, radius):
        if not np.isfinite(radius) or radius < 0:
            raise ValueError('invalid enclosing radius')
        return cls(lower(np.asarray(x)-radius),upper(np.asarray(x)+radius))

    def __getitem__(self, item):
        return Interval(self.lo[item],self.hi[item])

    def __neg__(self):
        return Interval(-self.hi,-self.lo)

    def __add__(self, other):
        other = self.point(other)
        return Interval(lower(self.lo+other.lo),upper(self.hi+other.hi))

    __radd__ = __add__

    def __sub__(self, other):
        return self+(-self.point(other))

    def __rsub__(self, other):
        return self.point(other)+(-self)

    def __mul__(self, other):
        other = self.point(other)
        products = (self.lo*other.lo,self.lo*other.hi,self.hi*other.lo,self.hi*other.hi)
        return Interval(lower(np.minimum.reduce(products)),upper(np.maximum.reduce(products)))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.point(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            raise ValueError('division denominator may vanish')
        return self*Interval(lower(1./other.hi),upper(1./other.lo))

    def sqrt(self):
        if np.any(self.lo < 0):
            raise ValueError('square root may leave real domain')
        return Interval(np.maximum(0.,lower(np.sqrt(self.lo))),upper(np.sqrt(self.hi)))

    def square(self):
        low = np.where((self.lo<=0)&(self.hi>=0),0.,np.minimum(abs(self.lo),abs(self.hi)))
        high = np.maximum(abs(self.lo),abs(self.hi))
        return Interval(np.maximum(0.,lower(low*low)),upper(high*high))

    def sign(self):
        return (self.lo>0).astype(np.int8)-(self.hi<0).astype(np.int8)

    def abs_upper(self):
        return np.maximum(abs(self.lo),abs(self.hi))

    def abs_lower(self):
        return np.maximum(0.,np.maximum(self.lo,-self.hi))


def stack(values, axis=-1):
    values = [Interval.point(v) for v in values]
    return Interval(np.stack([v.lo for v in values], axis=axis),
                    np.stack([v.hi for v in values], axis=axis))


def dot(a, b):
    p = Interval.point(a)*b
    return p[...,0]+p[...,1]+p[...,2]


def norm(a):
    p = Interval.point(a).square()
    s=p[...,0]+p[...,1]+p[...,2]
    return Interval(np.maximum(0.,s.lo),s.hi).sqrt()


def cross(a, b):
    a,b = Interval.point(a),Interval.point(b)
    return stack([a[...,1]*b[...,2]-a[...,2]*b[...,1],
                  a[...,2]*b[...,0]-a[...,0]*b[...,2],
                  a[...,0]*b[...,1]-a[...,1]*b[...,0]])


def orient(a, b, c, d):
    return dot(cross(b-a,c-a),d-a)
