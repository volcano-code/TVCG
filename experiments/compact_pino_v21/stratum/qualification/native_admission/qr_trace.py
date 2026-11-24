"""Binary64 interval execution of the pinned Eigen 3x3 tridiagonal QR value path."""
import math
from dataclasses import dataclass

EPS = 2.**-52
SMALLEST_NORMAL = 2.**-1022


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo>self.hi:
            raise ValueError('non-finite or reversed machine interval')
    @classmethod
    def point(cls, x):
        return x if isinstance(x,cls) else cls(float(x),float(x))
    def __neg__(self): return Interval(-self.hi,-self.lo)
    def __add__(self,x):
        x=self.point(x);return Interval(self.lo+x.lo,self.hi+x.hi)
    __radd__=__add__
    def __sub__(self,x):return self+(-self.point(x))
    def __rsub__(self,x):return self.point(x)+(-self)
    def __mul__(self,x):
        x=self.point(x); v=[p*q for p in (self.lo,self.hi) for q in (x.lo,x.hi)];return Interval(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,x):
        x=self.point(x)
        if x.lo<=0<=x.hi: raise ValueError('machine interval divisor includes zero')
        v=[p/q for p in (self.lo,self.hi) for q in (x.lo,x.hi)]
        return Interval(min(v),max(v))
    def __rtruediv__(self,x):return self.point(x)/self
    def sqrt(self):return Interval(math.sqrt(self.lo),math.sqrt(self.hi))
    def square(self):
        a=abs(self);return Interval(a.lo*a.lo,a.hi*a.hi)
    def __abs__(self):
        return Interval(max(self.lo,-self.hi,0.),max(abs(self.lo),abs(self.hi)))
    def span(self):return [self.lo,self.hi]


class Unresolved(Exception):
    def __init__(self, guard, left, right, state=None):
        self.record=dict(guard=guard,left=left.span(),right=right.span(),state=state or {})
        super().__init__(guard)


def leq(left,right,label,state):
    left,right=Interval.point(left),Interval.point(right)
    if left.hi<=right.lo:return True
    if left.lo>right.hi:return False
    raise Unresolved(label,left,right,state)


def maximum(x,y):return Interval(max(x.lo,y.lo),max(x.hi,y.hi))
def minimum(x,y):return Interval(min(x.lo,y.lo),min(x.hi,y.hi))


def givens(x,z,state):
    zero=Interval.point(0)
    if leq(abs(z),zero,'Givens second component == 0',state):
        return Interval.point(-1 if leq(x,zero,'first component sign',state) and x.hi<0 else 1),zero
    if leq(abs(x),zero,'Givens first component == 0',state):
        return zero,Interval.point(1 if z.hi<0 else -1)
    if not leq(abs(x),abs(z),'Givens magnitude comparison',state):
        t=z/x; u=(1+t.square()).sqrt()
        if leq(x,zero,'Givens first sign',state):u=-u
        c=1/u
        return c,-t*c
    t=x/z;u=(1+t.square()).sqrt()
    if leq(z,zero,'Givens second sign',state):u=-u
    s=-1/u
    return -t*s,s


def trace(coefficients):
    z=Interval.point(0);one=Interval.point(1)
    a00,a11,a22,a21,a20,a10=map(Interval.point,coefficients)
    scale=z
    for t in (a00,a11,a22,a21,a20,a10):scale=maximum(scale,abs(t))
    # For an actual singleton with identical input to scale, division is one.
    # This exact common subexpression is valid on a box when its maximum is unique.
    inputs=(a00,a11,a22,a21,a20,a10)
    def scaled(i):
        t=inputs[i]
        if all(abs(t).lo>abs(other).hi for j,other in enumerate(inputs) if i!=j):
            return one if t.lo>0 else -one
        return t/scale
    a00,a11,a22,a21,a20,a10=map(scaled,range(6))
    if leq(a20.square(),Interval.point(SMALLEST_NORMAL),'tridiagonal structural-zero comparison',{}):
        d=[a00,a11,a22]; e=[a10,a21]; v=[[one,z,z],[z,one,z],[z,z,one]]
    else:
        beta=(a10.square()+a20.square()).sqrt();inv=one/beta
        n1=a10*inv;n2=a20*inv
        h=2*n1*a21+n2*(a22-a11)
        d=[a00,a11+n2*h,a22-n2*h];e=[beta,a21-n1*h]
        v=[[one,z,z],[z,n1,n2],[z,n2,-n1]]
    end,start,iteration=2,0,0
    steps=[]
    while end>0:
        state=dict(iteration=iteration,start=start,end=end)
        for i in range(start,end):
            threshold=(abs(d[i])+abs(d[i+1]))*(2*EPS)
            if leq(abs(e[i]),maximum(threshold,Interval.point(SMALLEST_NORMAL)),
                   'QR deflation abs(e[i]) <= 2*eps*(abs(d[i])+abs(d[i+1]))',dict(state,index=i)):
                e[i]=z
        while end>0 and e[end-1].lo==0 and e[end-1].hi==0:end-=1
        if not end:break
        iteration+=1
        if iteration>90:raise ValueError('native QR iteration budget exhausted')
        start=end-1
        while start>0 and not (e[start-1].lo==0 and e[start-1].hi==0):start-=1
        state=dict(iteration=iteration,start=start,end=end)
        td=(d[end-1]-d[end])*.5; ei=e[end-1];mu=d[end]
        if leq(abs(td),z,'Wilkinson half-difference == 0',state):mu=mu-abs(ei)
        else:
            x,y=abs(td),abs(ei);p=maximum(x,y);t=minimum(x,y)/p
            h=p*(1+t.square()).sqrt()
            e2=ei.square();positive=td.lo>0
            if leq(e2,z,'Wilkinson shift square underflow',state):
                mu=mu-(ei/(td+(1 if positive else -1)))*(ei/h)
            else:mu=mu-e2/(td+(h if positive else -h))
        x=d[start]-mu;zz=e[start]
        for k in range(start,end):
            c,s=givens(x,zz,dict(state,k=k))
            dk,dk1,de=d[k],d[k+1],e[k]
            sdk=s*dk+c*de; dkp1=s*de+c*dk1
            d[k]=c*(c*dk-s*de)-s*(c*de-s*dk1)
            d[k+1]=s*sdk+c*dkp1
            e[k]=c*sdk-s*dkp1
            if k>start:e[k-1]=c*e[k-1]-s*zz
            x=e[k]
            if k<end-1:zz=-s*e[k+1];e[k+1]=c*e[k+1]
            for row in v:
                a,b=row[k:k+2]
                row[k]=c*a-s*b;row[k+1]=s*a+c*b
        steps.append(state)
    for i in range(2):
        j=i
        for k in range(i+1,3):
            if leq(d[k],d[j],'eigenvalue sorting order',dict(index=i)):j=k
        d[i],d[j]=d[j],d[i]
        for row in v:row[i],row[j]=row[j],row[i]
    return dict(values=[x*scale for x in d],vectors=v,steps=steps)
