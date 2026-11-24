// Fast path for a Cherchi packet whose emitted faces already have coherent winding.
// Integrals still run in the registered MPFR/GMP solver. Indeterminate orientation
// is delegated to that solver's original exact, per-component implementation.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <gmpxx.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Point = std::array<mpq_class, 3>;
using Triple = std::array<double,3>;
using TokenRow = std::array<std::string,3>;
using Matrix = std::array<Triple,3>;
using Doubles = py::array_t<double,py::array::c_style | py::array::forcecast>;
using Indices = py::array_t<std::int64_t,py::array::c_style | py::array::forcecast>;

struct ReferenceOrientation : std::runtime_error { using std::runtime_error::runtime_error; };

static mpq_class parse(const std::string &s) {
    mpq_class value;
    if (s.empty() || mpq_set_str(value.get_mpq_t(),s.c_str(),0) || !mpz_sgn(mpq_denref(value.get_mpq_t())))
        throw std::invalid_argument("invalid exact coordinate token");
    value.canonicalize();
    return value;
}

// mpq_get_d truncates toward zero. Resolve the exact binary64 midpoint including ties.
static double nearest(const mpq_class &q) {
    if (!sgn(q)) return 0.;
    double a = q.get_d();
    if (!std::isfinite(a)) throw std::invalid_argument("nonfinite coordinate");
    if (q==mpq_class(a)) return a;
    double b = std::nextafter(a, std::copysign(std::numeric_limits<double>::infinity(),sgn(q)));
    if (!std::isfinite(b)) throw ReferenceOrientation("extreme range requires registered rational path");
    auto cmp = sgn(mpq_class(sgn(q))*(2*q-mpq_class(a)-mpq_class(b)));
    if (cmp) return cmp>0 ? b : a;
    std::uint64_t bits;
    std::memcpy(&bits,&a,sizeof(a));
    return (bits&1u) ? b : a;
}

class Components {
    std::vector<int> parent;
public:
    explicit Components(int n):parent(n) { std::iota(parent.begin(),parent.end(),0); }
    int find(int n) {
        while (n!=parent[n]) { parent[n]=parent[parent[n]]; n=parent[n]; }
        return n;
    }
    void connect(int a,int b) {
        a=find(a); b=find(b);
        if (a!=b) parent[std::max(a,b)]=std::min(a,b);
    }
};

struct EdgeUse { int face; bool positive, paired; };

static void check_components(const std::vector<Point> &xyz,const Indices &indices) {
    auto f=indices.unchecked<2>();
    if (f.shape(1)!=3 || f.shape(0)<4 || f.shape(0)>std::numeric_limits<int>::max())
        throw ReferenceOrientation("invalid boundary shape");
    Components component(f.shape(0));
    std::unordered_map<std::uint64_t,EdgeUse> edges;
    edges.reserve(2*f.shape(0));
    for (int i=0;i<f.shape(0);++i) for (int k=0;k<3;++k) {
        auto a=f(i,k), b=f(i,(k+1)%3);
        if (a<0 || b<0 || a==b || a>=static_cast<std::int64_t>(xyz.size()) || b>=static_cast<std::int64_t>(xyz.size()))
            throw ReferenceOrientation("invalid face indices");
        auto key=(static_cast<std::uint64_t>(std::min(a,b))<<32) | static_cast<std::uint32_t>(std::max(a,b));
        auto [it,inserted]=edges.try_emplace(key,EdgeUse{i,a<b,false});
        if (!inserted) {
            auto &last=it->second;
            if (last.paired || last.positive==(a<b)) throw ReferenceOrientation("requires exact winding normalization");
            component.connect(i,last.face);
            last.paired=true;
        }
    }
    for (const auto &edge:edges) if (!edge.second.paired) throw ReferenceOrientation("open boundary");
    // Bound the exact signed volume of EACH edge-connected shell. With coherent
    // input the emitted and coherent sums in the reference are identical. Use
    // outward rounding on the exact determinants; defer any inconclusive sign.
    std::unordered_map<int,std::pair<long double,long double>> totals;
    for (int i=0;i<f.shape(0);++i) {
        const int root=component.find(i);
        const auto &o=xyz[f(root,0)], &a=xyz[f(i,0)], &b=xyz[f(i,1)], &c=xyz[f(i,2)];
        Point u,v;
        for (int k=0;k<3;++k) {u[k]=b[k]-a[k]; v[k]=c[k]-a[k];}
        mpq_class term=0;
        bool area=false;
        for (int k=0;k<3;++k) {
            int l=(k+1)%3, m=(k+2)%3;
            mpq_class cross=u[l]*v[m]-u[m]*v[l];
            area=area || sgn(cross)!=0;
            term+=(a[k]-o[k])*cross;
        }
        if (!area) throw ReferenceOrientation("exact zero-area face");
        if (!sgn(term)) continue;
        double trunc=term.get_d();
        if (!std::isfinite(trunc)) throw ReferenceOrientation("unbounded component");
        double lo=sgn(term)<0 ? std::nextafter(trunc,-INFINITY):trunc;
        double hi=sgn(term)>0 ? std::nextafter(trunc, INFINITY):trunc;
        auto &sum=totals[root];
        sum.first=std::nextafter(sum.first+static_cast<long double>(lo),-HUGE_VALL);
        sum.second=std::nextafter(sum.second+static_cast<long double>(hi),HUGE_VALL);
    }
    for (int i=0;i<f.shape(0);++i) if (component.find(i)==i) {
        auto [lo,hi]=totals[i];
        const long double min=6.L*std::numeric_limits<double>::min();
        if (!(lo>min || hi< -min) || std::max(std::abs(lo),std::abs(hi))>=std::numeric_limits<double>::max())
            throw ReferenceOrientation("registered exact volume check required");
    }
}

static py::tuple transform(const std::vector<TokenRow> &tokens,const Doubles &emitted,
        const Indices &faces,const Triple &origin,double length,const Triple &target_origin,const Matrix &axes) {
    if (!std::isfinite(length) || length<=0 || tokens.size()<4 || tokens.size()>UINT32_MAX)
        throw ReferenceOrientation("empty or invalid physical boundary");
    auto display=emitted.unchecked<2>();
    if (display.shape(0)!=static_cast<py::ssize_t>(tokens.size()) || display.shape(1)!=3)
        throw std::invalid_argument("coordinate display shape");
    py::array_t<double> result({static_cast<py::ssize_t>(tokens.size()),static_cast<py::ssize_t>(3)});
    auto out=result.mutable_unchecked<2>();
    Point offset;
    std::array<Point,3> coefficients;
    for (int j=0;j<3;++j) {
        offset[j]=0;
        for (int k=0;k<3;++k) {
            if (!std::isfinite(axes[k][j])|| !std::isfinite(origin[k]) || !std::isfinite(target_origin[k]))
                throw std::invalid_argument("nonfinite affine frame");
            coefficients[j][k]=mpq_class(axes[k][j])/mpq_class(length);
            offset[j]-=mpq_class(origin[k])*coefficients[j][k]+mpq_class(target_origin[k])*mpq_class(axes[k][j]);
        }
    }
    std::vector<Point> physical(tokens.size());
    std::vector<TokenRow> target(tokens.size());
    {
        py::gil_scoped_release guard;
        for (std::size_t i=0;i<tokens.size();++i) {
            auto &p=physical[i];
            for (int k=0;k<3;++k) {
                p[k]=parse(tokens[i][k]);
                if (nearest(p[k])!=display(i,k)) throw std::invalid_argument("rational/display mismatch");
            }
            for (int j=0;j<3;++j) {
                mpq_class y=offset[j];
                for (int k=0;k<3;++k) if (sgn(coefficients[j][k])) y+=coefficients[j][k]*p[k];
                out(i,j)=nearest(y);
                target[i][j]=y.get_str();
            }
        }
        check_components(physical,faces);
    }
    return py::make_tuple(std::move(result),std::move(target));
}

PYBIND11_MODULE(_tvcg_stratum_primal,m) {
    py::register_exception<ReferenceOrientation>(m,"ReferenceOrientation");
    m.def("transform",&transform);
}
