#include "spin/QuadricTriangulator.h"
#include <Eigen/Eigenvalues>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>
namespace py=pybind11;
using Q=spin::QuadricCoefficients10;
using Solver=Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>;
extern spin::QuadricSurfaceType originalClassify(const Q&,Solver*)
    asm("_ZN12_GLOBAL__N_18classifyERKN4spin21QuadricCoefficients10EPN5Eigen22SelfAdjointEigenSolverINS4_6MatrixIdLi3ELi3ELi0ELi3ELi3EEEEE");
using Product=std::array<std::vector<double>,6>;
using Vector=Eigen::Vector3d;
using Matrix=Eigen::Matrix3d;
constexpr double err=0x1p-46;  // > 32 binary64 epsilons; see explicit magnitude guards below.

struct Statistics {
    uint64_t enumerated=0,duplicates=0;
    double maxResidual=0,maxOrtho=0,maxRatio=0,minGap=10,minSignGap=10,minAxial=10,minLeafMargin=10;
    double maxAbsNull=0;
    std::array<std::array<double,2>,3> eigenBounds{{{{1.e6,-1.e6}},{{1.e6,-1.e6}},{{1.e6,-1.e6}}}};
};

bool contains(const Product& p,const std::array<double,6>& x) {
    for(int k=0;k<6;++k)
        if(!std::binary_search(p[k].begin(),p[k].end(),x[k]))return false;
    return true;
}

void check(const std::array<double,6>& x,const Q::Vector& reference,
           const Eigen::Matrix<double,3,2>& bRanges,int nullIndex,Statistics& stats) {
    Q::Vector v=reference;
    for(int k=0;k<6;++k)v[k+4]=x[k];
    // The constructor and classifier come from the very same verified objects.
    // Changing the linear reference below leaves the eigensolver input identical.
    const auto q=Q::fromNormalized(v);
    Solver eig;
    if(originalClassify(q,&eig)!=spin::QuadricSurfaceType::Paraboloid || eig.info()!=Eigen::Success)
        throw std::runtime_error("native classifier rejected an enumerated transverse quadratic block");
    const Vector ev=eig.eigenvalues();
    const Matrix ax=eig.eigenvectors(),a=q.quadraticMatrix();
    int k;
    const double null=ev.cwiseAbs().minCoeff(&k), scale=ev.cwiseAbs().maxCoeff();
    if(k!=nullIndex || !ev.allFinite() || !ax.allFinite() ||
       scale>2 || a.cwiseAbs().maxCoeff()>1.01 || ax.cwiseAbs().maxCoeff()>1.01)
        throw std::runtime_error("enumerated spectral or magnitude branch changed");
    const double maxR=(a*ax-ax*ev.asDiagonal()).cwiseAbs().maxCoeff();
    const double maxO=(ax.transpose()*ax-Matrix::Identity()).cwiseAbs().maxCoeff();
    // Each dot has at most 3 products and 2 sums; per entry the sum of
    // magnitudes is <3.061, the lambda term <2.02. The 2^-46 allowance
    // strictly exceeds the forward error of the rounded sums/products and
    // final subtraction (including denormals). The native guard uses exact Q.
    if (maxR+err>=0x1p-42 || maxO+err>=0x1p-42)
        throw std::runtime_error("rational eigenframe acceptance not enclosed");
    const double leaf=64*std::numeric_limits<double>::epsilon()*scale;
    if(null>leaf || !(scale>0))throw std::runtime_error("represented transverse curvature exceeds native rank leaf");
    const double gap=std::min(ev[1]-ev[0],ev[2]-ev[1]);
    if(gap<=1.e-10*std::max(1.,scale))throw std::runtime_error("eigenframe motion gap unresolved");
    // Check a stronger all-column native sign-pivot condition.
    for(int j=0;j<3;++j){
        std::array<double,3> c={std::abs(ax(0,j)),std::abs(ax(1,j)),std::abs(ax(2,j))};
        std::sort(c.begin(),c.end()); stats.minSignGap=std::min(stats.minSignGap,c[2]-c[1]);
    }
    if(stats.minSignGap<=1.e-10)throw std::runtime_error("eigenframe sign chart gap unresolved");
    double lower=0,upper=0;
    for(int i=0;i<3;++i){
        const double lo=ax(i,k)*bRanges(i,0),hi=ax(i,k)*bRanges(i,1);
        lower+=std::min(lo,hi);upper+=std::max(lo,hi);
    }
    const double axial=std::max(lower,-upper)-err;
    if(axial<=4.e-12)throw std::runtime_error("transverse linear classification may change inside box");
    stats.enumerated++;
    stats.maxResidual=std::max(stats.maxResidual,maxR);stats.maxOrtho=std::max(stats.maxOrtho,maxO);
    stats.maxRatio=std::max(stats.maxRatio,null/leaf);stats.maxAbsNull=std::max(stats.maxAbsNull,null);
    stats.minLeafMargin=std::min(stats.minLeafMargin,leaf-null);
    stats.minGap=std::min(stats.minGap,gap);stats.minAxial=std::min(stats.minAxial,axial);
    for(int i=0;i<3;++i){stats.eigenBounds[i][0]=std::min(stats.eigenBounds[i][0],ev[i]);
                           stats.eigenBounds[i][1]=std::max(stats.eigenBounds[i][1],ev[i]);}
}

py::dict cover(const std::vector<Product>& products,const Q::Vector& ref,
               const Eigen::Matrix<double,3,2>& linear,int nullIndex) {
    if(products.empty() || nullIndex<0 || nullIndex>2 || linear.cwiseAbs().maxCoeff()>1.01 ||
       !linear.allFinite() || (linear.col(0).array()>linear.col(1).array()).any())
        throw std::invalid_argument("invalid coefficient cover");
    Statistics stats;
    for(size_t n=0;n<products.size();++n){
        const auto& p=products[n];
        for(const auto& col:p)if(col.empty() || !std::is_sorted(col.begin(),col.end()) ||
            std::adjacent_find(col.begin(),col.end())!=col.end())throw std::invalid_argument("invalid product axis");
        std::array<size_t,6> index{}; std::array<double,6>x{};
        for(;;){
            for(int k=0;k<6;++k)x[k]=p[k][index[k]];
            bool seen=false;
            for(size_t j=0;j<n && !seen;++j)seen=contains(products[j],x);
            if(seen)++stats.duplicates;else check(x,ref,linear,nullIndex,stats);
            int k=5;
            while(k>=0 && ++index[k]==p[k].size()){index[k]=0;--k;}
            if(k<0)break;
        }
    }
    py::dict d;
    d["pair_count"]=stats.enumerated+stats.duplicates; d["distinct_matrices"]=stats.enumerated;
    d["overlap_duplicates"]=stats.duplicates;d["minimum_transverse_abs"]=stats.minAxial;
    d["eigenvalue_bounds"]=stats.eigenBounds;d["maximum_abs_null_eigenvalue"]=stats.maxAbsNull;
    d["maximum_native_leaf_fraction"]=stats.maxRatio;d["minimum_native_leaf_margin"]=stats.minLeafMargin;
    d["minimum_eigengap"]=stats.minGap;d["minimum_eigenvector_pivot_gap"]=stats.minSignGap;
    d["conservative_residual_upper"]=std::nextafter(stats.maxResidual+err,INFINITY);
    d["conservative_orthogonality_upper"]=std::nextafter(stats.maxOrtho+err,INFINITY);
    return d;
}

PYBIND11_MODULE(_rank_admission_probe,m){
    m.def("normalize",[](const Q::Vector& q)->Q::Vector{return Q::fromUnnormalized(q).values();});
    m.def("cover",&cover);
}
