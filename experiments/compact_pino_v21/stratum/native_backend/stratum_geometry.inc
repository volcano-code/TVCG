// Compiled inside the pinned candidate-03 triangulator to reuse its native
// rank decisions, square frame, radius, face stitching, orientation and hashes.
#include "stratum_api.h"
#include "spin/QuadricStratumTangent.h"
#include <CGAL/Sqrt_extension.h>
#include <cstring>

namespace compact_stratum {
namespace {
using Q = CGAL::Gmpq;
using E = CGAL::Sqrt_extension<Q,Q>;
using V = Eigen::Vector3d;
using R3 = std::array<Q,3>;
using Polynomial = spin::QuadricCoefficients10;
using Settings = spin::QuadricTriangulationSettings;

R3 rational(const V& p) { return {Q(p[0]),Q(p[1]),Q(p[2])}; }
Q dot(const R3& x, const R3& y) { return x[0]*y[0]+x[1]*y[1]+x[2]*y[2]; }
R3 subtract(const R3& x,const R3& y) { return {x[0]-y[0],x[1]-y[1],x[2]-y[2]}; }
R3 cross(const R3& x,const R3& y) {
    return {x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]};
}

// Same nearest-even binary64 rule as representedAlgebraic in the source of
// native-side-build-job-03. First try the stable numerical root as a hint.
// All acceptance and fallback decisions compare the exact algebraic value.
double nearestEven(const E& exact, double hint) {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double)==8);
    if (exact==E(0)) return 0.;
    const bool negative=exact<E(0);
    const E value=negative ? -exact : exact;
    const auto decode=[](uint64_t bits){double x; std::memcpy(&x,&bits,8); return x;};
    const uint64_t last=UINT64_C(0x7fefffffffffffff);
    double guess=std::abs(hint);
    uint64_t bits;
    std::memcpy(&bits,&guess,8);
    if (bits>0 && bits<last) {
        const E lo((Q(decode(bits-1))+Q(guess))/2), hi((Q(guess)+Q(decode(bits+1)))/2);
        const bool even=(bits&1)==0;
        if ((value>lo || (even && value==lo)) && (value<hi || (even && value==hi)))
            return negative ? -guess : guess;
    }
    if (value>E(Q(decode(last)))) throw std::domain_error("algebraic graph coordinate overflow");
    uint64_t low=0, high=last;
    while (low<high) {
        const uint64_t mid=low+(high-low+1)/2;
        if (value>=E(Q(decode(mid)))) low=mid;
        else high=mid-1;
    }
    if (low<last) {
        const E half((Q(decode(low))+Q(decode(low+1)))/2);
        if (value>half || (value==half && (low&1))) ++low;
    }
    return negative ? -decode(low) : decode(low);
}

struct Chart {
    Eigen::Matrix3d eigenvectors=Eigen::Matrix3d::Identity();
    int u=0, v=1;
    double nullSign=1;
};

void selectFrame(const Polynomial& q, spin::QuadricSurfaceType type,
                 const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>& eigen, Chart& chart, Result& out) {
    if (type==spin::QuadricSurfaceType::Plane) {
        if (!(q.quadraticMatrix().array()==0.).all())
            throw std::invalid_argument("R3 algebraic plane requires six exactly zero raw curvatures");
        out.axis=q.linearVector().normalized();
        const auto frame=spin::makeQuadricPlaneFrame(out.axis);
        out.first=frame.first;
        out.second=frame.second;
        out.referenceAxis=frame.chart==0 ? 0 : 1;
        out.frameMargin=std::abs(std::abs(out.axis.x())-0.9);
        if (!frame.chartStable) out.derivativeDiagnostic=frame.diagnostic;
        return;
    }
    if (type!=spin::QuadricSurfaceType::Paraboloid)
        throw std::invalid_argument("algebraic rank-chart entry supports only R3 planes and R8 paraboloids");
    chart.eigenvectors=canonicalEigenvectorSigns(eigen.eigenvectors());
    out.eigenvalues=eigen.eigenvalues();
    // Verify the represented eigenpairs used by this actual map in exact
    // binary-rational arithmetic. The bounds apply to normalized coefficients.
    Q residual(0), orthogonality(0);
    for (int j=0;j<3;++j) {
        const auto column=rational(chart.eigenvectors.col(j));
        for (int i=0;i<3;++i) {
            const Q error=dot(rational(q.quadraticMatrix().row(i)),column)-Q(out.eigenvalues[j])*column[i];
            const Q defect=dot(rational(chart.eigenvectors.col(i)),column)-Q(i==j ? 1 : 0);
            residual=std::max(residual,CGAL::abs(error));
            orthogonality=std::max(orthogonality,CGAL::abs(defect));
        }
    }
    if (residual>Q(0x1p-42) || orthogonality>Q(0x1p-42))
        throw std::domain_error("R8 represented eigenframe fails its rational backward-error guard");
    out.eigenResidual=CGAL::to_double(residual);
    out.eigenOrthogonality=CGAL::to_double(orthogonality);
    out.eigenvalues.cwiseAbs().minCoeff(&out.nullIndex);
    const double scale=out.eigenvalues.cwiseAbs().maxCoeff();
    // Recorded R8s are rounded representations of rank-two charts. Retain
    // their least eigenvalue in the polynomial. Only a roundoff-scale leaf is
    // admitted; an exact zero determinant is not asserted.
    if (std::abs(out.eigenvalues[out.nullIndex])>64*std::numeric_limits<double>::epsilon()*scale)
        throw std::invalid_argument("R8 represented transverse curvature exceeds roundoff-scale chart");
    chart.u=(out.nullIndex==0 ? 1 : 0);
    chart.v=(out.nullIndex==2 ? 1 : 2);
    chart.nullSign=q.linearVector().dot(chart.eigenvectors.col(out.nullIndex))<0 ? -1 : 1;
    out.axis=chart.nullSign*chart.eigenvectors.col(out.nullIndex);
    out.first=chart.eigenvectors.col(chart.u);
    out.second=chart.eigenvectors.col(chart.v);
    out.frameMargin=std::numeric_limits<double>::max();
    for (int j=0;j<3;++j) {
        std::array<std::pair<double,int>,3> pivots;
        for (int i=0;i<3;++i) pivots[i]={std::abs(chart.eigenvectors(i,j)),i};
        std::sort(pivots.rbegin(),pivots.rend());
        if (chart.eigenvectors(pivots[0].second,j)*chart.eigenvectors(pivots[1].second,j)<0)
            out.frameMargin=std::min(out.frameMargin,pivots[0].first-pivots[1].first);
    }
    for (int k=1;k<3;++k)
        if (out.eigenvalues[k]-out.eigenvalues[k-1]<=1.e-10*std::max(1.,scale))
            out.derivativeDiagnostic="R8 motion requires a separated eigenspectrum";
    if (out.frameMargin<=1.e-10) out.derivativeDiagnostic="R8 motion crosses an eigenvector sign-pivot seam";
}

struct Graph {
    std::array<R3,3> a;
    R3 b,n,u,v,an;
    Q c,alpha,bn;
    explicit Graph(const Polynomial& q,const Result& r)
        : b(rational(q.linearVector())), n(rational(r.axis)), u(rational(r.first)),
          v(rational(r.second)), c(q.constant()) {
        for (int i=0;i<3;++i) a[i]=rational(q.quadraticMatrix().row(i));
        for (int i=0;i<3;++i) an[i]=dot(a[i],n);
        alpha=dot(n,an); bn=dot(b,n);
        if (bn<=0) throw std::domain_error("canonical graph has no increasing transverse axis");
    }
    V point(double s,double t,double& height) const {
        R3 y,ay;
        for (int i=0;i<3;++i) y[i]=Q(s)*u[i]+Q(t)*v[i];
        for (int i=0;i<3;++i) ay[i]=dot(a[i],y);
        const Q linear=bn+2*dot(y,an), constant=c+dot(b,y)+dot(y,ay);
        if (linear<=0) throw std::domain_error("finite graph branch lost its positive axial coefficient");
        E root;
        double hint;
        if (alpha==0) {
            const Q h=-constant/linear;
            root=E(h); hint=CGAL::to_double(h);
        } else {
            const Q discriminant=linear*linear-4*alpha*constant;
            if (discriminant<=0) throw std::domain_error("finite graph has no regular real axial root");
            root=E(-linear/(2*alpha),Q(1)/(2*alpha),discriminant);
            hint=-2*CGAL::to_double(constant)/(CGAL::to_double(linear)+std::sqrt(CGAL::to_double(discriminant)));
        }
        height=nearestEven(root,hint);
        V p;
        for (int i=0;i<3;++i) {
            const E x=E(y[i])+root*E(n[i]);
            p[i]=nearestEven(x,CGAL::to_double(y[i])+CGAL::to_double(n[i])*height);
        }
        return p;
    }
    Q slope(const R3& p) const {return bn+2*dot(an,p);}
};

std::pair<Q,Q> boxSupport(const R3& direction, const V& lo, const V& hi) {
    Q bottom(0),top(0);
    for (int i=0;i<3;++i) {
        Q a=direction[i]*Q(lo[i]), b=direction[i]*Q(hi[i]);
        bottom+=std::min(a,b); top+=std::max(a,b);
    }
    return {bottom,top};
}

struct Grid {
    std::vector<V> points;
    std::vector<std::array<double,3>> coordinates;
    std::vector<Face> faces;
};

Grid makeGrid(const Graph& graph, const Result& out, bool plane, int resolution) {
    Grid g;
    const double r=out.radius;
    const auto add=[&](double s,double t) {
        double w;
        g.points.push_back(graph.point(s,t,w));
        g.coordinates.push_back({s,t,w});
    };
    if (plane) {
        add(-r,-r); add(r,-r); add(r,r); add(-r,r);
        g.faces={{0,1,2},{0,2,3}};
    } else {
        for (int j=0;j<=resolution;++j) for (int i=0;i<=resolution;++i)
            add(-r+2*r*i/resolution,-r+2*r*j/resolution);
        for (int j=0;j<resolution;++j) for (int i=0;i<resolution;++i) {
            const int a=j*(resolution+1)+i;
            g.faces.push_back({a,a+1,a+resolution+2});
            g.faces.push_back({a,a+resolution+2,a+resolution+1});
        }
    }
    // All graph faces have a common axial winding. Directed lower faces
    // point away from the positive extrusion. Reject folds or zero area.
    const int winding=dot(cross(graph.u,graph.v),graph.n)>0 ? 1 : -1;
    std::vector<R3> exact;
    for (const auto& p:g.points) exact.push_back(rational(p));
    for (auto& f:g.faces) {
        const Q crossN=dot(cross(subtract(exact[f[1]],exact[f[0]]),subtract(exact[f[2]],exact[f[0]])),graph.n);
        if (winding*crossN<=0) throw std::domain_error("finite graph triangle lost chart winding");
        if (winding>0) std::swap(f[1],f[2]);
    }
    return g;
}

void closeGraph(const Graph& graph, const Grid& g, const Eigen::MatrixXd& domain,
                spin::QuadricSurfaceType type, const Settings& settings, Result& out) {
    const V lo=domain.colwise().minCoeff(), hi=domain.colwise().maxCoeff();
    const auto nbox=boxSupport(graph.n,lo,hi);
    const auto ubox=boxSupport(graph.u,lo,hi), vbox=boxSupport(graph.v,lo,hi);
    const double clearanceMargin=1.e-10*std::max(1.,domain.cwiseAbs().maxCoeff());
    const Q margin(clearanceMargin);
    Eigen::MatrixXd box(8,3);
    for (int i=0;i<8;++i) for (int j=0;j<3;++j) box(i,j)=(i&(1<<j)) ? hi[j] : lo[j];
    const spin::detail::ProjectedDomainClearance clearance(box,out.axis,clearanceMargin);
    std::vector<V> rim;
    const auto loops=boundaryLoops(g.faces);
    if (loops.size()!=1) throw std::domain_error("finite graph projection has no single rim");
    for (int i:loops.front()) rim.push_back(g.points[i]);
    if (!clearance.inside(rim)) throw std::domain_error("finite graph rim does not strictly enclose design box projection");
    // Positive slope throughout the represented design AABB is a linear
    // inequality. Evaluate its true minimum on that box.
    if (graph.bn+2*boxSupport(graph.an,lo,hi).first<=0)
        throw std::domain_error("axial derivative changes side on design bounds");
    for (int k=0;k<=settings.maximumRadiusDoublings;++k) {
        const double depth=std::ldexp(4*out.radius,k);
        if (!std::isfinite(depth)) break;
        const V extrude=depth*out.axis;
        std::optional<Q> far;
        for (const auto& p:g.points) {
            const R3 point=rational(p+extrude);
            const Q gap=dot(point,graph.n)-nbox.second;
            if (!far || gap<*far) far=gap;
        }
        if (*far<=margin) {
            out.rejectedFarGap=CGAL::to_double(*far);
            continue;
        }
        out.farGap=CGAL::to_double(*far);
        out.mesh=closeExtrudedInterface(type,g.points,g.faces,extrude,out.radius);
        out.doublings=k; out.depth=depth;
        break;
    }
    if (out.depth<=0) throw std::domain_error("finite cap did not clear design box within growth budget");
    const auto& m=out.mesh.closedBoundary;
    std::vector<R3> p;
    for (int i=0;i<m.vertices.rows();++i) p.push_back(rational(m.vertices.row(i)));
    std::optional<Q> minimumSlope;
    for (const auto& v:p) {
        const Q slope=graph.slope(v);
        if (slope<=0) throw std::domain_error("selected side reverses within extruded source");
        if (!minimumSlope || slope<*minimumSlope) minimumSlope=slope;
    }
    out.minimumRootSlope=CGAL::to_double(*minimumSlope);
    std::optional<Q> lateral;
    for (int f=0;f<m.faces.rows();++f) {
        const R3& a=p[m.faces(f,0)], &b=p[m.faces(f,1)], &c=p[m.faces(f,2)];
        if (f<out.mesh.interfaceFaceCount) {
            if (dot(cross(subtract(b,a),subtract(c,a)),graph.n)>=0)
                throw std::domain_error("outward lower face disagrees with canonical extrusion side");
        } else if (f>=2*out.mesh.interfaceFaceCount) {
            const auto gap=[&](const R3& axis,const std::pair<Q,Q>& box) {
                const Q x=dot(a,axis),y=dot(b,axis),z=dot(c,axis);
                return std::max(box.first-std::max({x,y,z}),std::min({x,y,z})-box.second);
            };
            const Q separation=std::max(gap(graph.u,ubox),gap(graph.v,vbox));
            if (separation<=margin) throw std::domain_error("lateral cap intersects design projection bounds");
            if (!lateral || separation<*lateral) lateral=separation;
        }
    }
    out.lateralGap=CGAL::to_double(*lateral);
    const auto audit=verification::auditExactTriangleBoundary(m.vertices,m.faces,
        verification::ExactBoundaryOrientation::OutwardPositive);
    if (!audit.facts().boundaryValid) throw std::domain_error("finite graph is not an exact outward boundary");
    out.auditHash=audit.facts().inputDigest;
    std::ostringstream id;
    id.imbue(std::locale::classic());
    id<<version<<';'<<sideMethod<<';'<<out.sign<<';'<<out.doublings<<';'
      <<triangulationHash(out.mesh,true)<<';'<<out.auditHash;
    out.mesh.hash=manufacturing::computeContentSha256(id.str());
}

Eigen::Matrix3d directionMatrix(const Polynomial::Vector& d) {
    Eigen::Matrix3d a;
    a<<d[4],d[9],d[8], d[9],d[5],d[7], d[8],d[7],d[6];
    return a;
}

void motion(const Polynomial& rawUnit, const Polynomial& q, const Chart& chart, const Grid& g, Result& r) {
    if (!r.derivativeDiagnostic.empty()) return;
    const auto tangent=spin::evaluateQuadricStratumTangent(rawUnit);
    const bool plane=r.mesh.type==spin::QuadricSurfaceType::Plane;
    if (!tangent.defined || tangent.tangentDimension!=(plane?3:8))
        throw std::domain_error("native normalized coefficient tangent mismatch");
    r.basis=tangent.basis;
    const int count=g.points.size(), dim=r.basis.cols();
    r.flatVelocity.resize(6*count,dim);
    for (int k=0;k<dim;++k) {
        const Polynomial::Vector dq=r.sign*r.basis.col(k);
        const V db=dq.segment<3>(1);
        const Eigen::Matrix3d da=directionMatrix(dq);
        V du,dv,dn;
        if (plane) {
            dn=(db-r.axis*r.axis.dot(db))/q.linearVector().norm();
            const V reference=V::Unit(r.referenceAxis);
            const V change=dn.cross(reference);
            du=(change-r.first*r.first.dot(change))/r.axis.cross(reference).norm();
            dv=dn.cross(r.first)+r.axis.cross(du);
        } else {
            const auto de=[&](int i) {
                V d=V::Zero();
                for (int j=0;j<3;++j) if (j!=i)
                    d+=chart.eigenvectors.col(j)*
                        (chart.eigenvectors.col(j).dot(da*chart.eigenvectors.col(i))/(r.eigenvalues[i]-r.eigenvalues[j]));
                return d;
            };
            du=de(chart.u); dv=de(chart.v); dn=chart.nullSign*de(r.nullIndex);
        }
        for (int i=0;i<count;++i) {
            const V& p=g.points[i];
            const auto [s,t,w]=g.coordinates[i];
            const V fixed=s*du+t*dv+w*dn, grad=q.gradient(p);
            const double slope=grad.dot(r.axis);
            if (!std::isfinite(slope) || slope<=1.e-12) {
                r.derivativeDiagnostic="implicit axial root is ill-conditioned for motion";
                r.basis.resize(0,0); r.flatVelocity.resize(0,0); return;
            }
            const double dw=-(dq[0]+db.dot(p)+p.dot(da*p)+grad.dot(fixed))/slope;
            const V lower=fixed+r.axis*dw;
            r.flatVelocity.block<3,1>(3*i,k)=lower;
            r.flatVelocity.block<3,1>(3*(i+count),k)=lower+r.depth*dn;
        }
    }
    if (!r.flatVelocity.allFinite()) throw std::domain_error("nonfinite reduced source motion");
    std::ostringstream id;
    id.imbue(std::locale::classic());
    id<<version<<";tangent-v1;"<<rawUnit.hash()<<';'<<r.mesh.hash<<';'<<std::hexfloat;
    for (const auto* array:{&r.basis,&r.flatVelocity})
        for (int i=0;i<array->size();++i) id<<array->data()[i]<<';';
    r.motionHash=manufacturing::computeContentSha256(id.str());
}
}

Result construct(const Polynomial& rawUnit,const Eigen::MatrixXd& domain,const Settings& settings,bool withMotion) {
    if (!settings.algebraicFiniteMap || settings.fixedNormalParaboloid || settings.rationalSourceCoordinates ||
        settings.compactRationalSourceCoordinates || settings.rankTwoArrangementSource)
        throw std::invalid_argument("R3/R8 extension requires the binary64 algebraic finite-map profile");
    if (!std::isfinite(settings.truncationRadius) || settings.truncationRadius<0 ||
        settings.maximumRadiusDoublings<0 || settings.maximumRadiusDoublings>64 ||
        settings.paraboloidGridResolution<1 || settings.paraboloidGridResolution>512)
        throw std::invalid_argument("invalid finite rank-chart triangulation settings");
    Result out;
    const double automatic=automaticRadius(domain);
    out.radius=settings.truncationRadius==0 ? automatic : settings.truncationRadius;
    if (!std::isfinite(out.radius) || !std::isfinite(4*out.radius))
        throw std::domain_error("finite rank-chart radius overflow");
    const auto& coeff=rawUnit.values();
    coeff.cwiseAbs().maxCoeff(&out.signPivot);
    out.sign=coeff[out.signPivot]<0 ? -1 : 1;
    double second=0;
    for (int i=0;i<10;++i) if (i!=out.signPivot) second=std::max(second,std::abs(coeff[i]));
    out.signGap=std::abs(coeff[out.signPivot])-second;
    if (out.signGap<=1.e-10)
        for (int i=0;i<10;++i) if (coeff[i]*out.sign<0 && std::abs(coeff[out.signPivot])-std::abs(coeff[i])<=1.e-10)
            out.derivativeDiagnostic="canonical coefficient sign-pivot seam";
    const Polynomial q=Polynomial::fromNormalized(out.sign*coeff);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen;
    const auto type=classify(q,&eigen);
    Chart chart;
    selectFrame(q,type,eigen,chart,out);
    const Graph graph(q,out);
    const Grid grid=makeGrid(graph,out,type==spin::QuadricSurfaceType::Plane,settings.paraboloidGridResolution);
    closeGraph(graph,grid,domain,type,settings,out);
    if (withMotion) motion(rawUnit,q,chart,grid,out);
    return out;
}
}
