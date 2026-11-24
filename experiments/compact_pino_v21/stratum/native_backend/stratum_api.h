#pragma once
#include "spin/QuadricSourceMotion.h"

namespace compact_stratum {
inline constexpr const char *version = "ALGEBRAIC_FINITE_GRAPH_EXTRUSION_R3_R8_V1";
inline constexpr const char *sideMethod = "canonical-graph-positive-extrusion-exact-oriented-boundary-v1";

// The construction witness is descriptive. A point-local source audit does
// not certify an event-free coefficient neighborhood or a material Jacobian.
struct Result {
    spin::QuadricTriangulation mesh;
    Eigen::Vector3d axis, first, second, eigenvalues = Eigen::Vector3d::Zero();
    Eigen::MatrixXd basis, flatVelocity;
    int sign = 0, signPivot = -1, referenceAxis = -1, nullIndex = -1, doublings = 0;
    double signGap = 0, radius = 0, depth = 0, farGap = 0, lateralGap = 0;
    double frameMargin = 0, minimumRootSlope = 0, rejectedFarGap = 0;
    double eigenResidual = 0, eigenOrthogonality = 0;
    std::string auditHash, motionHash, derivativeDiagnostic;
};

Result construct(const spin::QuadricCoefficients10&, const Eigen::MatrixXd& domain,
                 const spin::QuadricTriangulationSettings&, bool withMotion);
}
