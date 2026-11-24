#include "stratum_api.h"
#include "spin/QuadricMomentEvaluator.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
spin::QuadricTriangulationSettings readSettings(const py::object& s) {
    spin::QuadricTriangulationSettings result;
    // Accept the deployed settings object without registering its C++ type a
    // second time. The deployment exposes the ordinary, binary64 source profile.
    result.algebraicFiniteMap = s.attr("algebraic_finite_map").cast<bool>();
    result.fixedNormalParaboloid = s.attr("fixed_normal_paraboloid").cast<bool>();
    result.paperProjectiveHyperboloid = s.attr("paper_projective_hyperboloid").cast<bool>();
    result.rationalSourceCoordinates = s.attr("rational_source_coordinates").cast<bool>();
    result.compactRationalSourceCoordinates = s.attr("compact_rational_source_coordinates").cast<bool>();
    result.icosphereSubdivisions = s.attr("icosphere_subdivisions").cast<int>();
    result.maximumIcosphereSubdivisions = s.attr("maximum_icosphere_subdivisions").cast<int>();
    result.maximumRadiusDoublings = s.attr("maximum_radius_doublings").cast<int>();
    result.truncationRadius = s.attr("truncation_radius").cast<double>();
    result.paraboloidGridResolution = s.attr("paraboloid_grid_resolution").cast<int>();
    result.retainFixedNormalProvenance = s.attr("retain_fixed_normal_provenance").cast<bool>();
    return result;
}

py::dict packet(const spin::QuadricCoefficients10::Vector& raw, const Eigen::MatrixXd& design,
                const py::object& options, bool withMotion) {
    const auto settings = readSettings(options);
    const auto q = spin::QuadricCoefficients10::fromUnnormalized(raw);
    const auto result = compact_stratum::construct(q, design, settings, withMotion);
    const auto& mesh = result.mesh;
    py::dict out, witness;
    witness["unit_coefficients"] = q.values();
    witness["coefficient_sign"] = result.sign;
    witness["coefficient_pivot"] = result.signPivot;
    witness["coefficient_pivot_gap"] = result.signGap;
    witness["axis"] = result.axis;
    witness["first"] = result.first;
    witness["second"] = result.second;
    witness["frame_margin"] = result.frameMargin;
    witness["frame_reference_axis"] = result.referenceAxis;
    witness["canonical_quadratic_eigenvalues"] = result.eigenvalues;
    witness["eigen_residual_max_abs"] = result.eigenResidual;
    witness["eigen_orthogonality_max_abs"] = result.eigenOrthogonality;
    witness["eigen_verification_bound"] = 0x1p-42;
    witness["least_eigenvalue_index"] = result.nullIndex;
    witness["paraboloid_grid_resolution"] = settings.paraboloidGridResolution;
    witness["initial_radius"] = result.radius;
    witness["cap_depth"] = result.depth;
    witness["cap_depth_doublings"] = result.doublings;
    witness["maximum_radius_doublings"] = settings.maximumRadiusDoublings;
    witness["far_support_gap"] = result.farGap;
    witness["previous_far_support_gap"] = result.rejectedFarGap;
    witness["lateral_support_gap"] = result.lateralGap;
    witness["minimum_interface_root_slope"] = result.minimumRootSlope;
    witness["target_boundary_audit_digest"] = result.auditHash;
    out["defined"] = !withMotion || result.derivativeDiagnostic.empty();
    out["diagnostic"] = result.derivativeDiagnostic;
    out["algebraic_finite_map"] = true;
    out["source_map_version"] = compact_stratum::version;
    out["surface_type"] = spin::quadricSurfaceTypeToken(mesh.type);
    out["vertices"] = mesh.closedBoundary.vertices;
    out["faces"] = mesh.closedBoundary.faces;
    out["face_is_cap"] = mesh.faceIsCap;
    out["interface_vertex_count"] = mesh.interfaceVertexCount;
    out["interface_face_count"] = mesh.interfaceFaceCount;
    out["cap_face_count"] = mesh.capFaceCount;
    out["truncation_radius"] = mesh.truncationRadius;
    out["triangulation_hash"] = mesh.hash;
    out["side_selection_defined"] = true;
    out["side_selection_diagnostic"] = "";
    out["side_selection_method"] = compact_stratum::sideMethod;
    out["interior_is_positive"] = result.sign > 0;
    out["construction_witness"] = witness;
    if (withMotion && result.derivativeDiagnostic.empty()) {
        out["derivative_scope"] = "REDUCED_FIXED_STRATUM_NOT_FULL_EQ14";
        out["scientific_hash"] = result.motionHash;
        out["coefficient_basis"] = result.basis;
        out["vertex_velocity"] = result.flatVelocity;
    }
    return out;
}

py::dict fullRank(const spin::QuadricCoefficients10::Vector& raw, const Eigen::MatrixXd& design,
                  const py::object& options) {
    const auto q = spin::QuadricCoefficients10::fromUnnormalized(raw);
    const auto m = spin::evaluateQuadricSourceMotion(q, design, readSettings(options));
    py::dict out;
    out["defined"] = m.defined;
    out["diagnostic"] = m.diagnostic;
    if (m.defined) {
        const auto& t = m.triangulation;
        out["vertices"] = t.closedBoundary.vertices;
        out["faces"] = t.closedBoundary.faces;
        out["face_is_cap"] = t.faceIsCap;
        out["triangulation_hash"] = t.hash;
        out["scientific_hash"] = m.scientificHash;
        out["interior_is_positive"] = spin::quadricSolidInteriorIsPositiveRegion(q,t);
        Eigen::MatrixXd velocity(3*m.velocity.size(),10);
        for (size_t i=0; i<m.velocity.size(); ++i) velocity.block<3,10>(3*i,0) = m.velocity[i];
        out["vertex_velocity"] = velocity;
    }
    return out;
}
}

PYBIND11_MODULE(_compact_stratum_v1, module) {
    module.attr("source_map_version") = compact_stratum::version;
    module.def("source_value", [](const spin::QuadricCoefficients10::Vector& q, const Eigen::MatrixXd& design,
                                 const py::object& settings) { return packet(q, design, settings, false); });
    module.def("source_motion", [](const spin::QuadricCoefficients10::Vector& q, const Eigen::MatrixXd& design,
                                  const py::object& settings) { return packet(q, design, settings, true); });
    module.def("full_rank_control", &fullRank);
}
