// A narrow C ABI around the repository's UNMODIFIED numerical solver.
// Including its translation unit is intentional: the M-only diagnostic needs
// its private evaluateState(), and copying that formula would create a second
// acceptance implementation. Hidden symbols and section GC omit geometry and
// export entry points; this library does not certify or perform mesh Booleans.
#include "../../../src/spin/QuadricHollowingSolver.cpp"
#include <cstring>

#define CPINO_EXPORT extern "C" __attribute__((visibility("default")))

struct CpinoMapPacket {
    double moments[10];
    double jacobian[100]; // row-major dM/d(native, unit, geometry coefficients)
    int derivative_defined;
    char geometry_hash[129];
    char topology_hash[129];
    char diagnostic[1024];
    char evaluation_map_identity[129];
    char boolean_backend_identity[513];
    char derivative_identity[129];
};
using CpinoEvaluator = int (*)(const double*, CpinoMapPacket*, void*);

namespace cpino_bridge {
using V10 = Eigen::Matrix<double, 10, 1>;
using Row10 = Eigen::Matrix<double, 10, 10, Eigen::RowMajor>;
thread_local std::string response;

std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u00" << std::hex << std::setw(2)
                            << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}

template<class T> void vector(std::ostream& out, const T& values) {
    out << '[';
    for (Eigen::Index i=0; i<values.size(); ++i) {
        if (i) out << ',';
        if (std::isfinite(values(i))) out << values(i); else out << "null";
    }
    out << ']';
}
void metric(std::ostream& out, const std::optional<double>& value) {
    if (value && std::isfinite(*value)) out << *value; else out << "null";
}

spin::SpinMomentProblem problem(int mode, double weight, double massScale) {
    if (!std::isfinite(weight) || !std::isfinite(massScale) || massScale <= 0)
        throw std::invalid_argument("nonfinite objective setting or nonpositive mass scale");
    switch (mode) {
        case 0: return spin::SpinMomentProblem::staticBalance(spin::StaticObjective::MaximumMass, massScale);
        case 1: return spin::SpinMomentProblem::staticBalance(spin::StaticObjective::LowerCenterOfMass, massScale);
        case 2: return spin::SpinMomentProblem::staticBalancePareto(weight, massScale);
        case 3: return spin::SpinMomentProblem::fidgetSpinner();
        case 4: return spin::SpinMomentProblem::yoYo();
        case 5: return spin::SpinMomentProblem::spinningTop(weight);
        case 6: return spin::SpinMomentProblem::spinningTop(weight, spin::SpinningTopObjective::LowerCenterOfMass);
        default: throw std::invalid_argument("unsupported repository objective mode");
    }
}

spin::SpinSolverSettings settings(double equation, double balance, int budget) {
    spin::SpinSolverSettings s;
    s.equationProfile = spin::SpinEquationProfile::DualMonomialRawV1;
    s.maximumIterations = 1;
    s.maximumRestarts = 1;
    s.maximumFunctionEvaluationsPerRestart = budget;
    s.equationTolerance = equation;
    s.physicalRelativeTolerance = balance;
    s.captureEvaluationPayload = true;
    validateSettings(s);
    return s;
}

spin::SpinMomentMapEvaluation material(const CpinoMapPacket& packet) {
    spin::SpinMomentMapEvaluation value;
    value.rawMoments = Eigen::Map<const V10>(packet.moments);
    value.momentJacobian = Eigen::Map<const Row10>(packet.jacobian);
    value.derivativeDefined = packet.derivative_defined == 1;
    value.geometryHash = packet.geometry_hash;
    value.topologyHash = packet.topology_hash;
    value.derivativeDiagnostic = packet.diagnostic;
    value.evaluationMapIdentity = packet.evaluation_map_identity;
    value.booleanBackendIdentity = packet.boolean_backend_identity;
    value.derivativeIdentity = packet.derivative_identity;
    return value;
}

const char* error(const std::exception& exc) {
    response = "{\"error\":" + quote(exc.what()) + '}';
    return response.c_str();
}
std::ostringstream stream() {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17) << std::boolalpha;
    return out;
}
}

CPINO_EXPORT const char* cpino_native_identity() {
    return "TVCG_REPO_DUAL_MONOMIAL_RAW_V1_CABI_V1";
}

CPINO_EXPORT const char* cpino_native_accept(
    int mode, double weight, double massScale, const double* c,
    const double* moments, double equation, double balance) {
    using namespace cpino_bridge;
    try {
        const auto q = spin::QuadricCoefficients10::fromNormalized(Eigen::Map<const V10>(c));
        const auto p = problem(mode, weight, massScale);
        const auto s = settings(equation, balance, 60);
        spin::SpinMomentMapEvaluation map;
        map.rawMoments = Eigen::Map<const V10>(moments);
        // evaluateState requires the flag even though its residual and native
        // multiplier solve never read J. This local zero is NOT a derivative
        // label, never leaves this function and is never sent to Newton.
        map.derivativeDefined = true;
        const auto state = evaluateState(p, q, map, s.equationProfile);
        auto out = stream();
        out << "{\"stationarity_residual\":" << state.rawResidual
            << ",\"balance_residual\":" << state.physicalResidual
            << ",\"accepted\":" << converged(state, s)
            << ",\"objective\":" << state.problem.objective
            << ",\"normalized_equation_residual\":" << state.normalizedResidual
            << ",\"multiplier_mu\":" << state.mu << ",\"multipliers\":";
        vector(out, state.lambda);
        out << ",\"constraints\":"; vector(out, state.problem.constraints);
        out << ",\"optimality_vector\":"; vector(out, state.optimalityResidual);
        out << ",\"acceptance_reads_jacobian\":false,\"multiplier_policy\":\"native_recompute\"}";
        response=out.str(); return response.c_str();
    } catch (const std::exception& exc) { return error(exc); }
}

CPINO_EXPORT const char* cpino_native_step(
    int mode, double weight, double massScale, const double* c,
    const CpinoMapPacket* center, double equation, double balance,
    int budget, CpinoEvaluator evaluate, void* context) {
    using namespace cpino_bridge;
    try {
        if (!evaluate || !center || center->derivative_defined != 1)
            throw std::invalid_argument("one-step requires qualified center and actual trial evaluator");
        const auto q = spin::QuadricCoefficients10::fromNormalized(Eigen::Map<const V10>(c));
        const auto p = problem(mode, weight, massScale);
        const auto s = settings(equation, balance, budget);
        int calls=0;
        const auto evaluator = [&](const spin::QuadricCoefficients10& trial) {
            if (calls++ == 0) {
                if (!(trial.values().array() == q.values().array()).all())
                    throw std::runtime_error("native solver changed saved initial coefficients");
                return material(*center);
            }
            CpinoMapPacket packet{};
            const int code=evaluate(trial.values().data(), &packet, context);
            if (code != 0)
                throw std::runtime_error(std::string("native trial evaluator rejected: ")+packet.diagnostic);
            return material(packet);
        };
        // The complete existing Eq.14 system, native solve, tangent projection,
        // geodesic update, fresh trial evaluation and line search run here.
        const auto result=spin::solveSpinMomentSystem(p, s, evaluator, {q});
        std::optional<V10> dualDirection;
        const spin::SpinSolverEvent* acceptedTrial=nullptr;
        const spin::SpinSolverEvent* terminal=nullptr;
        for (const auto& event : result.events) {
            if (event.newtonSystemPayload && !dualDirection) {
                const auto& payload=*event.newtonSystemPayload;
                // Native events store the exact system, not the solution.
                // Recover its direction with the SAME native linear solver;
                // never invert the material Jacobian or invent a Newton map.
                const auto linear=spin::solveSpinNewtonLinearSystem(payload.matrix, payload.rightHandSide);
                if (linear.solved)
                    dualDirection=spin::DualMonomialCoordinates10::fromGeometry(q)
                        .projectToTangent(linear.solution.head<10>());
            }
            if (event.type=="evaluation" && event.iteration==1 && event.outcome=="accepted")
                acceptedTrial=&event;
            if (event.type=="attempt_terminal") terminal=&event;
        }
        auto out=stream();
        out << "{\"direction\":";
        if (dualDirection) {
            const V10 raw=spin::DualMonomialCoordinates10::fromGeometry(q).geometryJacobian()*(*dualDirection);
            vector(out,raw);
        } else out << "null";
        out << ",\"native_dual_direction\":";
        if (dualDirection) vector(out,*dualDirection); else out << "null";
        out << ",\"direction_origin\":\"native_event_bordered_system_replay\""
            << ",\"direction_coordinates\":\"raw_unit_geometry_tangent\""
            << ",\"line_search_accepted\":" << (acceptedTrial!=nullptr)
            << ",\"reason\":" << quote(acceptedTrial ? acceptedTrial->reason : terminal ? terminal->reason : result.diagnostic)
            << ",\"trial_coefficients\":";
        if (acceptedTrial && acceptedTrial->quadricCoefficients) {
            const auto& values=*acceptedTrial->quadricCoefficients;
            vector(out,Eigen::Map<const V10>(values.data()));
        } else out << "null";
        out << ",\"trial_native_stationarity_residual\":";
        metric(out,acceptedTrial ? acceptedTrial->rawEquationResidual : std::optional<double>());
        out << ",\"trial_native_balance_residual\":";
        metric(out,acceptedTrial ? acceptedTrial->physicalRelativeResidual : std::optional<double>());
        out << ",\"function_evaluations\":" << result.functionEvaluations
            << ",\"trial_evaluations\":" << std::max(0,calls-1)
            << ",\"solver_status\":" << quote(spin::spinSolverStatusToken(result.status))
            << ",\"solver_diagnostic\":" << quote(result.diagnostic)
            << ",\"scientific_event_sha256\":" << quote(spin::spinSolverScientificEventHash(result))
            << ",\"trace_sha256\":" << quote(result.traceHash)
            << ",\"events\":[";
        bool first=true;
        for (const auto& event:result.events) {
            if (!first) out << ','; first=false;
            out << "{\"type\":" << quote(event.type)
                << ",\"iteration\":" << event.iteration
                << ",\"evaluation_index\":" << event.evaluationIndex
                << ",\"outcome\":" << quote(event.outcome)
                << ",\"reason\":" << quote(event.reason)
                << ",\"detail\":" << quote(event.detail) << ",\"step\":";
            metric(out,event.step);
            out << ",\"stationarity_residual\":";metric(out,event.rawEquationResidual);
            out << ",\"balance_residual\":";metric(out,event.physicalRelativeResidual);
            out << ",\"multiplier_mu\":";metric(out,event.multiplierMu);
            out << ",\"coefficients\":";
            if (event.quadricCoefficients) vector(out,Eigen::Map<const V10>(event.quadricCoefficients->data()));
            else out << "null";
            if (event.evaluationPayload) {
                const auto& payload=*event.evaluationPayload;
                out << ",\"evaluation_map_identity\":" << quote(payload.evaluationMapIdentity)
                    << ",\"boolean_backend_identity\":" << quote(payload.booleanBackendIdentity)
                    << ",\"derivative_identity\":" << quote(payload.derivativeIdentity)
                    << ",\"geometry_hash\":" << quote(payload.geometryHash)
                    << ",\"topology_hash\":" << quote(payload.topologyHash);
            }
            out << '}';
        }
        out << "]}";
        response=out.str();return response.c_str();
    } catch (const std::exception& exc) { return error(exc); }
}
