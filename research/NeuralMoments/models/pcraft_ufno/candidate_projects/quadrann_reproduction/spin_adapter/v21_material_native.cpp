// One-time construction for v21_material.py. Unit grid coordinates keep clipping
// and the degree-two moment fit local to a single native voxel.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using P = std::array<double, 3>;
using Node = std::array<double, 4>;
using M = std::array<double, 10>;
using Row = std::array<int32_t, 4>;
using Polygon = std::vector<P>;
constexpr double eps = 64 * std::numeric_limits<double>::epsilon();
constexpr double g2 = 0.28867513459481288225;

P blend(const P &a, const P &b, double t) {
    return {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])};
}
Polygon clip(const Polygon &poly, int axis, double face, bool above) {
    Polygon out;
    if (poly.empty())
        return out;
    out.reserve(poly.size() + 1);
    P a = poly.back();
    for (const P &b : poly) {
        double u = a[axis] - face, v = b[axis] - face;
        if ((u < 0 && v > 0) || (u > 0 && v < 0)) {
            P p = blend(a, b, u / (u - v));
            p[axis] = face;
            out.push_back(p);
        }
        if (above ? v >= 0 : v <= 0)
            out.push_back(b);
        a = b;
    }
    return out;
}
M basis(double x, double y, double z) {
    return {1, x, y, z, x * x, y * y, z * z, y * z, x * z, x * y};
}
M node_moments(const std::vector<Node> &nodes) {
    M m{};
    for (const Node &p : nodes) {
        const M b = basis(p[0], p[1], p[2]);
        for (int j = 0; j < 10; ++j)
            m[j] += p[3] * b[j];
    }
    return m;
}
// 4-point cubic triangle rule (only the offline oriented flux may be signed).
// z is integrated analytically from the cell floor to the triangle's affine plane.
void face_flux(const Polygon &poly, int k, M &ramp, M &full) {
    constexpr double coeff[4][4] = {{1. / 3, 1. / 3, 1. / 3, -27. / 48},
                                    {.6, .2, .2, 25. / 48},
                                    {.2, .6, .2, 25. / 48},
                                    {.2, .2, .6, 25. / 48}};
    for (size_t t = 1; t + 1 < poly.size(); ++t) {
        const P &a = poly[0], &b = poly[t], &c = poly[t + 1];
        double area = .5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
        if (area == 0)
            continue;
        for (const auto &q : coeff) {
            double x = q[0] * a[0] + q[1] * b[0] + q[2] * c[0];
            double y = q[0] * a[1] + q[1] * b[1] + q[2] * c[1];
            double h = q[0] * (a[2] - k) + q[1] * (b[2] - k) + q[2] * (c[2] - k), w = area * q[3];
            double wh = w * h, z = wh * h * .5;
            ramp[0] += wh;
            ramp[1] += wh * x;
            ramp[2] += wh * y;
            ramp[3] += z;
            ramp[4] += wh * x * x;
            ramp[5] += wh * y * y;
            ramp[6] += z * h * (2. / 3);
            ramp[7] += z * y;
            ramp[8] += z * x;
            ramp[9] += wh * x * y;
            full[0] += w;
            full[1] += w * x;
            full[2] += w * y;
            full[3] += w * .5;
            full[4] += w * x * x;
            full[5] += w * y * y;
            full[6] += w / 3;
            full[7] += w * y * .5;
            full[8] += w * x * .5;
            full[9] += w * x * y;
        }
    }
}
struct Face {
    P a, b, c;
    double det, minx, maxx, miny, maxy;
    bool covers(double x, double y) const {
        if (det == 0 || x < minx || x > maxx || y < miny || y > maxy)
            return false;
        const P *pv[3] = {&a, &b, &c};
        for (int j = 0; j < 3; ++j) {
            const P &u = *pv[j], &v = *pv[(j + 1) % 3];
            // Both incident triangles must evaluate a common indexed edge with
            // the *same* anchor before applying opposite orientations. Independent
            // reversed cross-products can each round to the outside of the edge.
            bool forward = u[0] < v[0] || (u[0] == v[0] && u[1] < v[1]);
            const P &a = forward ? u : v, &b = forward ? v : u;
            double dx = b[0] - a[0], dy = b[1] - a[1];
            double left = dx * (y - a[1]), right = dy * (x - a[0]), side = left - right;
            if (std::abs(side) < eps * (std::abs(left) + std::abs(right))) {
                long double ax = a[0], ay = a[1];
                side = double(
                    (static_cast<long double>(b[0]) - ax) * (static_cast<long double>(y) - ay) -
                    (static_cast<long double>(b[1]) - ay) * (static_cast<long double>(x) - ax));
            }
            if (forward != (det > 0)) {
                side = -side;
                dx = -dx;
                dy = -dy;
            }
            if (side < 0 || (side == 0 && !(dy > 0 || (dy == 0 && dx < 0))))
                return false;
        }
        return true;
    }
    double height(double x, double y) const {
        double u = ((x - a[0]) * (c[1] - a[1]) - (y - a[1]) * (c[0] - a[0])) / det;
        double v = ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])) / det;
        return a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]);
    }
};
Face make_face(P a, P b, P c) {
    return {a,
            b,
            c,
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
            std::min({a[0], b[0], c[0]}),
            std::max({a[0], b[0], c[0]}),
            std::min({a[1], b[1], c[1]}),
            std::max({a[1], b[1], c[1]})};
}
bool inside(const std::vector<Face> &faces, const P &p, int k) {
    if (!(p[0] > 0 && p[0] < 1 && p[1] > 0 && p[1] < 1 && p[2] > 0 && p[2] < 1))
        return false;
    int depth = 0;
    for (const Face &f : faces)
        if (f.covers(p[0], p[1]) && f.height(p[0], p[1]) > p[2] + k)
            depth += (f.det > 0 ? 1 : -1);
    return depth == 1;
}
using Rule = std::array<P, 4>;
std::vector<Rule> rotations() {
    const Rule tetra = {P{1, 1, 1}, P{1, -1, -1}, P{-1, 1, -1}, P{-1, -1, 1}};
    std::vector<Rule> rules;
    for (int j = 0; j < 15; ++j) {
        double u = std::fmod(j * .61803398874989485, 1.), v = j * 2.3999632297286533;
        P axis = {std::sqrt(1 - u * u) * std::cos(v), std::sqrt(1 - u * u) * std::sin(v), u};
        double theta = j * 1.324717957244746, cs = std::cos(theta), sn = std::sin(theta);
        Rule rule{};
        for (int n = 0; n < 4; ++n) {
            auto p = tetra[n];
            double dot = p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
            for (int i = 0; i < 3; ++i)
                rule[n][i] =
                    cs * p[i] +
                    sn * (axis[(i + 1) % 3] * p[(i + 2) % 3] - axis[(i + 2) % 3] * p[(i + 1) % 3]) +
                    (1 - cs) * dot * axis[i];
        }
        rules.push_back(rule);
        for (auto &p : rule)
            for (double &a : p)
                a = -a;
        rules.push_back(rule);
    }
    return rules;
}
// A tetrahedral sign rule realizes any SPD covariance. Retain it only when all
// nodes are strictly inside the voxel and the original material. Some solids
// admit no such four-point rule, e.g. separated islands; those take the sweep.
bool fit_supported_rule(const M &m, int k, const std::vector<Face> &faces,
                        const std::vector<Rule> &rules, std::vector<Node> &out) {
    if (m[0] < 1e-11 || m[0] > 1 + 1e-12)
        return false;
    P mean = {m[1] / m[0], m[2] / m[0], m[3] / m[0]};
    double C[3][3] = {{m[4] / m[0], m[9] / m[0], m[8] / m[0]},
                      {m[9] / m[0], m[5] / m[0], m[7] / m[0]},
                      {m[8] / m[0], m[7] / m[0], m[6] / m[0]}};
    double L[3][3]{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) {
            double t = C[i][j] - mean[i] * mean[j];
            for (int a = 0; a < j; ++a)
                t -= L[i][a] * L[j][a];
            if (i == j) {
                if (!(t > 0))
                    return false;
                L[i][j] = std::sqrt(t);
            } else
                L[i][j] = t / L[j][j];
        }
    for (const auto &rule : rules) {
        Rule points{};
        bool ok = true;
        for (int n = 0; n < 4; ++n) {
            P p = mean;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j <= i; ++j)
                    p[i] += L[i][j] * rule[n][j];
            if (!inside(faces, p, k)) {
                ok = false;
                break;
            }
            points[n] = p;
        }
        if (ok) {
            for (const auto &p : points)
                out.push_back({p[0], p[1], p[2], m[0] * .25});
            return true;
        }
    }
    return false;
}
struct Segment {
    double x0, y0, x1, y1;
    double y(double x) const {
        return y0 + (x - x0) / (x1 - x0) * (y1 - y0);
    }
};
void segments_of(const Polygon &p, std::vector<Segment> &segs, std::vector<double> &xs) {
    if (p.size() < 2)
        return;
    P a = p.back();
    for (const P &b : p) {
        xs.push_back(b[0]);
        if (a[0] != b[0] && !(a[1] == b[1] && (a[1] == 0 || a[1] == 1))) {
            if (a[0] < b[0])
                segs.push_back({a[0], a[1], b[0], b[1]});
            else
                segs.push_back({b[0], b[1], a[0], a[1]});
        }
        a = b;
    }
}
struct Hit {
    double h;
    const Face *face;
};
struct Layer {
    const Face *lower, *upper;
};
std::vector<Layer> layers(const std::vector<Face> &faces, double x, double y, int k) {
    std::vector<Hit> hits;
    for (const Face &f : faces)
        if (f.covers(x, y))
            hits.push_back({f.height(x, y), &f});
    std::sort(hits.begin(), hits.end(), [](const Hit &a, const Hit &b) { return a.h < b.h; });
    std::vector<Layer> out;
    int depth = 0;
    Hit lower{};
    for (auto h : hits) {
        int before = depth;
        depth += (h.face->det > 0 ? -1 : 1);
        if (before == 0 && depth == 1)
            lower = h;
        if (before == 1 && depth == 0 && h.h > k && lower.h < k + 1 && h.h > lower.h)
            out.push_back({lower.h > k ? lower.face : nullptr, h.h < k + 1 ? h.face : nullptr});
    }
    if (depth != 0) {
        char diagnostic[256];
        std::snprintf(diagnostic, sizeof(diagnostic),
                      "unclosed vertical winding %d at projected %.17g %.17g in slab %d (%zu hits, "
                      "%zu column faces)",
                      depth, x, y, k, hits.size(), faces.size());
        throw std::runtime_error(diagnostic);
    }
    return out;
}
// Within each x-slab, projected segment order is constant. The y bounds are
// affine in x, and each layer is bounded by two affine z planes. After exact
// z integration of a quadratic the degree in y is <=3, and after integration
// over the affine y limits the degree in x is <=4. GL(3,2,2) is positive.
std::vector<Node> sweep(int k, const std::vector<Polygon> &faces_in_cell,
                        const std::vector<Face> &faces) {
    std::vector<Segment> segs = {{0, 0, 1, 0}, {0, 1, 1, 1}};
    std::vector<double> xs = {0, 1};
    for (const auto &p : faces_in_cell)
        segments_of(p, segs, xs);
    for (size_t j = 0; j < segs.size(); ++j)
        for (size_t i = 0; i < j; ++i) {
            const auto &a = segs[i], &b = segs[j];
            double l = std::max(a.x0, b.x0), r = std::min(a.x1, b.x1);
            if (!(r > l))
                continue;
            double d = a.y(l) - b.y(l), e = a.y(r) - b.y(r);
            if ((d > 0 && e < 0) || (d < 0 && e > 0))
                xs.push_back(l + (r - l) * d / (d - e));
        }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end(),
                         [](double a, double b) { return std::abs(a - b) <= eps; }),
             xs.end());
    constexpr double ux[3] = {.11270166537925831148, .5, .88729833462074168852};
    constexpr double wx[3] = {5. / 18, 8. / 18, 5. / 18};
    std::vector<Node> nodes;
    for (size_t n = 1; n < xs.size(); ++n) {
        double l = xs[n - 1], r = xs[n], xc = (r + l) * .5;
        std::vector<std::pair<double, const Segment *>> active;
        for (const auto &s : segs)
            if (s.x0 < xc && xc < s.x1)
                active.emplace_back(s.y(xc), &s);
        std::sort(active.begin(), active.end(), [](auto a, auto b) { return a.first < b.first; });
        for (size_t i = 1; i < active.size(); ++i) {
            if (active[i].first - active[i - 1].first <= eps)
                continue;
            const auto band = layers(faces, xc, (active[i].first + active[i - 1].first) * .5, k);
            for (const auto &p : band)
                for (int j = 0; j < 3; ++j) {
                    double x = l + ux[j] * (r - l), ymin = active[i - 1].second->y(x),
                           ymax = active[i].second->y(x);
                    for (double s : {.5 - g2, .5 + g2}) {
                        double y = ymin + s * (ymax - ymin);
                        double zlo = p.lower ? std::clamp(p.lower->height(x, y) - k, 0., 1.) : 0.;
                        double zhi = p.upper ? std::clamp(p.upper->height(x, y) - k, 0., 1.) : 1.;
                        double w = (r - l) * wx[j] * (ymax - ymin) * (zhi - zlo) * .25;
                        if (w > 0)
                            for (double t : {.5 - g2, .5 + g2})
                                nodes.push_back({x, y, zlo + t * (zhi - zlo), w});
                    }
                }
        }
    }
    return nodes;
}

// Preserve the ten local monomials by eliminating a linear dependency among
// eleven supported nodes. Group means allow a large rule to shrink by roughly
// one half per pass. A separate moment check can keep the unreduced rule.
void reduce_groups(const std::vector<M> &means, std::vector<double> &mass) {
    std::vector<int> active(means.size());
    for (size_t j = 0; j < active.size(); ++j)
        active[j] = int(j);
    while (active.size() > 10) {
        double a[10][11]{};
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 11; ++j)
                a[i][j] = means[active[j]][i];
        int piv[10]{}, rank = 0, free_col = -1;
        for (int j = 0; j < 11; ++j) {
            int p = rank;
            for (int i = rank + 1; i < 10; ++i)
                if (std::abs(a[i][j]) > std::abs(a[p][j]))
                    p = i;
            if (rank == 10 || std::abs(a[p][j]) < 1e-13) {
                free_col = j;
                break;
            }
            for (int t = j; t < 11; ++t)
                std::swap(a[rank][t], a[p][t]);
            for (int i = rank + 1; i < 10; ++i) {
                double q = a[i][j] / a[rank][j];
                for (int t = j + 1; t < 11; ++t)
                    a[i][t] -= q * a[rank][t];
            }
            piv[rank++] = j;
        }
        double null[11]{};
        null[free_col] = 1;
        for (int i = rank - 1; i >= 0; --i) {
            int j = piv[i];
            double t = 0;
            for (int a_col = j + 1; a_col < 11; ++a_col)
                t += a[i][a_col] * null[a_col];
            null[j] = -t / a[i][j];
        }
        double step = std::numeric_limits<double>::infinity();
        int drop = 0;
        for (int i = 0; i < 11; ++i)
            if (null[i] > 0 && mass[active[i]] / null[i] < step) {
                step = mass[active[i]] / null[i];
                drop = i;
            }
        for (int i = 0; i < 11; ++i)
            mass[active[i]] = std::max(0., mass[active[i]] - step * null[i]);
        mass[active[drop]] = 0;
        active.erase(active.begin() + drop);
    }
}
std::vector<Node> compress_rule(const std::vector<Node> &raw) {
    auto nodes = raw;
    while (nodes.size() > 10) {
        int ng = int(std::min<size_t>(20, nodes.size()));
        std::vector<M> means(ng);
        std::vector<double> old(ng), mass;
        for (size_t i = 0; i < nodes.size(); ++i) {
            int g = int(i * ng / nodes.size());
            const auto &p = nodes[i];
            M b = basis(p[0] - .5, p[1] - .5, p[2] - .5);
            old[g] += p[3];
            for (int j = 0; j < 10; ++j)
                means[g][j] += p[3] * b[j];
        }
        for (int g = 0; g < ng; ++g)
            for (double &v : means[g])
                v /= old[g];
        mass = old;
        reduce_groups(means, mass);
        std::vector<Node> next;
        for (size_t i = 0; i < nodes.size(); ++i) {
            int g = int(i * ng / nodes.size());
            if (mass[g] > 0) {
                auto p = nodes[i];
                p[3] *= mass[g] / old[g];
                if (p[3] > 0)
                    next.push_back(p);
            }
        }
        nodes = std::move(next);
    }
    M a = node_moments(raw), b = node_moments(nodes);
    for (int j = 0; j < 10; ++j)
        if (std::abs(a[j] - b[j]) > 1e-11 * a[0])
            return raw;
    return nodes;
}

struct Column {
    std::vector<Row> runs, partials;
    std::vector<Node> nodes;
    uint64_t fallback = 0, outside = 0;
    double local_residual = 0;
};
Column integrate_column(int x, int y, int side, const std::vector<int32_t> &indices,
                        const double *v, const int32_t *f, const std::vector<Rule> &rules,
                        int partial_rule) {
    std::vector<Face> faces;
    std::vector<M> ramps(side + 1), full(side + 1);
    std::vector<std::vector<Polygon>> boundaries(side);
    for (int id : indices) {
        Polygon p;
        for (int j = 0; j < 3; ++j) {
            auto a = v + 3 * int64_t(f[3 * int64_t(id) + j]);
            p.push_back({a[0] - x, a[1] - y, a[2]});
        }
        Face face = make_face(p[0], p[1], p[2]);
        for (int d = 0; d < 2; ++d) {
            p = clip(p, d, 0, true);
            p = clip(p, d, 1, false);
        }
        if (p.empty())
            continue;
        if (face.det != 0)
            faces.push_back(face);
        P lo = p[0], hi = lo;
        for (auto &a : p)
            for (int d = 0; d < 3; ++d) {
                lo[d] = std::min(lo[d], a[d]);
                hi[d] = std::max(hi[d], a[d]);
            }
        bool inside_xy = !(lo[0] >= 1 || hi[0] <= 0 || lo[1] >= 1 || hi[1] <= 0);
        int first = std::clamp(int(std::floor(lo[2])), 0, side);
        int last = lo[2] == hi[2] ? first : std::clamp(int(std::ceil(hi[2])) - 1, 0, side);
        for (int k = first; k <= last; ++k) {
            auto band = (first == last ? p : clip(clip(p, 2, k, true), 2, k + 1, false));
            if (band.empty())
                continue;
            if (face.det != 0)
                face_flux(band, k, ramps[k], full[k]);
            if (inside_xy && k < side && !(lo[2] == hi[2] && lo[2] == k))
                boundaries[k].push_back(std::move(band));
        }
    }
    Column out;
    M sum{};
    int run_end = -1;
    for (int k = side - 1; k >= 0; --k) {
        M m = ramps[k];
        for (int j = 0; j < 10; ++j) {
            sum[j] += full[k + 1][j];
            m[j] += sum[j];
        }
        bool bulk = boundaries[k].empty();
        if (bulk && std::abs(m[0] - 1) < 1e-10) {
            if (run_end < 0)
                run_end = k + 1;
            continue;
        }
        if (run_end >= 0) {
            out.runs.push_back({x, y, k + 1, run_end});
            run_end = -1;
        }
        if (bulk && std::abs(m[0]) < 1e-10)
            continue;
        if (bulk)
            throw std::runtime_error("uncut voxel has nonbinary oriented winding");
        std::vector<Node> nodes;
        // 0 retains the original supported fit + compressed fallback. Modes 1
        // and 2 use the exact same geometric sweep, differing only in reduction.
        // Reduction never moves nodes, so all three rules retain material support.
        if (partial_rule != 0 || !fit_supported_rule(m, k, faces, rules, nodes)) {
            nodes = sweep(k, boundaries[k], faces);
            if (partial_rule != 2)
                nodes = compress_rule(nodes);
            ++out.fallback;
        }
        M check = node_moments(nodes);
        for (int j = 0; j < 10; ++j) {
            double error = std::abs(m[j] - check[j]);
            out.local_residual = std::max(out.local_residual, error);
            if (error > 2e-9)
                throw std::runtime_error(
                    "positive cell cubature disagrees with oriented cell flux at (" +
                    std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(k) +
                    "), moment " + std::to_string(j) + ", absolute unit-cell discrepancy " +
                    std::to_string(error));
        }
        if (nodes.empty())
            continue;
        if (!inside(faces, {.5, .5, .5}, k))
            ++out.outside;
        out.partials.push_back({x, y, k, int32_t(nodes.size())});
        for (auto p : nodes) {
            p[0] += x;
            p[1] += y;
            p[2] += k;
            out.nodes.push_back(p);
        }
    }
    if (run_end >= 0)
        out.runs.push_back({x, y, 0, run_end});
    return out;
}
template <class T> void write_array(std::ofstream &out, const std::vector<T> &data) {
    out.write(reinterpret_cast<const char *>(data.data()),
              std::streamsize(data.size() * sizeof(T)));
}
} // namespace

extern "C" int build_material(const double *vertices, const int32_t *faces, int nfaces, int side,
                              int partial_rule,
                              const char *dir, int threads, char *error, size_t error_size) {
    try {
        if (partial_rule < 0 || partial_rule > 2)
            throw std::runtime_error("unknown partial material quadrature rule");
        std::vector<std::vector<int32_t>> bins(size_t(side) * side);
        for (int i = 0; i < nfaces; ++i) {
            const double *a = vertices + 3 * int64_t(faces[3 * int64_t(i)]);
            const double *b = vertices + 3 * int64_t(faces[3 * int64_t(i) + 1]);
            const double *c = vertices + 3 * int64_t(faces[3 * int64_t(i) + 2]);
            int lo[2], hi[2];
            for (int d = 0; d < 2; ++d) {
                lo[d] = std::clamp(int(std::floor(std::min({a[d], b[d], c[d]}))), 0, side - 1);
                hi[d] = std::clamp(int(std::floor(std::max({a[d], b[d], c[d]}))), 0, side - 1);
            }
            for (int x = lo[0]; x <= hi[0]; ++x)
                for (int y = lo[1]; y <= hi[1]; ++y)
                    bins[x * side + y].push_back(i);
        }
        const auto rules = rotations();
        std::string base(dir);
        std::ofstream runs(base + "/runs", std::ios::binary),
            partials(base + "/partials", std::ios::binary),
            nodes(base + "/nodes", std::ios::binary);
        runs.exceptions(std::ios::failbit | std::ios::badbit);
        partials.exceptions(std::ios::failbit | std::ios::badbit);
        nodes.exceptions(std::ios::failbit | std::ios::badbit);
        uint64_t fallback = 0, outside = 0;
        double local_residual = 0.;
        // Bound simultaneous output memory independently of total mesh size.
        for (int first = 0; first < side * side; first += 1024) {
            int count = std::min(1024, side * side - first);
            std::vector<Column> result(count);
            std::vector<std::string> errors(count);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
            for (int j = 0; j < count; ++j) {
                int t = first + j;
                if (bins[t].empty())
                    continue;
                try {
                    result[j] =
                        integrate_column(t / side, t % side, side, bins[t], vertices, faces, rules,
                                         partial_rule);
                } catch (const std::exception &exc) {
                    errors[j] = "column " + std::to_string(t / side) + "," +
                                std::to_string(t % side) + ": " + exc.what();
                }
            }
            for (int j = 0; j < count; ++j) {
                if (!errors[j].empty())
                    throw std::runtime_error(errors[j]);
                const auto &r = result[j];
                write_array(runs, r.runs);
                write_array(partials, r.partials);
                write_array(nodes, r.nodes);
                fallback += r.fallback;
                outside += r.outside;
                local_residual = std::max(local_residual, r.local_residual);
            }
        }
        runs.close();
        partials.close();
        nodes.close();
        std::ofstream stats(base + "/stats");
        stats.precision(17);
        stats << fallback << ' ' << outside << ' ' << local_residual << '\n';
        stats.close();
        return 0;
    } catch (const std::exception &exc) {
        std::snprintf(error, error_size, "%s", exc.what());
        return 1;
    }
}
