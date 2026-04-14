#include "QuadricSimplifier.h"
#include "renderer/Mesh.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>

namespace acuity {
    // Quadric implementation

    Quadric Quadric::operator+(const Quadric& o) const {
        Quadric r;
        for (int i = 0; i < 10; ++i) r.data[i] = data[i] + o.data[i];
        return r;
    }

    Quadric& Quadric::operator+=(const Quadric& o) {
        for (int i = 0; i < 10; ++i) data[i] += o.data[i];
        return *this;
    }

    // Evaluate v^T * Q * v where Q is the 4x4 symmetric matrix
    double Quadric::evaluate(const glm::dvec3& v) const {
        double x = v.x, y = v.y, z = v.z;
        // Expand the matrix multiply for the 10 stored values
        return  data[0]*x*x + 2*data[1]*x*y + 2*data[2]*x*z + 2*data[3]*x
            + data[4]*y*y + 2*data[5]*y*z + 2*data[6]*y
            + data[7]*z*z + 2*data[8]*z
            + data[9];
    }

    // Build a quadric from a plane ax+by+cz+d=0 derived from a triangle
    Quadric Quadric::fromTriangle(const glm::dvec3& p0,
                                const glm::dvec3& p1,
                                const glm::dvec3& p2)
    {
        // Compute plane normal
        glm::dvec3 e1  = p1 - p0;
        glm::dvec3 e2  = p2 - p0;
        glm::dvec3 n   = glm::cross(e1, e2);
        double     len = glm::length(n);

        Quadric q;
        if (len < 1e-12) return q; // degenerate triangle

        n /= len; // normalize
        double d = -glm::dot(n, p0);

        // Q = [n.x, n.y, n.z, d]^T * [n.x, n.y, n.z, d]
        // Store upper triangle of the outer product
        double a = n.x, b = n.y, c = n.z;
        q.data[0] = a*a; q.data[1] = a*b; q.data[2] = a*c; q.data[3] = a*d;
        q.data[4] = b*b; q.data[5] = b*c; q.data[6] = b*d;
        q.data[7] = c*c; q.data[8] = c*d;
        q.data[9] = d*d;
        return q;
    }

    // Build per-vertex quadrics

    std::vector<Quadric> QuadricSimplifier::buildQuadrics(
        const std::vector<glm::dvec3>& positions,
        const std::vector<uint32_t>&   indices)
    {
        std::vector<Quadric> Q(positions.size());

        // For each triangle, compute its plane quadric and add to all 3 vertices
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            uint32_t i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];
            if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
                continue;
            Quadric tq = Quadric::fromTriangle(positions[i0], positions[i1], positions[i2]);
            Q[i0] += tq;
            Q[i1] += tq;
            Q[i2] += tq;
        }
        return Q;
    }

    // Compute optimal edge collapse

    EdgeCollapse QuadricSimplifier::computeCollapse(
        uint32_t                       v0,
        uint32_t                       v1,
        const std::vector<glm::dvec3>& positions,
        const std::vector<Quadric>&    quadrics)
    {
        EdgeCollapse ec;
        ec.v0 = v0;
        ec.v1 = v1;

        Quadric combined = quadrics[v0] + quadrics[v1];

        // Try to find the optimal position by solving the linear system
        // The optimal point minimizes Q(v): dQ/dv = 0
        // This gives a 3x3 linear system from the top-left of the 4x4 matrix
        double a00 = combined.data[0], a01 = combined.data[1], a02 = combined.data[2];
        double a11 = combined.data[4], a12 = combined.data[5];
        double a22 = combined.data[7];
        double b0  = -combined.data[3], b1 = -combined.data[6], b2 = -combined.data[8];

        // Determinant of the 3x3 matrix
        double det = a00*(a11*a22 - a12*a12)
                - a01*(a01*a22 - a12*a02)
                + a02*(a01*a12 - a11*a02);

        if (std::abs(det) > 1e-10) {
            // Cramer's rule to solve for optimal position
            double x = (b0*(a11*a22 - a12*a12) - a01*(b1*a22 - a12*b2) + a02*(b1*a12 - a11*b2)) / det;
            double y = (a00*(b1*a22 - a12*b2) - b0*(a01*a22 - a12*a02) + a02*(a01*b2 - b1*a02)) / det;
            double z = (a00*(a11*b2 - b1*a12) - a01*(a01*b2 - b1*a02) + b0*(a01*a12 - a11*a02)) / det;
            ec.optimalPos = glm::dvec3(x, y, z);
        } else {
            // Degenerate: fall back to midpoint or endpoint with lower error
            glm::dvec3 mid = (positions[v0] + positions[v1]) * 0.5;
            double c0  = combined.evaluate(positions[v0]);
            double c1  = combined.evaluate(positions[v1]);
            double cm  = combined.evaluate(mid);
            if (c0 <= c1 && c0 <= cm)      ec.optimalPos = positions[v0];
            else if (c1 <= c0 && c1 <= cm) ec.optimalPos = positions[v1];
            else                            ec.optimalPos = mid;
        }

        ec.cost = combined.evaluate(ec.optimalPos);
        return ec;
    }

    // Main simplification algorithm

    void QuadricSimplifier::simplify(
        const std::vector<Vertex>&   inVertices,
        const std::vector<uint32_t>& inIndices,
        float                        targetRatio,
        std::vector<Vertex>&         outVertices,
        std::vector<uint32_t>&       outIndices)
    {
        auto t0 = std::chrono::steady_clock::now();

        size_t origTriCount = inIndices.size() / 3;
        size_t targetTris   = static_cast<size_t>(origTriCount * targetRatio);

        // If ratio is close to 1.0, just copy and return
        if (targetRatio >= 0.99f) {
            outVertices = inVertices;
            outIndices  = inIndices;
            return;
        }

        size_t N = inVertices.size();

        // Extract positions as double for numerical stability
        std::vector<glm::dvec3> positions(N);
        for (size_t i = 0; i < N; ++i)
            positions[i] = glm::dvec3(inVertices[i].position);

        // Copy indices
        std::vector<uint32_t> indices = inIndices;

        // Build per-vertex quadrics
        std::vector<Quadric> quadrics = buildQuadrics(positions, indices);

        // Union-Find for tracking vertex merges
        // collapsed[i] = the vertex i has been merged into (itself if not collapsed)
        std::vector<uint32_t> representative(N);
        std::iota(representative.begin(), representative.end(), 0);

        std::function<uint32_t(uint32_t)> find = [&](uint32_t v) -> uint32_t {
            while (representative[v] != v) {
                representative[v] = representative[representative[v]];
                v = representative[v];
            }
            return v;
        };

        // Build edge set (undirected, no duplicates)
        std::unordered_set<uint64_t> edgeSet;
        auto edgeKey = [](uint32_t a, uint32_t b) -> uint64_t {
            if (a > b) std::swap(a, b);
            return (uint64_t(a) << 32) | b;
        };

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            uint32_t i0 = indices[i], i1 = indices[i+1], i2 = indices[i+2];
            edgeSet.insert(edgeKey(i0, i1));
            edgeSet.insert(edgeKey(i1, i2));
            edgeSet.insert(edgeKey(i0, i2));
        }

        // Build initial priority queue of edge collapses
        using MinHeap = std::priority_queue<EdgeCollapse,
                                            std::vector<EdgeCollapse>,
                                            std::greater<EdgeCollapse>>;
        MinHeap heap;
        for (uint64_t key : edgeSet) {
            uint32_t a = uint32_t(key >> 32);
            uint32_t b = uint32_t(key & 0xFFFFFFFF);
            heap.push(computeCollapse(a, b, positions, quadrics));
        }

        size_t currentTris = origTriCount;

        // Main collapse loop
        while (currentTris > targetTris && !heap.empty()) {
            EdgeCollapse ec = heap.top();
            heap.pop();

            uint32_t ra = find(ec.v0);
            uint32_t rb = find(ec.v1);

            // Skip if already merged or stale
            if (ra == rb) continue;

            // Merge rb into ra
            positions[ra]  = ec.optimalPos;
            quadrics[ra]  += quadrics[rb];
            representative[rb] = ra;

            // Count how many triangles become degenerate after this collapse
            // A triangle is degenerate if two or more of its vertices map to the same representative
            size_t degenerateTris = 0;
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t r0 = find(indices[i]);
                uint32_t r1 = find(indices[i+1]);
                uint32_t r2 = find(indices[i+2]);
                if (r0 == r1 || r1 == r2 || r0 == r2) ++degenerateTris;
            }
            currentTris = origTriCount - degenerateTris;

            // Push new collapses for edges adjacent to ra
            // We scan edges in the current set that involve ra or rb
            std::unordered_set<uint32_t> neighbors;
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t r0 = find(indices[i]);
                uint32_t r1 = find(indices[i+1]);
                uint32_t r2 = find(indices[i+2]);
                if (r0 == ra || r1 == ra || r2 == ra) {
                    if (r0 != ra) neighbors.insert(r0);
                    if (r1 != ra) neighbors.insert(r1);
                    if (r2 != ra) neighbors.insert(r2);
                }
            }
            for (uint32_t nb : neighbors)
                heap.push(computeCollapse(ra, nb, positions, quadrics));
        }

        // Extract simplified mesh

        // Remap surviving vertices
        std::unordered_map<uint32_t, uint32_t> remap;
        outVertices.clear();

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            uint32_t r0 = find(indices[i]);
            uint32_t r1 = find(indices[i+1]);
            uint32_t r2 = find(indices[i+2]);

            // Skip degenerate triangles
            if (r0 == r1 || r1 == r2 || r0 == r2) continue;

            uint32_t mapped[3];
            for (auto [rep, m] : std::array<std::pair<uint32_t,uint32_t*>,3>{
                    {{r0,&mapped[0]},{r1,&mapped[1]},{r2,&mapped[2]}}}) {
                if (remap.find(rep) == remap.end()) {
                    remap[rep] = static_cast<uint32_t>(outVertices.size());
                    Vertex v   = inVertices[rep];
                    v.position = glm::vec3(positions[rep]);
                    outVertices.push_back(v);
                }
                *m = remap[rep];
            }
            outIndices.push_back(mapped[0]);
            outIndices.push_back(mapped[1]);
            outIndices.push_back(mapped[2]);
        }

        auto t1 = std::chrono::steady_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        std::cout << "[QEM] " << origTriCount << " -> " << outIndices.size()/3
                << " triangles (target: " << targetTris << ")"
                << " | ratio: " << targetRatio
                << " | time: " << ms << "ms" << std::endl;
    }
}