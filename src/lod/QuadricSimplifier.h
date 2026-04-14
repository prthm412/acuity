#pragma once
#include <glm/glm.hpp>
#include <vector>

namespace acuity {
    struct Vertex;

    // Q(v) = v^T * Q * v ----> gives the geometric error at position v.

    // Matrix layout:
    // [ a b c d ]
    // [ b e f g ]
    // [ c f h i ]
    // [ d g i j ]
    struct Quadric {
        double data[10] = {0.0};

        // Combine two quadrics when merging vertices
        Quadric  operator+ (const Quadric& o) const;
        Quadric& operator+=(const Quadric& o);

        // Evaluate geometric error at position v
        double evaluate(const glm::dvec3& v) const;

        // Build a quadric from a triangle defined by three positions
        // The quadric encodes the plane equation of the triangle
        static Quadric fromTriangle(const glm::dvec3& p0,
                                    const glm::dvec3& p1,
                                    const glm::dvec3& p2);
    };

    // Represents one candidate edge collapse with its cost
    struct EdgeCollapse {
        uint32_t    v0, v1;         // two vertices to merge
        double      cost;           // quadric error cost of this collapse
        glm::dvec3  optimalPos;     // best position for the merged vertex

        // Used by std::priority_queue (min-heap: cheapest collapse first)
        bool operator>(const EdgeCollapse& o) const { return cost > o.cost; }
    };

    class QuadricSimplifier {
        public:
            // Simplify inVertices/inIndices to targetRatio fraction of triangles.
            // targetRatio values:
            // 1.0 - as it is, 0.5 - half, 0.25 - 25%, 0.125 - 12.5%, 0.0625 - 6.25%
            static void simplify(
                const std::vector<Vertex>&  inVertices,
                const std::vector<uint32_t>& inIndices,
                float                       targetRatio,
                std::vector<Vertex>&        outVertices,
                std::vector<uint32_t>&      outIndices
            );
        
        private:
            static std::vector<Quadric> buildQuadrics(
                const std::vector<glm::dvec3>& positions,
                const std::vector<uint32_t>&   indices
            );

            static EdgeCollapse computeCollapse(
                uint32_t                        v0,
                uint32_t                        v1,
                const std::vector<glm::dvec3>&  positions,
                const std::vector<Quadric>&     quadrics
            );
    };
}