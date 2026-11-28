// One job per execution, in command order. Match the single clamp's limits
// and budget, including repeated records that have to be charged again.
cbuffer Limits : register(b0) { uint widest; uint take_rest; };
StructuredBuffer<uint> args : register(t0);
StructuredBuffer<uint4> jobs : register(t1);
RWByteAddressBuffer dest : register(u0);
RWByteAddressBuffer budget : register(u1);
RWStructuredBuffer<uint> budgetWords : register(u2);

static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;

uint cost_and_arguments(uint kind, uint src, out uint3 arguments) {
    if (kind <= 1) {
        uint count = min(args[src], 1u << 24);
        uint instances = min(args[src + 1], (1u << 26) / max(count, 1u));
        arguments = uint3(count, instances, 0);
        return count * instances;
    }

    uint3 g = uint3(args[src], args[src + 1], args[src + 2]);
    if (kind == 2) {
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        arguments = g;
        return g.x * g.y * g.z * 64;
    }

    g = g.x > 65535u ? uint3(0, 0, 0) : uint3(g.x, min(g.y, 1u), min(g.z, 1u));
    arguments = g;
    return g.x * g.y * g.z * 128;
}

void store_arguments(uint i, uint2 job, uint3 arguments) {
    uint dst = i * 32u;
    if (job.y <= 1u) {
        dest.Store4(dst, uint4(arguments.xy, args[job.x + 2u], args[job.x + 3u]));
        if (job.y == 1u) {
            dest.Store(dst + 16u, args[job.x + 4u]);
        }
    } else {
        dest.Store3(dst, arguments);
    }
}

#ifdef PARALLEL_BATCH
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), UAV(u2)"
static const uint TILE = 32u;
static const uint REJECTED = 0xffffffffu;
groupshared uint costs[TILE];

// Job metadata is finalized before submission, so all lanes see the same
// tile count and reach both barriers, including lanes without a final job.
[RootSignature(RS)]
[numthreads(32, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0u;
    if (lane == 0u) {
        left = BUDGET - min(budgetWords[0], BUDGET);
    }
    uint count = jobs[0].x;
    for (uint base = 0u; base < count; base += TILE) {
        uint i = base + lane;
        uint2 job = uint2(0, 0);
        uint3 arguments = uint3(0, 0, 0);
        uint cost = 0u;
        if (i < count) {
            job = jobs[i + 1u].xy;
            cost = cost_and_arguments(job.y, job.x, arguments);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();

        if (lane == 0u) {
            uint active = min(TILE, count - base);
            for (uint slot = 0u; slot < active; slot++) {
                uint charge = costs[slot];
                if (charge > left) {
                    costs[slot] = REJECTED;
                    if (take_rest != 0u) {
                        left = 0u;
                    }
                } else {
                    left -= charge;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (i < count) {
            store_arguments(i, job, costs[lane] == REJECTED ? uint3(0, 0, 0) : arguments);
        }
        // The first barrier of the next tile protects lane zero's reads
        // from incomplete writes. Each lane reads only its own result.
    }
    if (lane == 0u) {
        budget.Store(0, BUDGET - left);
    }
}
#else
// Wine's runtime compiler cannot store to indexed group-shared memory.
[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint count = jobs[0].x;
    for (uint i = 0u; i < count; i++) {
        uint2 job = jobs[i + 1u].xy;
        uint3 arguments;
        uint cost = cost_and_arguments(job.y, job.x, arguments);
        if (cost > left) {
            arguments = uint3(0, 0, 0);
            if (take_rest != 0u) {
                left = 0u;
            }
        } else {
            left -= cost;
        }
        store_arguments(i, job, arguments);
    }
    budget.Store(0, BUDGET - left);
}
#endif
