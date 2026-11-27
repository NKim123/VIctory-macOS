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

uint clamped(uint kind, uint src, uint dst) {
    if (kind <= 1) {
        uint count = min(args[src], 1u << 24);
        uint instances = min(args[src + 1], (1u << 26) / max(count, 1u));
        dest.Store4(dst, uint4(count, instances, args[src + 2], args[src + 3]));
        if (kind == 1) {
            dest.Store(dst + 16, args[src + 4]);
        }
        return count * instances;
    }

    uint3 g = uint3(args[src], args[src + 1], args[src + 2]);
    if (kind == 2) {
        g = uint3(g.x <= widest ? g.x : 65535u, min(g.y, 65535u), min(g.z, 65535u));
        if ((g.y != 0 && g.x > GROUPS / g.y) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        dest.Store3(dst, g);
        return g.x * g.y * g.z * 64;
    }

    g = g.x > 65535u ? uint3(0, 0, 0) : uint3(g.x, min(g.y, 1u), min(g.z, 1u));
    dest.Store3(dst, g);
    return g.x * g.y * g.z * 128;
}

[numthreads(1, 1, 1)]
void main() {
    uint left = BUDGET - min(budgetWords[0], BUDGET);
    uint count = jobs[0].x;
    for (uint i = 0; i < count; i++) {
        uint2 job = jobs[i + 1].xy;
        uint dst = i * 32;
        uint cost = clamped(job.y, job.x, dst);
        if (cost > left) {
            if (job.y <= 1u) {
                dest.Store2(dst, uint2(0, 0));
            } else {
                dest.Store3(dst, uint3(0, 0, 0));
            }
            if (take_rest != 0u) {
                left = 0;
            }
        } else {
            left -= cost;
        }
    }
    budget.Store(0, BUDGET - left);
}
