// Reference Effect
// float4x3 bonesMat[58];
//
// struct a2v
// {
//     float3 position;
//     float3 normal;
//     float4 boneIndex;
//     float4 boneWeight;
// };
//
// struct v2f
// {
//     float3 position;
//     float3 normal;
// };
//
// v2f animatedModelVS(a2v IN)
// {
// 	v2f OUT;
//
// 	// Final bone transformation matrix
// 	float4x3 bonesTransform =
// 	    bonesMat[IN.boneIndex.x] * IN.boneWeight.x +
// 	    bonesMat[IN.boneIndex.y] * IN.boneWeight.y +
// 	    bonesMat[IN.boneIndex.z] * IN.boneWeight.z +
// 	    bonesMat[IN.boneIndex.w] * IN.boneWeight.w;
//
// 	float3 position = float4(IN.position, 1.0f);
// 	OUT.position = mul(position, bonesTransform);
//
// 	float3x3 normalTransform = float3x3(bonesTransform)
// 	// normalTransform = inverse(transpose(normalTransform)) // orthogonal matrix
// 	OUT.normal = mul(IN.normal, normalTransform);
//
//     return OUT;
// }

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../cutil_math.cuh"

#pragma pack(push, 1)
struct a2v {
    float3 position;
    float3 normal;
    float bone_weight[4];
    uint8_t bone_index[4];
    float2 uv0;
    float2 uv1;
} __align__(1);

struct v2f {
    float3 position;
    float3 normal;
    float2 uv0;
    float2 uv1;
} __align__(1);
#pragma pack(pop)

__device__ float4 float4_from(float* data, float scale) {
    return make_float4(data[0] * scale, data[1] * scale, data[2] * scale, data[3] * scale);
}

__global__ void skinning_kernel(const a2v* IN, v2f* OUT) {
    int bid = blockIdx.x;
    int bsize = blockDim.x;
    int tid = threadIdx.x;
    int vertex_id = bid * bsize + tid;

    // Up to 256 bones
    extern __shared__ uint8_t shared_mem[];
    float* bones_mat = reinterpret_cast<float*>(shared_mem);
    int32_t floats_per_bone = 12;

    const a2v& vertex = IN[vertex_id];
    int bone_index = vertex.bone_index[0];
    float bone_weight = vertex.bone_weight[0];

    float4 c0 = float4_from(&bones_mat[bone_index * floats_per_bone + 0], bone_weight);
    float4 c1 = float4_from(&bones_mat[bone_index * floats_per_bone + 4], bone_weight);
    float4 c2 = float4_from(&bones_mat[bone_index * floats_per_bone + 8], bone_weight);

    for (int32_t i=1; i < 4; ++i) {
        bone_index = vertex.bone_index[i];
        bone_weight = vertex.bone_weight[i];
        c0 += float4_from(&bones_mat[bone_index * floats_per_bone + 0], bone_weight);
        c1 += float4_from(&bones_mat[bone_index * floats_per_bone + 4], bone_weight);
        c2 += float4_from(&bones_mat[bone_index * floats_per_bone + 8], bone_weight);
    }

    v2f& out = OUT[vertex_id];
    float4 position = make_float4(vertex.position, 1.0f);
    float3 normal = vertex.normal;
    out.position = make_float3(
        dot(position, c0),
        dot(position, c1),
        dot(position, c2)
    );
    out.normal = make_float3(
        dot(normal, make_float3(c0)),
        dot(normal, make_float3(c1)),
        dot(normal, make_float3(c2))
    );
    out.uv0 = vertex.uv0;
    out.uv1 = vertex.uv1;
}


void printDeviceProp() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Capability: %d.%d\n", prop.major, prop.minor);
    printf("  totalGlobalMem:\t %lu MB\n", prop.totalGlobalMem / 1024 / 1024);
    printf("  sharedMemPerBlock:\t %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  totalConstMem:\t %lu KB\n", prop.totalConstMem / 1024);
    printf("  l2CacheSize:\t\t %d KB\n\n", prop.l2CacheSize / 1024);

    printf("  multiProcessorCount:\t %d\n", prop.multiProcessorCount);
    printf("  maxThreadsPerBlock:\t %d\n", prop.maxThreadsPerBlock);
    printf("  warpSize:\t\t %d\n\n", prop.warpSize);
}


int main() {
    printDeviceProp();

    a2v* input_vertices = nullptr;
    v2f* output_vertices = nullptr;

    int32_t vertex_count = 1024 * 1024; // 1M
    cudaMalloc(&input_vertices, vertex_count * sizeof(a2v));
    cudaMalloc(&output_vertices, vertex_count * sizeof(v2f));

    int32_t block_size = 16; // match FB32 ALU units, faster than warp_size (32)
    int32_t block_count = (vertex_count + block_size - 1) / block_size;
    int32_t shared_mem_size = 256 * 12 * sizeof(float);

    printf("blocks %d, threads %d\n", block_count, block_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    int32_t kernel_count = 16;
    for (int32_t i=0; i < kernel_count; ++i) {
        skinning_kernel<<<block_count, block_size, shared_mem_size>>>(input_vertices, output_vertices);
    }
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, start, stop);
    printf("elapsed_time_ms: %.3fms\n", elapsedMs/(float)kernel_count);

    cudaFree(input_vertices);
    cudaFree(output_vertices);

    cudaDeviceReset();

    return 0;
}