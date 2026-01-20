/*
 * CUDA kernels for IMU preprocessing.
 * Windowing, normalization, and FIR filtering.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace cuda_preprocessing {

// Windowing kernel
template <typename scalar_t>
__global__ void sliding_window_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int channels,
    const int window_size,
    const int stride,
    const int n_windows
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * n_windows * window_size * channels;

    if (idx >= total) return;

    const int c = idx % channels;
    const int w = (idx / channels) % window_size;
    const int n = (idx / (channels * window_size)) % n_windows;
    const int b = idx / (channels * window_size * n_windows);

    const int t = n * stride + w;
    const int in_idx = b * seq_len * channels + t * channels + c;
    output[idx] = input[in_idx];
}

torch::Tensor sliding_window_cuda(
    torch::Tensor input,
    int window_size,
    int stride
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, T, C)");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int n_windows = (seq_len - window_size) / stride + 1;

    auto output = torch::empty({batch_size * n_windows, window_size, channels},
                               input.options());

    const int total = batch_size * n_windows * window_size * channels;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sliding_window_cuda", ([&] {
        sliding_window_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, seq_len, channels, window_size, stride, n_windows
        );
    }));

    return output;
}

// Parallel reduction for mean computation
template <typename scalar_t>
__global__ void compute_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ mean,
    const int batch_size,
    const int seq_len,
    const int channels
) {
    extern __shared__ scalar_t sdata[];

    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;

    scalar_t sum = 0;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        sum += input[b * seq_len * channels + t * channels + c];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        mean[b * channels + c] = sdata[0] / seq_len;
    }
}

// Parallel reduction for variance computation
template <typename scalar_t>
__global__ void compute_var_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var,
    const int batch_size,
    const int seq_len,
    const int channels
) {
    extern __shared__ scalar_t sdata[];

    const int b = blockIdx.x;
    const int c = blockIdx.y;
    const int tid = threadIdx.x;
    const scalar_t m = mean[b * channels + c];

    scalar_t sum = 0;
    for (int t = tid; t < seq_len; t += blockDim.x) {
        scalar_t diff = input[b * seq_len * channels + t * channels + c] - m;
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        var[b * channels + c] = sdata[0] / seq_len;
    }
}

// Normalize kernel
template <typename scalar_t>
__global__ void normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const int batch_size,
    const int seq_len,
    const int channels,
    const scalar_t eps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * channels;

    if (idx >= total) return;

    const int c = idx % channels;
    const int t = (idx / channels) % seq_len;
    const int b = idx / (channels * seq_len);

    const scalar_t m = mean[b * channels + c];
    const scalar_t s = sqrt(var[b * channels + c] + eps);
    output[idx] = (input[idx] - m) / s;
}

torch::Tensor normalize_cuda(
    torch::Tensor input,
    float eps
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, T, C)");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    auto mean = torch::empty({batch_size, channels}, input.options());
    auto var = torch::empty({batch_size, channels}, input.options());
    auto output = torch::empty_like(input);

    dim3 grid(batch_size, channels);
    const int threads = min(256, seq_len);
    const int smem = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "normalize_cuda", ([&] {
        compute_mean_kernel<scalar_t><<<grid, threads, smem>>>(
            input.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            batch_size, seq_len, channels
        );

        compute_var_kernel<scalar_t><<<grid, threads, smem>>>(
            input.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            batch_size, seq_len, channels
        );

        const int total = batch_size * seq_len * channels;
        const int norm_threads = 256;
        const int norm_blocks = (total + norm_threads - 1) / norm_threads;

        normalize_kernel<scalar_t><<<norm_blocks, norm_threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            batch_size, seq_len, channels,
            static_cast<scalar_t>(eps)
        );
    }));

    return output;
}

// FIR filter kernel (1D convolution)
template <typename scalar_t>
__global__ void fir_filter_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ kernel,
    const int batch_size,
    const int seq_len,
    const int channels,
    const int kernel_size,
    const int padding
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len * channels;

    if (idx >= total) return;

    const int c = idx % channels;
    const int t = (idx / channels) % seq_len;
    const int b = idx / (channels * seq_len);

    scalar_t sum = 0;
    for (int k = 0; k < kernel_size; k++) {
        int in_t = t - padding + k;
        if (in_t >= 0 && in_t < seq_len) {
            sum += input[b * seq_len * channels + in_t * channels + c] * kernel[k];
        }
    }
    output[idx] = sum;
}

torch::Tensor fir_filter_cuda(
    torch::Tensor input,
    torch::Tensor kernel,
    bool zero_phase
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(kernel.is_cuda(), "Kernel must be CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, T, C)");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);
    const int kernel_size = kernel.size(0);
    const int padding = kernel_size / 2;

    auto output = torch::empty_like(input);

    const int total = batch_size * seq_len * channels;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    auto kernel_flipped = kernel.flip(0);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fir_filter_cuda", ([&] {
        fir_filter_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            kernel_flipped.data_ptr<scalar_t>(),
            batch_size, seq_len, channels, kernel_size, padding
        );
    }));

    if (zero_phase) {
        auto output2 = torch::empty_like(input);
        auto output_flipped = output.flip(1);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fir_filter_cuda_rev", ([&] {
            fir_filter_kernel<scalar_t><<<blocks, threads>>>(
                output_flipped.data_ptr<scalar_t>(),
                output2.data_ptr<scalar_t>(),
                kernel_flipped.data_ptr<scalar_t>(),
                batch_size, seq_len, channels, kernel_size, padding
            );
        }));

        return output2.flip(1);
    }

    return output;
}

// SMV kernel
template <typename scalar_t>
__global__ void smv_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int channels
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * seq_len;

    if (idx >= total) return;

    const int t = idx % seq_len;
    const int b = idx / seq_len;

    scalar_t sum = 0;
    for (int c = 0; c < channels; c++) {
        scalar_t val = input[b * seq_len * channels + t * channels + c];
        sum += val * val;
    }
    output[b * seq_len + t] = sqrt(sum);
}

torch::Tensor compute_smv_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, T, C)");

    const int batch_size = input.size(0);
    const int seq_len = input.size(1);
    const int channels = input.size(2);

    auto output = torch::empty({batch_size, seq_len, 1}, input.options());

    const int total = batch_size * seq_len;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "smv_cuda", ([&] {
        smv_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size, seq_len, channels
        );
    }));

    return output;
}

}  // namespace cuda_preprocessing

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sliding_window", &cuda_preprocessing::sliding_window_cuda, "Sliding window (CUDA)");
    m.def("normalize", &cuda_preprocessing::normalize_cuda, "Z-score normalize (CUDA)");
    m.def("fir_filter", &cuda_preprocessing::fir_filter_cuda, "FIR filter (CUDA)");
    m.def("compute_smv", &cuda_preprocessing::compute_smv_cuda, "Signal Vector Magnitude (CUDA)");
}
