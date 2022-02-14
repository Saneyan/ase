/**
 * Library of Adaptive Stream-based Entropy Coding (ASE-Coding) for CUDA GPUs
 * 
 * .cu ソースコード向けヘッダファイル.
 * 
 * Author: Saneyuki Tadokoro (201311374)
 */

#include "ase.h"

namespace ase {

// ASE コンテキスト. カーネルのデバイス関数間で利用する.
// culling_num メンバは, 一度セットしたら読み取りのみ (readonly) として取り扱うこと.
struct Context {
  int occupied;       // 占有エントリ数
  int global_counter; // グローバルカウンター
  int culling_num;    // エントロピーカリング数 (readonly)
  int max_entropy;    // エントロピーの最大値
};

int resetDevice();

__host__ __device__
Data* malloc_ase_data();

__host__ __device__
Data* malloc_next_ase_data(Buffer* buf);

__host__ __device__
Data* malloc_next_ase_data(Buffer* buf);

__host__ __device__
Buffer* malloc_ase_buffer();

__host__ __device__
Context* malloc_ase_context(int global_counter);

__host__ __device__
int free_head_ase_buffer(Buffer *buf);

__host__ __device__
int write_data_to_buf(Buffer* buf, const char* data, const unsigned int width);

__host__ __device__
int read_data_from_buf(Buffer* buf, char* data, const unsigned int width);

__host__ __device__
void entropy_culling(Context *context);

__host__ __device__
void arrange_table(Context *context,
                   char *entries,
                   const int hit_index,
                   const char symbol);

__host__ __device__
void register_to_table(Context *context,
                       char *entries,
                       const char symbol);

__host__ __device__
int push(Context *context,
         char *entries,
         const char symbol);

__host__ __device__
int entropy_calc(Context *context);

// __global__
// void kernel_compress(const char *d_input_data,
//                      Buffer *d_out_bufs,
//                      long *d_counts,
//                      const CompDescriptor *descs);

// __global__
// void kernel_decompress(Buffer **d_input_bufs,
//                        char *d_output_data,
//                        long *d_counts,
//                        const DecompDescriptor *descs);

__host__
void host_compress(const char *input_data,
                   Buffer *out_buf,
                   long *counts,
                   const CompDescriptor *desc);

__host__
void host_decompress(Buffer *input_buf,
                     char *output_data,
                     long *counts,
                     const DecompDescriptor *desc);

} // namespace ase
