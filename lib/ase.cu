/**
 * Implementation of Adaptive Stream-based Entropy Coding (ASE-Coding) for CUDA GPUs
 * 
 * このプログラムは, ASE-Coding のリファレンス実装である. GPU のメニーコアを利用して, 並列圧縮を行う.
 * また, NVIDIA CUDA に対応したハードウェア上でのみ動作する.
 * 
 * Author: Saneyuki Tadokoro (201311374)
 */

#include <iostream>
#include <tuple>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "ase.cuh"

__device__ static const char CMARK_TRUE = 1;
__device__ static const char CMARK_FALSE = 0;

namespace ase {

// デバイスのリセットを行う.
int resetDevice() {
  const cudaError_t error = cudaDeviceReset();
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 
    fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));
    return -1;
  }
  return 0;
}

// スレッド数が1より多い場合は, 入力ストリームを十分に分割できないので, 不足しているスレッド数の計算を行う.
int correct_threads(int threads, int total_size, int chunk_size) {
  int lack_threads = 0;

  if (threads > 1) {
    int remaining = total_size - (total_size / threads) * threads;
    if (remaining <= chunk_size) {
      lack_threads = 1;
    } else {
      lack_threads = (remaining / chunk_size) + 1;
    }
  }

  return threads + lack_threads;
}

CompDescriptor** malloc_comp_descriptors(const Partition *partition, int nread) {
  CompDescriptor **descs = (CompDescriptor**)malloc(partition->grids * sizeof(CompDescriptor));
  for (int i = 0; i < partition->grids; i++) {
    descs[i]->entry_size = partition->allocations[i].entry_size;
    descs[i]->global_counter = partition->allocations[i].global_counter;
    descs[i]->chunk_size = nread / partition->allocations[i].threads;
    descs[i]->total_size = nread / partition->grids;
    descs[i]->threads = correct_threads(partition->allocations[i].threads, descs[i]->total_size, descs[i]->chunk_size);
  }
  return descs;
}

DecompDescriptor** malloc_decomp_descriptors(const Partition *partition, long *counts) {
  DecompDescriptor **descs = (DecompDescriptor**)malloc(partition->grids * sizeof(DecompDescriptor));
  for (int i = 0; i< partition->grids; i++) {
    descs[i]->entry_size = partition->allocations[i].entry_size;
    descs[i]->global_counter = partition->allocations[i].global_counter;
    descs[i]->threads = partition->allocations[i].threads;
    descs[i]->counts = counts;
  }
  return descs;
}

__host__ __device__
Data* malloc_ase_data() {
  Data *new_data;

  if ((new_data = (Data*)malloc(sizeof(Data))) == NULL)
    return NULL;
  new_data->data = 0;
  return new_data;
}

__host__ __device__
Data* malloc_next_ase_data(Buffer* buf) {
  Data *new_data;

  if ((new_data = malloc_ase_data()) == NULL)
    return NULL;
  buf->current->next = new_data;
  buf->current = new_data;
  buf->t_offset = 0;
  return new_data;
}

__host__ __device__
Buffer* malloc_ase_buffer() {
  Buffer *new_buf;
  Data *new_data;

  if ((new_buf = (Buffer*)malloc(sizeof(Buffer))) == NULL)
    return NULL;
  if ((new_data = malloc_ase_data()) == NULL)
    return NULL;
  new_buf->head = new_data;
  new_buf->current = new_data;
  new_buf->max_width = D_MAX_WIDTH;
  new_buf->t_offset = 0;
  new_buf->h_offset = 0;
  return new_buf;
}

__host__ __device__
Buffer** malloc_ase_buffer(int nums) {
  Buffer **new_bufs = (Buffer**)malloc(nums * sizeof(Buffer));
  Data *new_data;
  int i;

  for (i = 0; i < nums; i++) {
    if ((new_data = malloc_ase_data()) == NULL)
      return NULL;
    new_bufs[i]->head = new_data;
    new_bufs[i]->current = new_data;
    new_bufs[i]->max_width = D_MAX_WIDTH;
    new_bufs[i]->t_offset = 0;
    new_bufs[i]->h_offset = 0;
  }
  return new_bufs;
}

__host__ __device__
Context* malloc_ase_context(int global_counter) {
  Context *context;

  if ((context = (Context *)malloc(sizeof(Context))) == NULL)
    return NULL;
  context->occupied = 0;
  context->global_counter = global_counter;
  context->culling_num = global_counter;
  context->max_entropy = 1;
  return context;
}

__host__ __device__
int free_head_ase_buffer(Buffer *buf) {
  Data *new_head;

  if (buf->head == NULL) {
    return -1;
  }

  if (buf->head->next != NULL) {
    new_head = buf->head->next;
    free(buf->head);
    buf->head = new_head;
  } else {
    free(buf->head);
    buf->current = NULL;
    buf->head = NULL;
  }
  buf->h_offset = 0;
  return 0;
}

// 8ビット長のデータバッファに任意サイズの入力データを書き込む関数.
// データバッファには, 上位ビットから入力データを詰める形式で書き込む.
__host__ __device__
int write_data_to_buf(Buffer* buf, const char* data, const unsigned int width) {
  int shift = buf->max_width - buf->t_offset - width;

  // シフト量が 0 より少ない
  if (shift < 0) {
    // 入力データに対する残量シフト量, すなわち -1 * shift に等しい.
    // 注意! unsigned char で論理右シフトを行う.
    buf->current->data |= (unsigned char)*data >> (-1 * shift);

    // 新規バッファ確保
    if (malloc_next_ase_data(buf) == NULL)
      return -1;

    // データの最大の長さにマイナス値の shift を加えると, 残りの書き込み量が計算できる (反転).
    shift += buf->max_width;
  }
  buf->current->data |= *data << shift;
  buf->t_offset = buf->max_width - shift;

  return 0;
}

__host__ __device__
int read_data_from_buf(Buffer* buf, char* data, const unsigned int width) {
  int shift = buf->max_width - buf->h_offset - width;
  int remaining = 0;
  *data = 0;

  if (shift < 0) {
    *data = (buf->head->data & (
      (unsigned char)0xFF >> buf->h_offset
    )) << (-1 * shift);

    // 不要バッファ解放
    if (free_head_ase_buffer(buf) == -1) {
      return -1;
    }
    
    remaining = width + shift;
    shift += buf->max_width;
  }
  *data |= (unsigned char)(
    buf->head->data & (
      (unsigned char)(0xFF << buf->max_width - (width - remaining)) >> buf->h_offset
    )) >> shift;

  buf->h_offset = buf->max_width - shift;

  if (buf->h_offset == buf->max_width) {
    if (free_head_ase_buffer(buf) == -1)
      return -1;
  } 

  return 0;
}

// エントロピーカリングを行う.
__host__ __device__
void entropy_culling(Context *context) {
  if (context->global_counter > 0) {
    context->global_counter--;
  } else {
    if (context->occupied > 0) {
      context->occupied--;
    }
    context->global_counter = context->culling_num;
  }
}

// ヒットしたシンボルがある場合に呼び出される.
// ルックアップテーブル内にヒットしたシンボルは, 最下位エントリにピボットし、
// それ以外のシンボルは上位エントリにピボットする.
__host__ __device__
void arrange_table(Context *context,
                   char *entries,
                   const int hit_index,
                   const char symbol) {
  int i;

  // 占有エントリ数が1でかつ最下位エントリのシンボルがヒットし続ける場合、
  // エントロピーカリングが実行されて占有エントリ数が 0 にデクリメントされないようにする.
  // つまり, この場合は常に最短ビットが生成されることになる.
  if (context->occupied > 1 && hit_index > 0) {
    for (i = hit_index - 1; i >= 0; i--) {
      entries[i + 1] = entries[i];
    }
    entries[0] = symbol;
    entropy_culling(context);
  }
}

// ミスヒットしたシンボルがある場合に呼び出される.
// 最下位エントリにシンボルを追加し, それ以外のシンボルは上位エントリにピボットする.
// ルックアップテーブルのエントリ数がいっぱいになっている場合は, シンボルを追加しない.
__host__ __device__
void register_to_table(Context *context,
                       char *entries,
                       const char symbol) {
  int i;

  for (i = context->occupied - 1; i >= 0; i--) {
    if (i + 1 < E_LENGTH) {
      entries[i + 1] = entries[i];
    }
  }
  entries[0] = symbol;
  if (context->occupied + 1 < E_LENGTH) {
    context->occupied++;
  }
}

// ルックアップテーブルに登録されているシンボルがヒットするかどうかを確認し,
// ヒットすれば該当するエントリインデックスが, ミスヒットすれば-1を返す.
// また, いずれの場合であってもルックアップテーブル操作を試みる.
__host__ __device__
int push(Context *context,
         char *entries,
         const char symbol) {
  int i;

  for (i = 0; i < context->occupied; i++) {
    if (entries[i] == symbol) {
      arrange_table(context, entries, i, symbol);
      return i;
    }
  }
  register_to_table(context, entries, symbol);
  return -1;
}

// 現在の占有エントリ数を用いてエントロピー計算を行う.
__host__ __device__
int entropy_calc(Context *context) {
  int m = (int) ceilf(log2f(context->occupied));
  if (context->max_entropy < m)
    context->max_entropy = m;
  return context->max_entropy;
}

// ASE 圧縮を行うカーネル関数. 入力ストリームを N 分割したストリームをそれぞれのスレッドが ASE 圧縮
// を行う. スレッドが処理すべきデータサイズは ase_settings の chunk_size に定められている.
__global__
void kernel_compress(const char *d_input_data,
                     Buffer **d_out_bufs,
                     long *d_counts,
                     const CompDescriptor **descs) {
  int i, m, hit_index;
  char hit_index_m, symbol;
  const int chunk_size = descs[0]->chunk_size;

  // スレッド番号の計算
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // ルックアップテーブルのエントリ初期化
  char entries[E_LENGTH] = {0};

  Context *context = malloc_ase_context(descs[0]->global_counter);

  for (i = 0; i < chunk_size; i++) {
    // オーバーフローチェック. スレッドが処理すべきデータ範囲を超える場合には処理を終了する.
    if (idx * chunk_size + i > descs[0]->total_size)
      break;

    // シンボルがヒットするかどうかを確認し, ルックアップテーブルを操作する.
    symbol = d_input_data[idx * chunk_size + i];
    hit_index = push(context, entries, symbol);

    // ヒットしなかった場合は, cmark ビットを 0 とし, 圧縮せずにバッファに追加する (シリアライズ).
    if (hit_index == -1) {
      d_counts[idx] = d_counts[idx] + 1 + SYM_SIZE;

      write_data_to_buf(d_out_bufs[idx], &CMARK_FALSE, 1);
      write_data_to_buf(d_out_bufs[idx], &symbol, SYM_SIZE);

    // ヒットした場合は, cmark ビットを 1 とし, 圧縮してシリアライズする.
    } else {
      m = entropy_calc(context);
      hit_index_m = hit_index & ((1 << m) - 1);

      d_counts[idx] = d_counts[idx] + 1 + m;

      write_data_to_buf(d_out_bufs[idx], &CMARK_TRUE, 1);
      write_data_to_buf(d_out_bufs[idx], &hit_index_m, m);
    }
  }
}

// ASE 解凍を行うカーネル関数. 入力ストリームを N 分割したストリームをそれぞれのスレッドが ASE 解凍
// を行う. スレッドが処理すべきデータサイズは ase_settings の chunk_size に定められている.
__global__
void kernel_decompress(Buffer **d_input_bufs,
                       char *d_output_data,
                       long *d_counts,
                       const DecompDescriptor **descs) {
  // int m;
  // int counter = 0;
  // int remaining = settings->bit_size;
  // char index_m, cmark, symbol;
  // char entries[E_LENGTH] = {0};

  // Context *context = malloc_ase_context(settings);

  // read_data_from_buf(input_buf, &cmark, 1);

  // while (true) {
  //   if (cmark == CMARK_TRUE) {
  //     m = entropy_calc(context);

  //     read_data_from_buf(input_buf, &index_m, m);
  //     symbol = entries[index_m];
  //     output_data[counter] = symbol;

  //     arrange_table(context, entries, index_m, symbol);
  //     remaining = remaining - 1 - m;
  //   } else {
  //     read_data_from_buf(input_buf, &symbol, SYM_SIZE);
  //     output_data[counter] = symbol;

  //     register_to_table(context, entries, symbol);
  //     remaining = remaining - 1 - SYM_SIZE;
  //   }

  //   counter++;
  //   *counts = *counts + SYM_SIZE;

  //   if (remaining <= 0)
  //     break;

  //   read_data_from_buf(input_buf, &cmark, 1);
  // }

  // free(context);
}

__host__
void host_compress(const char *input_data,
                   Buffer *out_buf,
                   long *counts,
                   const CompDescriptor *desc) {
  int m, hit_index;
  char hit_index_m, symbol;
  char entries[E_LENGTH] = {0};

  Context *context = malloc_ase_context(desc->global_counter);

  for (int i = 0; i < desc->chunk_size; i++) {
    symbol = input_data[i];
    hit_index = push(context, entries, symbol);

    if (hit_index == -1) {
      *counts = *counts + 1 + SYM_SIZE;

      // CMark (0) ビットの追加
      write_data_to_buf(out_buf, &CMARK_FALSE, 1);
      write_data_to_buf(out_buf, &symbol, SYM_SIZE);
    } else {
      m = entropy_calc(context);

      hit_index_m = hit_index & ((1 << m) - 1);
      *counts = *counts + 1 + m;

      // CMark (1) ビットの追加
      write_data_to_buf(out_buf, &CMARK_TRUE, 1);
      write_data_to_buf(out_buf, &hit_index_m, m);
    }
  }

  free(context);
}

__host__
void host_decompress(Buffer *input_buf,
                     char *output_data,
                     long *counts,
                     const DecompDescriptor *desc) {
  int m;
  int counter = 0;
  int remaining = desc->counts[0];
  char index_m, cmark, symbol;
  char entries[E_LENGTH] = {0};

  Context *context = malloc_ase_context(desc->global_counter);

  read_data_from_buf(input_buf, &cmark, 1);

  while (true) {
    if (cmark == CMARK_TRUE) {
      m = entropy_calc(context);

      read_data_from_buf(input_buf, &index_m, m);
      symbol = entries[index_m];
      output_data[counter] = symbol;

      arrange_table(context, entries, index_m, symbol);
      remaining = remaining - 1 - m;
    } else {
      read_data_from_buf(input_buf, &symbol, SYM_SIZE);
      output_data[counter] = symbol;

      register_to_table(context, entries, symbol);
      remaining = remaining - 1 - SYM_SIZE;
    }

    counter++;
    *counts = *counts + SYM_SIZE;

    if (remaining <= 0)
      break;

    read_data_from_buf(input_buf, &cmark, 1);
  }

  free(context);
}

// ホスト側で ASE 圧縮の準備を行う. 入力ストリームと ASE 設定プロファイルを PCI 転送でデバイスに
// コピーする.
std::tuple<long*, Buffer**> parallel_compress(const char *input_data,
                                              const CompDescriptor **descs,
                                              const Partition *partition) {
  char *d_input_data;
  long *counts, *d_counts;
  const CompDescriptor **d_descs;
  Buffer **d_out_bufs, **out_bufs;

  const int threads = descs[0]->threads;

  // メモリ確保 (ホスト)
  counts = (long*)malloc(threads * sizeof(long));
  out_bufs = malloc_ase_buffer(threads);

  // メモリ確保 (デバイス)
  cudaMalloc((void**)&d_input_data, D_SIZE);
  cudaMalloc((void**)&d_out_bufs, threads * sizeof(Buffer));
  cudaMalloc((void**)&d_descs, sizeof(CompDescriptor));
  cudaMalloc((void**)&d_counts, threads * sizeof(long));

  // ホストからデバイスにバス転送
  cudaMemcpy(d_input_data, input_data, D_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_descs, descs, sizeof(CompDescriptor), cudaMemcpyHostToDevice);
  cudaMemcpy(d_out_bufs, out_bufs, sizeof(CompDescriptor), cudaMemcpyHostToDevice);

  // カーネル関数呼び出し
  kernel_compress<<<partition->grids, threads>>>(d_input_data, d_out_bufs, d_counts, d_descs);

  // すべてのスレッドが処理を完了するまで待つ
  cudaDeviceSynchronize();

  // デバイスリセット
  resetDevice();

  // デバイスからホストにバス転送
  cudaMemcpy(out_bufs, d_out_bufs, threads * sizeof(Buffer), cudaMemcpyDeviceToHost);
  cudaMemcpy(counts, d_counts, threads * sizeof(long), cudaMemcpyDeviceToHost);

  // メモリ解放 (デバイス)
  cudaFree(d_counts);
  cudaFree(d_descs);
  cudaFree(d_out_bufs);
  cudaFree(d_input_data);

  return {
    counts,
    out_bufs
  };
}

std::tuple<long, char*> parallel_decompress(Buffer **buffer,
                                            const DecompDescriptor **descs,
                                            const Partition *Partition) {
  return {
    0,
    ""
  };
}

std::tuple<long, Buffer*> compress(const char *input_data, const CompDescriptor *desc) {
  long counts = 0;
  Buffer *buffer = malloc_ase_buffer();
  char *output_data = (char*)malloc(D_SIZE * sizeof(char));

  host_compress(input_data, buffer, &counts, desc);

  return {
    counts,
    buffer
  };
}

std::tuple<long, char*> decompress(Buffer *buffer, const DecompDescriptor *desc) {
  long counts = 0;
  char *output_data = (char*)malloc(D_SIZE * sizeof(char));

  host_decompress(buffer, output_data, &counts, desc);

  return {
    counts,
    output_data
  };
}

} // namespace ase
