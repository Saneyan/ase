/**
 * Library of Adaptive Stream-based Entropy Coding (ASE-Coding) for CUDA GPUs
 * 
 * .cpp/.cc ソースコード向けヘッダファイル.
 * 
 * Author: Saneyuki Tadokoro (201311374)
 */
#include <tuple>

// 標準の入力ストリームのバッファサイズ
#define D_SIZE 50 * 1024 * 1024
// 標準のルックアップテーブルのエントリー数
#define E_LENGTH 8
// 標準のエントロピーカリングカウント
#define GLOBAL_C 4
// データシンボルのビットサイズ (= 1 byte)
#define SYM_SIZE 8
// 圧縮データ単位あたりのビットサイズ (= 1 byte)
#define D_MAX_WIDTH 8

namespace ase {

// 圧縮ディスクリプタ. ホストまたはカーネルで圧縮パラメータとして利用する.
// 構造体のメンバは, 一度セットしたら読み取りのみ (readonly) として取り扱うこと.
struct CompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  int chunk_size;     // 1スレッドに割り当てられるデータチャンクのサイズ (readonly)
  int total_size;     // 圧縮前データのサイズ (readonly)
};

// 解凍ディスクリプタ. ホストまたはカーネルで解凍パラメータとして利用する.
struct DecompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  long counts;        // ビットのカウント数 (readonly)
};

// データバッファノード
struct Data {
  char data;          // データの実体
  struct Data* next;  // 次のノード
};

// データバッファ (Linked list)
// 通常実行モード (CPU)で圧縮・解凍を行う場合に用いる.
struct Buffer {
  unsigned int h_offset;  // 1 ノードに対して書き込んだビット数 (head)
  unsigned int t_offset;  // (tail)
  unsigned int max_width; // データの最大の長さ
  struct Data* head;      // 先頭ノード
  struct Data* current;   // 現在のノード
};

// グリッドごとに割り当てられる.
// データチャンクごとに異なる ASE パラメータを割り当てるときに利用する.
struct PartitionAllocation {
  int entry_size;                   // ルックアップテーブルのエントリー数 (readonly)
  int global_counter;               // グローバルカウンター (readonly)
  int num_target_blocks;            // 対象となるブロック数 (readonly, optional)
  int *target_block_ids;            // 対象となるブロック ID (readonly, optional)
};

// 圧縮・解凍データのパーティショニングに必要なディスクリプタ.
struct Partition {
  int num_blocks;                   // データブロック数 (readonly, optional)
  int num_allocations;              // アロケーション数 (readonly)
  PartitionAllocation *allocations; // (readonly)
};

// 圧縮ディスクリプタ. ホストまたはカーネルで圧縮パラメータとして利用する.
// 構造体のメンバは, 一度セットしたら読み取りのみ (readonly) として取り扱うこと.
struct ParallelCompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  int chunk_size;     // 1スレッドに割り当てられるデータチャンクのサイズ (readonly)
  int total_size;     // 圧縮前データのサイズ (readonly)
  int output_size;
  int num_blocks;     // ブロックサイズ (readonly)
};

// 解凍ディスクリプタ. ホストまたはカーネルで解凍パラメータとして利用する.
struct ParallelDecompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  int num_blocks;     // ブロックサイズ (readonly)
  long *counts;       // ビットのカウント数 (readonly)
};

// メモリプール情報
// 並列実行モード (GPU) で圧縮・解凍を行う場合に用いる.
struct PoolInfo {
  unsigned int h_offset;  // 1 バイトあたりの領域に対して書き込んだビット数 (head)
  unsigned int t_offset;  // (tail)
  unsigned int max_width; // データの最大の長さ
  unsigned int index;     // 現在の参照インデックス
  long counts;            // 書き込んだビットのカウント数
};

struct TargetBlock {
  int allocation_id;
  int target_block_id;
};

ParallelCompDescriptor* malloc_parallel_comp_descriptors(const Partition *partition, int nread);

ParallelDecompDescriptor* malloc_parallel_decomp_descriptors(const Partition *partition, long *counts);

// ASE 圧縮を行うホスト関数
std::tuple<long, Buffer*> compress(const char *input_data, const CompDescriptor *descs);

// ASE 圧縮を行うホスト関数
std::tuple<long, char*> decompress(Buffer *buffer, const DecompDescriptor *descs);

// 並列 ASE 圧縮を行うカーネル関数
std::tuple<PoolInfo*, char*> parallel_compress(const char *input_data,
                                              const ParallelCompDescriptor *desc,
                                              const Partition *partition);

// 並列 ASE 圧縮を行うカーネル関数
std::tuple<long, char*> parallel_decompress(Buffer *buffer,
                                            const ParallelDecompDescriptor *desc,
                                            const Partition *partition);

} // namespace ase
