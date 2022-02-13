/**
 * Library of Adaptive Stream-based Entropy Coding (ASE-Coding) for CUDA GPUs
 * 
 * .cpp/.cc ソースコード向けヘッダファイル.
 * 
 * Author: Saneyuki Tadokoro (201311374)
 */

#include <tuple>

// 入力ストリームのバッファサイズ
#define D_SIZE 50 * 1024 * 1000
// ルックアップテーブルのエントリー数
#define E_LENGTH 8
// エントロピーカリングカウント
#define GLOBAL_C 4
#define SYM_SIZE 8
#define D_MAX_WIDTH 8

namespace ase {

// ASE ディスクリプタ. ホストとカーネルで圧縮パラメータとして利用する.
// 構造体のメンバは, 一度セットしたら読み取りのみ (readonly) として取り扱うこと.
struct CompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  int chunk_size;     // 1スレッドに割り当てられるデータチャンクのサイズ (readonly)
  int total_size;     // 圧縮前データのサイズ (readonly)
  int threads;        // スレッド数 (readonly)
};

struct DecompDescriptor {
  int entry_size;     // エントリー数 (readonly)
  int global_counter; // グローバルカウンター (readonly)
  int threads;        // スレッド数 (readonly)
  long *counts;       // ビットのカウント数 (readonly)
};

// データバッファノード
struct Data {
  char data;          // データの実体
  struct Data* next;  // 次のノード
};

// データバッファ (Linked list)
struct Buffer {
  unsigned int h_offset;  // 1 ノードに対して書き込んだビット数 (head)
  unsigned int t_offset;  // (tail)
  unsigned int max_width; // データの最大の長さ
  struct Data* head;      // 先頭ノード
  struct Data* current;   // 現在のノード
};

// ASE 圧縮を行うホスト関数
std::tuple<long, Buffer*> compress(const char *input_data, const CompDescriptor *descs);

// ASE 圧縮を行うホスト関数
std::tuple<long, char*> decompress(Buffer *buffer, const DecompDescriptor *descs);

// 並列 ASE 圧縮を行うカーネル関数
std::tuple<long*, Buffer**> parallel_compress(const char *input_data, const CompDescriptor **desc);

// 並列 ASE 圧縮を行うカーネル関数
std::tuple<long, char*> parallel_decompress(Buffer **buffer, const DecompDescriptor **desc);

} // namespace ase