#include "ase.h"
#include "test.h"

#define NUM_ALLOC 1

int main(int argc, char **argv) {
  // 入力データファイル名
  const char in_filename[] = "../assets/beauty.bmp";
  // 出力データファイル名
  const char out_filename[] = "../result.bmp";
  // 並列実行モードの設定 (false の場合は CPU での通常実行モード)
  const bool parallel_mode = true;
  // アロケーションの設定
  ase::PartitionAllocation allocations[NUM_ALLOC] = {
    ase::PartitionAllocation{ E_LENGTH, GLOBAL_C, NULL }
  };
  const ase::Partition comp_partition = { 1024, NUM_ALLOC, allocations };

  test_comp_and_decomp(in_filename, out_filename, parallel_mode, &comp_partition);

  return 0;
}
