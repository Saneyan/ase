#include "ase.h"
#include "test.h"

int main(int argc, char **argv) {
  const char in_filename[] = "../assets/beauty.bmp";
  const char out_filename[] = "../result.bmp";
  const bool parallel_mode = false;
  ase::PartitionAllocation allocations[1] = {
    ase::PartitionAllocation{ 1000, E_LENGTH, GLOBAL_C }
  };
  const ase::Partition comp_partition = { 1, allocations };

  test_comp_and_decomp(in_filename, out_filename, true, &comp_partition);

  return 0;
}
