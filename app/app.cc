#include "ase.h"
#include "test.h"

#define NUM_ALLOC 1

int main(int argc, char **argv) {
  const char in_filename[] = "../assets/beauty.bmp";
  const char out_filename[] = "../result.bmp";
  const bool parallel_mode = false;
  ase::PartitionAllocation allocations[NUM_ALLOC] = {
    ase::PartitionAllocation{ E_LENGTH, GLOBAL_C, NULL }
  };
  const ase::Partition comp_partition = { 1024, NUM_ALLOC, allocations };

  test_comp_and_decomp(in_filename, out_filename, true, &comp_partition);

  return 0;
}
