#include "ase.h"
#include "test.h"

int main(int argc, char **argv) {
  const char in_filename[] = "../assets/beauty.bmp";
  const char out_filename[] = "../result.bmp";
  const bool parallel_mode = false;
  const int nthreads = 1000;
  const ase::Partition comp_partition = { 1, 1000 };

  // test_comp_and_decomp(in_filename, out_filename, parallel_mode, nthreads);
  test_comp_and_decomp(in_filename, out_filename, false, &comp_partition, &comp_partition);

  return 0;
}
