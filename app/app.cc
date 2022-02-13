#include "test.h"

int main(int argc, char **argv) {
  const char in_filename[] = "../assets/beauty.bmp";
  const char out_filename[] = "../result.bmp";
  const bool parallel_mode = false;
  const int nthreads = 1000;

  test_comp_and_decomp(in_filename, out_filename, parallel_mode, nthreads);

  return 0;
}
