#include <iostream>
#include <tuple>
#include <stdio.h>
#include "ase.h"
#include "utils.h"

int test_parallel_comp_and_decomp(const char* in_filename,
                                  const char* out_filename,
                                  const int nthreads) {
  size_t nread;
  FILE *file;
  ase::CompDescriptor **comp_descs;
  ase::DecompDescriptor **decomp_descs;
  ase::Buffer *buffer;
  std::tuple<long*, ase::Buffer**>ct;
  std::tuple<long, char*>dt;
  char *input_data, *output_data;
  long bit_size, b_time, a_time, comp_time, decomp_time;
  long comp_total_size = 0;
  long decomp_total_size = 0;

      settings->chunk_size = nread / nthreads;
        ct = ase::parallel_compress(input_data, settings);
        dt = ase::parallel_decompress(buffer, settings);
  // // 圧縮後のビット長を集計する.
  // long total_counts = 0;
  // for (int i = 0; i < threads ; i++)
  //   total_counts += counts[i];

}

int test_comp_and_decomp(const char* in_filename,
                         const char* out_filename) {
  size_t nread;
  FILE *in_file, *out_file;
  ase::CompDescriptor *comp_desc;
  ase::DecompDescriptor *decomp_desc;
  ase::Buffer *buffer;
  std::tuple<long, ase::Buffer*>ct;
  std::tuple<long, char*>dt;
  char *input_data, *output_data;
  long bit_counts, b_time, a_time, comp_time, decomp_time;
  long comp_total_size = 0;
  long decomp_total_size = 0;

  comp_desc = (ase::CompDescriptor*)malloc(sizeof(ase::CompDescriptor));
  comp_desc->entry_size = E_LENGTH;
  comp_desc->global_counter = GLOBAL_C;

  decomp_desc = (ase::DecompDescriptor*)malloc(sizeof(ase::DecompDescriptor));
  decomp_desc->entry_size = E_LENGTH;
  decomp_desc->global_counter = GLOBAL_C;
  decomp_desc->counts = (long*)malloc(sizeof(long));

  input_data = (char*)malloc(D_SIZE * sizeof(char));

  // データチャンクごとに, ファイルシステムから読み込み, 圧縮, 解凍, ファイルシステムへ書き込みを行う.
  in_file = fopen(in_filename, "r");
  out_file = fopen(out_filename, "w+");

  if (in_file && out_file) {
    while ((nread = fread(input_data, 1, D_SIZE * sizeof(char), in_file)) > 0) {
      comp_desc->chunk_size = nread;
      comp_desc->total_size = nread;

      // 圧縮の計測
      b_time = timer(); // 計測開始
      ct = ase::compress(input_data, comp_desc);
      a_time = timer(); // 計測終了

      bit_counts = std::get<0>(ct);
      buffer = std::get<1>(ct);
      comp_time += (a_time - b_time);
      comp_total_size += (bit_counts / 8 + (bit_counts % 8 > 0 ? 1 : 0));

      // 解凍の計測
      decomp_desc->counts[0] = bit_counts;
      b_time = timer(); // 計測開始
      dt = ase::decompress(buffer, decomp_desc);
      a_time = timer(); // 計測終了

      bit_counts = std::get<0>(dt);
      output_data = std::get<1>(dt);
      decomp_time += (a_time - b_time);
      decomp_total_size += (bit_counts / 8 + (bit_counts % 8 > 0 ? 1 : 0));

      // 書き込み
      fwrite(output_data, 1, nread, out_file);
    }
    fclose(out_file);
    fclose(in_file);
  } else {
    fprintf(stderr, "Cannot open this file.\n");
    return 1;
  }

  std::cout << "Filename: "           << in_filename << std::endl;
  std::cout << "Raw: "                << nread << " bytes" << std::endl;
  std::cout << "Compressed: "         << comp_total_size << " bytes" << std::endl;
  std::cout << "Decompressed: "       << decomp_total_size << " bytes" << std::endl;
  std::cout << "Compression rate: "   << ((float)(comp_total_size) / (float)nread) * 100 << "%" << std::endl;
  std::cout << "Compression time: "   << comp_time << " ms" << std::endl;
  std::cout << "Decompression time: " << decomp_time << " ms" << std::endl;

  free(comp_desc);
  free(decomp_desc);
  free(input_data);

  return 0;
}
