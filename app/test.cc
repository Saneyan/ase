#include <iostream>
#include <tuple>
#include <stdio.h>
#include "ase.h"
#include "utils.h"
#include "test.h"

int test_comp_and_decomp(const char* in_filename,
                         const char* out_filename,
                         const bool parallel_mode,
                         const ase::Partition *comp_partition) {
  size_t nread;
  FILE *in_file, *out_file;
  ase::PartitionAllocation *allocations;
  ase::CompDescriptor *comp_desc;
  ase::DecompDescriptor *decomp_desc;
  ase::ParallelCompDescriptor *comp_descs;
  ase::ParallelDecompDescriptor *decomp_descs;
  ase::Buffer *buffer;
  ase::PoolInfo *pi;
  std::tuple<long, ase::Buffer*>ct;
  std::tuple<ase::PoolInfo*, char*>cts;
  std::tuple<long, char*>dt;
  std::tuple<ase::PoolInfo*, char*>dts;
  int i, total_size, chunk_size, threads;
  char *input_data, *output_data;
  char *p_data;
  long bit_counts, *d_bit_counts, b_time, a_time;
  long total_counts;
  long comp_time = 0;
  long decomp_time = 0;
  long read_total_size = 0;
  long comp_total_size = 0;
  long decomp_total_size = 0;

  // データチャンクヒープ
  input_data = (char*)malloc(D_SIZE * sizeof(char));

  // データチャンクごとに, ファイルシステムから読み込み, 圧縮, 解凍, ファイルシステムへ書き込みを行う.
  in_file = fopen(in_filename, "r");
  out_file = fopen(out_filename, "w+");

  if (in_file && out_file) {
    // 1. ファイルシステムから読み込み
    while ((nread = fread(input_data, 1, D_SIZE * sizeof(char), in_file)) > 0) {
      read_total_size += nread;

      // 並列実行モード (GPU 処理) の場合
      if (parallel_mode) {
        // 2. 圧縮
        comp_descs = malloc_parallel_comp_descriptors(comp_partition, nread);

        b_time = timer(); // 計測開始
        cts = ase::parallel_compress(input_data, const_cast<const ase::ParallelCompDescriptor*>(comp_descs), comp_partition);
        a_time = timer(); // 計測終了

        pi = std::get<0>(cts);
        p_data = std::get<1>(cts);
        comp_time += (a_time - b_time);

        total_counts = 0;
        for (i = 0; i < comp_partition->num_blocks; i++)
          total_counts += pi[i].write_counts;

        comp_total_size += (total_counts / 8 + (total_counts % 8 > 0 ? 1 : 0));

        // 3. 解凍
        // 圧縮時のディスクリプタ情報を元に, 解凍用のパーティションを作成.
        allocations = (ase::PartitionAllocation*)malloc(comp_partition->num_allocations * sizeof(ase::PartitionAllocation));
        for (i = 0; i < comp_partition->num_allocations; i++)
          allocations[i] = ase::PartitionAllocation{ E_LENGTH, GLOBAL_C, NULL };
        ase::Partition *decomp_partition = (ase::Partition*)malloc(sizeof(ase::Partition));
        *decomp_partition = ase::Partition{ comp_partition->num_blocks, comp_partition->num_allocations, allocations };

        decomp_descs = malloc_parallel_decomp_descriptors(decomp_partition, nread);

        b_time = timer(); // 計測開始
        dts = ase::parallel_decompress(p_data, pi, const_cast<const ase::ParallelDecompDescriptor*>(decomp_descs), decomp_partition);
        a_time = timer(); // 計測終了

        pi = std::get<0>(dts);
        output_data = std::get<1>(dts);
        decomp_time += (a_time - b_time);

        total_counts = 0;
        for (i = 0; i < comp_partition->num_blocks; i++)
          total_counts += pi[i].read_counts;

        decomp_total_size += (total_counts / 8 + (total_counts % 8 > 0 ? 1 : 0));

        free(comp_descs);
        free(decomp_descs);

      // 通常実行モード (CPU 処理) の場合
      } else {
        // 2. 圧縮
        comp_desc = (ase::CompDescriptor*)malloc(sizeof(ase::CompDescriptor));
        comp_desc->entry_size = E_LENGTH;
        comp_desc->global_counter = GLOBAL_C;
        comp_desc->chunk_size = nread;
        comp_desc->total_size = nread;

        b_time = timer(); // 計測開始
        ct = ase::compress(input_data, comp_desc);
        a_time = timer(); // 計測終了

        bit_counts = std::get<0>(ct);
        buffer = std::get<1>(ct);
        comp_time += (a_time - b_time);
        comp_total_size += (bit_counts / 8 + (bit_counts % 8 > 0 ? 1 : 0));

        // 3. 解凍
        decomp_desc = (ase::DecompDescriptor*)malloc(sizeof(ase::DecompDescriptor));
        decomp_desc->entry_size = E_LENGTH;
        decomp_desc->global_counter = GLOBAL_C;
        decomp_desc->counts = bit_counts;

        b_time = timer(); // 計測開始
        dt = ase::decompress(buffer, decomp_desc);
        a_time = timer(); // 計測終了

        bit_counts = std::get<0>(dt);
        output_data = std::get<1>(dt);
        decomp_time += (a_time - b_time);
        decomp_total_size += (bit_counts / 8 + (bit_counts % 8 > 0 ? 1 : 0));

        free(comp_desc);
        free(decomp_desc);
      }

      // 4. 書き込み
      fwrite(output_data, 1, nread, out_file);
    }
    fclose(out_file);
    fclose(in_file);
  } else {
    fprintf(stderr, "Cannot open this file.\n");
    return 1;
  }

  std::cout << "Filename: "           << in_filename << std::endl;
  std::cout << "Raw: "                << read_total_size << " bytes" << std::endl;
  std::cout << "Compressed: "         << comp_total_size << " bytes" << std::endl;
  std::cout << "Decompressed: "       << decomp_total_size << " bytes" << std::endl;
  std::cout << "Compression rate: "   << ((float)(comp_total_size) / (float)read_total_size) * 100 << "%" << std::endl;
  std::cout << "Compression time: "   << comp_time << " ms" << std::endl;
  std::cout << "Decompression time: " << decomp_time << " ms" << std::endl;

  free(input_data);

  return 0;
}
