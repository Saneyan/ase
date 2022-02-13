/**
 * Implementation of Adaptive Stream-based Entropy Coding (ASE-Coding) for CPUs.
 * 
 * このプログラムは, ASE-Coding のリファレンス実装である.
 * 詳細は GPU 版のソースコードをご覧いただきたい.
 * 
 * Date: 2022/2/2
 * Author: Saneyuki Tadokoro (201311374)
 * Version: v0.0.1
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef char *ase_sym;

#define D_SIZE 50 * 1024 * 1000
#define E_LENGTH 8
#define GLOBAL_C 4
#define SYM_SIZE 8
#define D_MAX_WIDTH 8

// ref: https://stackoverflow.com/questions/111928/is-there-a-printf-converter-to-print-in-binary-format
// printf("Leading text "BYTE_TO_BINARY_PATTERN, BYTE_TO_BINARY(*data));
#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(byte)  \
  (byte & 0x80 ? '1' : '0'), \
  (byte & 0x40 ? '1' : '0'), \
  (byte & 0x20 ? '1' : '0'), \
  (byte & 0x10 ? '1' : '0'), \
  (byte & 0x08 ? '1' : '0'), \
  (byte & 0x04 ? '1' : '0'), \
  (byte & 0x02 ? '1' : '0'), \
  (byte & 0x01 ? '1' : '0')

static const char CMARK_TRUE = 1;
static const char CMARK_FALSE = 0;

struct ase_settings {
  int entry_size;
  int global_counter;
  int chunk_size;
  int total_size;
  int bit_size;
};

struct ase_context {
  int occupied;
  int global_counter;
  int culling_num;
};

// データバッファノード
struct ase_data {
  char data;
  struct ase_data* next;
};

// データバッファ (Linked list)
struct ase_buffer {
  unsigned int h_offset;  // 1 ノードに対して書き込んだビット数
  unsigned int t_offset;
  unsigned int max_width; // データの最大の長さ
  struct ase_data* head;
  struct ase_data* current;
};

ase_data* malloc_ase_data() {
  ase_data *new_data;

  if ((new_data = (ase_data*)malloc(sizeof(ase_data))) == NULL) {
    fprintf(stderr, "Cannot allocate memory for ase data.\n");
    return NULL;
  }
  new_data->data = 0;
  return new_data;
}

ase_data* malloc_next_ase_data(ase_buffer* buf) {
  ase_data *new_data;

  if ((new_data = malloc_ase_data()) == NULL)
    return NULL;
  buf->current->next = new_data;
  buf->current = new_data;
  buf->t_offset = 0;
  return new_data;
}

ase_buffer* malloc_ase_buffer() {
  ase_buffer *new_buf;
  ase_data *new_data;

  if ((new_buf = (ase_buffer*)malloc(sizeof(ase_buffer))) == NULL) {
    fprintf(stderr, "Cannot allocate memory for ase buffer.\n");
    return NULL;
  }
  if ((new_data = malloc_ase_data()) == NULL)
    return NULL;
  new_buf->head = new_data;
  new_buf->current = new_data;
  new_buf->max_width = D_MAX_WIDTH;
  new_buf->t_offset = 0;
  new_buf->h_offset = 0;
  return new_buf;
}

ase_context* malloc_ase_context(const ase_settings* settings) {
  ase_context *context;

  if ((context = (ase_context *)malloc(sizeof(ase_context))) == NULL) {
    fprintf(stderr, "Cannot allocate memory for ase context.\n");
    return NULL;
  }
  context->occupied = 0;
  context->global_counter = settings->global_counter;
  context->culling_num = settings->global_counter;
  return context;
}

int free_head_ase_buffer(ase_buffer *buf) {
  ase_data *new_head;

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
int data_to_buf(ase_buffer* buf, const char* data, const unsigned int width) {
  int shift = buf->max_width - buf->t_offset - width;

  if (shift < 0) {
    // 入力データに対する残量シフト量, すなわち -1 * shift に等しい.
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

int data_from_buf(ase_buffer* buf, char* data, const unsigned int width) {
  int shift = buf->max_width - buf->h_offset - width;
  int remaining = 0;
  *data = 0;

  if (shift < 0) {
    *data = (buf->head->data & ((unsigned char)0xFF >> buf->h_offset)) << (-1 * shift);

    // 不要バッファ解放
    if (free_head_ase_buffer(buf) == -1) {
      return -1;
    }
    
    remaining = width + shift;
    shift += buf->max_width;
  }
  *data |= (unsigned char)(buf->head->data & ((unsigned char)(0xFF << buf->max_width - (width - remaining)) >> buf->h_offset)) >> shift;
  buf->h_offset = buf->max_width - shift;

  if (buf->h_offset == buf->max_width) {
    if (free_head_ase_buffer(buf) == -1)
      return -1;
  } 

  return 0;
}

void entropy_culling(ase_context *context) {
  if (context->global_counter > 0) {
    context->global_counter--;
  } else {
    if (context->occupied > 0)
      context->occupied--;
    context->global_counter = context->culling_num;
  }
}

void arrange_table(ase_context *context,
                   char *entries,
                   const int hit_index,
                   const char symbol) {
  int i;

  if (context->occupied > 1 && hit_index > 0) {
    for (i = hit_index - 1; i >= 0; i--)
      entries[i + 1] = entries[i];

    entries[0] = symbol;
    entropy_culling(context);
  }
}

void register_to_table(ase_context *context, char *entries, const char symbol) {
  int i;

  for (i = context->occupied - 1; i >= 0; i--) {
    if (i + 1 < E_LENGTH)
      entries[i + 1] = entries[i];
  }
  entries[0] = symbol;
  if (context->occupied + 1 < E_LENGTH)
    context->occupied++;
}

int push(ase_context *context, char *entries, const char symbol) {
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

int entropy_calc(ase_context *context) {
  int m = (int) ceilf(log2f(context->occupied));
  return m == 0 ? 1 : m;
}

void ase_compress(const char *input_data,
                  ase_buffer *out_buf,
                  long *counts,
                  const ase_settings *settings) {
  int m;
  int hit_index;
  char hit_index_m;
  int max_m = 0;
  char symbol;
  char entries[E_LENGTH] = {0};
  ase_context *context = malloc_ase_context(settings);

  for (int i = 0; i < settings->chunk_size; i++) {
    symbol = input_data[i];
    hit_index = push(context, entries, symbol);

    if (hit_index == -1) {
      *counts = *counts + 1 + SYM_SIZE;

      // CMark (0) ビットの追加
      data_to_buf(out_buf, &CMARK_FALSE, 1);
      data_to_buf(out_buf, &symbol, SYM_SIZE);
    } else {
      m = entropy_calc(context);
      if (max_m < m)
        max_m = m;

      hit_index_m = hit_index & ((1 << max_m) - 1);
      *counts = *counts + 1 + max_m;

      // CMark (1) ビットの追加
      data_to_buf(out_buf, &CMARK_TRUE, 1);
      data_to_buf(out_buf, &hit_index_m, max_m);
    }
  }

  free(context);
}

void ase_decompress(ase_buffer *input_buf,
                    char *output_data,
                    long *counts,
                    const ase_settings *settings) {
  int m;
  int counter = 0;
  int remaining = settings->bit_size;
  int index;
  int max_m = 0;
  char index_m, cmark, symbol;
  char entries[E_LENGTH] = {0};
  ase_context *context = malloc_ase_context(settings);

  data_from_buf(input_buf, &cmark, 1);

  while (true) {
    if (cmark == CMARK_TRUE) {
      m = entropy_calc(context);
      if (max_m < m)
        max_m = m;

      data_from_buf(input_buf, &index_m, max_m);
      symbol = entries[index_m];
      output_data[counter] = symbol;

      arrange_table(context, entries, index_m, symbol);
      remaining = remaining - 1 - max_m;
    } else {
      data_from_buf(input_buf, &symbol, SYM_SIZE);
      output_data[counter] = symbol;

      register_to_table(context, entries, symbol);
      remaining = remaining - 1 - SYM_SIZE;
    }

    counter++;
    *counts = *counts + SYM_SIZE;

    if (remaining <= 0) {
      printf("counts: %d\n", *counts);
      printf("remaining: %d\n", remaining);
      break;
    }

    data_from_buf(input_buf, &cmark, 1);
  }

  free(context);
}

long start_compress(const char *input_data, const ase_settings *settings) {
  long counts = 0;
  long out_counts = 0;
  ase_buffer *buffer = malloc_ase_buffer();
  char *output_data = (char*)malloc(D_SIZE * sizeof(char));

  ase_compress(input_data, buffer, &counts, settings);

  ase_settings *o_settings = (ase_settings *)malloc(sizeof(ase_settings));
  o_settings->entry_size = E_LENGTH;
  o_settings->bit_size = counts;
  o_settings->global_counter = GLOBAL_C;


  ase_decompress(buffer, output_data, &out_counts, o_settings);
  FILE *file = fopen("result.iso", "w+");
  fwrite(output_data, 1, settings->total_size, file);
  fclose(file);

  printf("After compressed: %d\n", counts / 8);
  printf("After decompressed: %d\n", out_counts / 8);

  free(buffer);

  return counts / 8 + (counts % 8 > 0 ? 1 : 0);
}

long timer() {
  struct timespec ts;
	struct tm tm;

  clock_gettime(CLOCK_REALTIME, &ts);
  localtime_r(&ts.tv_sec, &tm);
  return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}

int main(int argc, char **argv) {
  char *input_data;
  ase_settings *settings;
  size_t nread;
  long b_time, a_time, total_size;
  const char filename[] = "./test.iso";

  input_data = (char*)malloc(D_SIZE * sizeof(char));

  FILE *file = fopen(filename, "r");
  if (file) {
    while ((nread = fread(input_data, 1, D_SIZE * sizeof(char), file)) > 0) {
      // printf("%zu\n", nread);

      b_time = timer();

      settings = (ase_settings *)malloc(sizeof(ase_settings));
      settings->entry_size = E_LENGTH;
      settings->chunk_size = nread;
      settings->total_size = nread;
      settings->global_counter = GLOBAL_C;

      total_size = start_compress(input_data, settings);

      a_time = timer();

      printf("%s\n", filename);
      printf("Raw: %ld bytes\n", nread);
      printf("Compressed: %ld bytes\n", total_size);
      printf("Compression rate: %f%%\n", ((float)total_size / (float)nread) * 100);
      printf("Timer: %ld msec\n", a_time - b_time);

      free(input_data);
      free(settings);
    }
    fclose(file);
  } else {
    fprintf(stderr, "Cannot open this file.\n");
    return 1;
  }

  return 0;
}
