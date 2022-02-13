#include <iostream>

// UNIX 時間のミリ秒を返す.
long timer() {
  struct timespec ts;
	struct tm tm;

  clock_gettime(CLOCK_REALTIME, &ts);
  // ローカル時間の設定
  localtime_r(&ts.tv_sec, &tm);

  return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}
