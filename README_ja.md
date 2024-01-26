# ASE Coding

このプロジェクトは, ASE Coding のリファレンス実装である. CPU および GPU (CUDA) の実行環境で動作する.

### Requirements 

* Linux
* CUDA: `>= 3`
* CMake: `>= 3.10`
* C++14 に対応している C++ コンパイラ

### Build

build ディレクトリを作成し, CMake と make を実行してコンパイルする.
コンパイル後は, build ディレクトリに実行可能なファイル app が作成されているので, このファイルを実行すること.

```bash
$ mkdir build/; cd build/
$ cmake ..
$ make
$ ./app
```

### Project Structure

* `app/`: ライブラリを用いた圧縮・解凍のプログラムに関するファイル群
* `lib/`: ASE Coding のライブラリ
* `old/`: 不要になった過去のソースコード

### Version

#### v0.2.0
* GPU 実行環境で動作する圧縮機および解凍機を作成

#### v0.1.0
* ビルドシステム CMake を導入

#### v0.0.1
* CPU 実行環境で動作する圧縮機および解凍機を作成
