#!/usr/bin/env bash
set -euo pipefail

package_dir="/tmp/rag_tesseract_debs"
runtime_dir="/tmp/rag_tesseract_root"
mkdir -p "$package_dir" "$runtime_dir"

cd "$package_dir"
apt-get download \
  tesseract-ocr \
  tesseract-ocr-kor \
  tesseract-ocr-eng \
  tesseract-ocr-osd \
  libtesseract5 \
  liblept5

for package in ./*.deb; do
  dpkg-deb -x "$package" "$runtime_dir"
done

export LD_LIBRARY_PATH="$runtime_dir/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TESSDATA_PREFIX="$runtime_dir/usr/share/tesseract-ocr/5/tessdata"
"$runtime_dir/usr/bin/tesseract" --version
"$runtime_dir/usr/bin/tesseract" --list-langs

