#!/usr/bin/env bash
# gym 0.19.0 의 setup.py 에 'opencv-python>=3.' (잘못된 버전 표기)가 있어서 최신 setuptools 에서 설치 실패함.
# sdist 받아 한 글자 고친 뒤 설치. 코랩 노트북 첫 셀이 이걸 부름.
# 사용법: bash camsim/scripts/install_gym019.sh [python 실행 파일]   (기본: python)
set -euo pipefail
PY="${1:-python}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fsSL https://files.pythonhosted.org/packages/source/g/gym/gym-0.19.0.tar.gz -o "$TMP/gym.tgz"
tar xzf "$TMP/gym.tgz" -C "$TMP"
sed -i 's/opencv-python>=3\./opencv-python>=3/' "$TMP/gym-0.19.0/setup.py"
"$PY" -m pip install --no-deps --no-build-isolation "$TMP/gym-0.19.0"
