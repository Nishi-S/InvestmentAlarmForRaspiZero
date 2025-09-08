#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTest 実行時の一時ディレクトリ/キャプチャ設定を安定化するための sitecustomize。
- 書き込み不可な /tmp 環境向けに TMPDIR をリポジトリ配下へ誘導
- 必要に応じてキャプチャを無効化（-s）し、TemporaryFile を回避

Python は起動時に sitecustomize を自動 import します（CWD が sys.path に含まれる前提）。
"""
import os
import tempfile

try:
    base = os.path.dirname(os.path.abspath(__file__))
except Exception:
    base = os.getcwd()

# 1) TMPDIR をプロジェクト配下に設定
tmpdir = os.path.join(base, ".pytest_tmp")
try:
    os.makedirs(tmpdir, exist_ok=True)
except Exception:
    # 環境で作成できない場合は黙って続行
    pass

if not os.environ.get("TMPDIR"):
    os.environ["TMPDIR"] = tmpdir

# tempfile のデフォルトも上書き（環境に依存せず）
try:
    tempfile.tempdir = os.environ.get("TMPDIR", tmpdir)
except Exception:
    pass

# 2) pytest のキャプチャを抑止（TemporaryFile を使わない）
if "PYTEST_ADDOPTS" not in os.environ:
    os.environ["PYTEST_ADDOPTS"] = "-s"

