# InvestmentAlarmForRaspiZero

日本株の翌営業日用プラン出力（デイリー）と、長期投資向けアラーム監視を行う小型ツール群です。Raspberry Pi/Ubuntu 等の常時起動環境での自動実行を想定しています。

注意: 本リポジトリは教育・情報提供のみを目的としています。実売買は自己責任で、必ずご自身で検証してください。

**主な構成**
- パッケージ: `src/investment_alarm/`（デイリー/アラーム/通知/ステータス/データ取得）
- デイリー選定: `src/investment_alarm/daily_system_jp_plus.py`
- アラーム監視: `src/investment_alarm/investment_alarms.py`
- 設定: `config/config.jp.yml`, `config/config.alarms.yml`, `config/events_ignore.csv`
- エントリ: `bin/daily-jp`, `bin/investment-alarms`
- systemdユニット: `systemd/*.service`, `systemd/*.timer`

## 機能概要
- 日本株ユニバースに対し、指標計算・スコア合成で翌営業日の候補と株数を算出
  - 指標: EMA20/50, SMA100, ATR(14), Donchian20, Bollinger %B, 出来高レシオ(VOLR), OBV傾き, RSI(2)
  - ハードフィルタ: 20日平均売買代金の下限、イベント除外（`events_ignore.csv`）
  - スコア合成: Trend/Momentum/Volume/Breakout/Structure（重み調整可）
  - リスクオン判定: `benchmark` のSMA傾きと位置で regime 判定
- 長期アラーム（米10年金利/VIX/USDJPY/比率クロス等）を監視し、状態変化のみ通知
- 生成物の可視化: 最新要約の `status.html` とメトリクス `metrics.prom`
- Gmail送信（任意）: `EMAIL_USER`/`EMAIL_PASS`（アプリパスワード）
- 休日スキップ: 日本/米国の取引日判定（ライブラリが無い場合は平日判定）

## 必要要件
- Python 3.10/3.11
- 依存: `requirements.txt`（`numpy`, `pandas`, `PyYAML`, `yfinance` ほか）
- ネットワーク（初回DL時）。以降はキャッシュで補助

## セットアップ
1) 仮想環境と依存の導入
- `python -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- （推奨）パッケージを開発インストール: `pip install -e .`

2) 設定ファイルを編集
- `config.jp.yml`: ユニバース/資金/ロット/売買代金しきい値/重み 等
- `config.alarms.yml`: しきい値/比率シグナル/状態管理/同報設定 等

3) メール（任意）
- `/etc/jpplus.env` に環境変数を設定（権限600推奨）
  - `EMAIL_USER=your.name@gmail.com`
  - `EMAIL_PASS=your_app_password`
  - `LOG_LEVEL=INFO`（任意: `DEBUG/INFO/WARN/ERROR`）

## 使い方（手動実行）
- いずれかの方法で実行してください。
  - 例1（推奨・簡単）: `./bin/daily-jp -c config/config.jp.yml` / `./bin/investment-alarms -c config/config.alarms.yml`
  - 例2（開発インストール後）: `python -m investment_alarm.daily_system_jp_plus -c config/config.jp.yml`
    / `python -m investment_alarm.investment_alarms -c config/config.alarms.yml`
  - 例3（環境変数で実行）: `PYTHONPATH=src python -m investment_alarm.daily_system_jp_plus -c config/config.jp.yml`
    / `PYTHONPATH=src python -m investment_alarm.investment_alarms -c config/config.alarms.yml`

出力（既定）:
- デイリー: `reports/plan_*.csv`, `reports/ranks_*.csv`, `reports/plan_latest.json`, `reports/status.html`, `reports/metrics.prom`
- アラーム: `reports_alarms/alarms_*.csv`, `reports_alarms/alarms_latest.json`, `reports_alarms/status.html`, `reports_alarms/metrics.prom`

## systemd による自動実行（root）
本リポジトリ同梱のユニットは root 実行を前提にしています。`WorkingDirectory` と `ExecStart` を実際のパスに合わせて必要なら編集してください。

1) ユニット展開（root）
- `sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/`
- 例: `systemd/daily-jp.service` は以下を参照
  - `WorkingDirectory=/home/nishi/InvestmentAlarmForRaspiZero`
  - `ExecStart=/home/nishi/InvestmentAlarmForRaspiZero/.venv/bin/python -u -m investment_alarm.daily_system_jp_plus -c config/config.jp.yml`

2) タイムゾーン（JST）設定（任意）
- `sudo timedatectl set-timezone Asia/Tokyo`
- 実行時刻: デイリー=平日 15:10 / アラーム=平日 08:05（`Persistent=true` でオフライン時補完）

3) 有効化と起動
- `sudo systemctl daemon-reload`
- `sudo systemctl enable --now daily-jp.timer investment-alarms.timer`

4) ログ確認
- `sudo journalctl -u daily-jp.service -f`
- `sudo journalctl -u investment-alarms.service -f`

メモ:
- ハードニング: `ProtectSystem=full`, `PrivateTmp=yes`, `NoNewPrivileges=yes` 等を既定で付与
- 出力の所有権: root 実行のため生成物は root:root 所有（`UMask=0077`）

## 設定リファレンス（抜粋）

`config.jp.yml`
- `universe`: 対象銘柄リスト
- `benchmark`: レジーム判定用（例: `1306.T`）
- `risk_on_assets`/`defensive_assets`: レジーム別プール（オフ時は休む想定）
- `data_days`: 取得期間（日）
- `regime_ma`: レジーム用SMA日数
- `atr_period`/`atr_mult_stop`: ATRと損切り幅
- `top_k`: 採用件数
- `capital`/`risk_per_trade`: 資金・1トレード当たりリスク
- `min_turnover_jpy`: 20日平均売買代金の下限
- `lot_size_default`/`lot_size_map`: 売買単位（1株/100株など）
- `events_csv`: `ticker,until,reason`（YYYY-MM-DD）
- `reentry_cooloff_days`: 再エントリー抑止（日）。履歴は `reports/picks_history.json`
- `min_notional_jpy`/`notional_cap_per_ticker_jpy`: 最低約定金額/銘柄ごとの上限
- `hold_days`/`trailing_stop_mult`: エグジット表現
- `holiday_skip_jp`: 日本の休日スキップ（`jpholiday` があれば祝日除外）
- `email_*`: メール通知の有効化と宛先
- `w_trend/momo/volume/breakout/structure`: スコア重み

`config.alarms.yml`
- `us10y_*`/`vix_*`/`usdjpy_*`: しきい値
- `ratio_short/ratio_long/ratio_sma_days`: 比率クロス
- `ratio_signals`: 追加比率シグナル配列
- `spy_sma200_down`/`topix_t1306_sma200_down`: ベースライン判定
- `holiday_skip_us`: 米国市場の休日スキップ（`pandas-market-calendars` があれば取引所休場）
- `notify_on_change_only`/`cooldown_hours`/`state_file`: 状態変化のみ通知とクールダウン
- `digest_include_daily`/`daily_latest_json`: デイリー要約を同報
- `email_*`: メール通知の有効化と宛先

## イベント除外
- `events_ignore.csv` に `ticker,until,reason` を記入（until は `YYYY-MM-DD`）
- 指定日まで該当銘柄は候補から除外

## トラブルシューティング
- 起動しない/即終了する
  - `WorkingDirectory`/`ExecStart` のパス不一致を確認
  - 仮想環境 `.venv` が存在し、実行可能か確認
  - ネットワーク不通時はキャッシュが使われますが、キャッシュも無い銘柄は警告ログになります
- メールが送れない
  - `EMAIL_USER`/`EMAIL_PASS` の未設定、もしくはアプリパスワード未許可
  - `email_enabled: true` と `email_to:` の設定
- 休日スキップが意図通りでない
  - 日本: `jpholiday`、米国: `pandas-market-calendars` の有無を確認

## 開発
- テスト: `pytest -q`（ネットワーク不要）
- CI: GitHub Actions（Python 3.10/3.11）
- ログ: `LOG_LEVEL` で調整（既定 `INFO`）

--
US10Y は Yahoo の `^TNX` を使用し、値はパーセント換算のため `Close/10` として扱います。VIX は `^VIX`、USDJPY は `USDJPY=X` を使用します。比率は終値（`Adj Close` があれば優先）で計算します。
