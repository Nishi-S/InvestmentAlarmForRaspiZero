# Daily Trading Helper JP PLUS（1日サイクル）

## できること
- 日本株ユニバース（東京エレクトロン除外）を対象に、**翌営業日の候補**と**株数（ATR×ストップ前提）**を自動算出
- 指標: EMA20/50, SMA100, ATR(14), Donchian20, Bollinger %B, 出来高レシオ, OBV傾き, RSI(2)
- ハードフィルター: 20日平均**売買代金**の下限、イベント除外（CSV）
- スコア合成: トレンド/モメンタム/出来高/ブレイクアウト/構造（重みは設定可能）

## 使い方
1. `pip install pandas numpy yfinance pyyaml`
2. `config.jp.yml` を必要に応じて編集（資金、ロット、売買代金しきい値 等）
3. 実行: `python daily_system_jp_plus.py -c config.jp.yml`
4. 出力: `reports/plan_*.csv`, `reports/ranks_*.csv`, `reports/plan_latest.json`

## スケジューリング（systemd）
ユーザー単位の systemd タイマーでの実行を想定しています（Raspberry Pi/Ubuntu 等）。

1) ユニットを配置（ユーザー単位）
- `mkdir -p ~/.config/systemd/user`
- `cp systemd/daily-jp.service systemd/daily-jp.timer ~/.config/systemd/user/`
- `cp systemd/investment-alarms.service systemd/investment-alarms.timer ~/.config/systemd/user/`

2) パス調整（必要に応じて）
- `systemd/*.service` の `WorkingDirectory` と `ExecStart` を、リポジトリの場所に合わせて編集
  - 例: `%h/InvestmentAlarmForRaspiZero/.venv/bin/python ...`

3) 有効化＆起動
- `systemctl --user daemon-reload`
- `systemctl --user enable --now daily-jp.timer investment-alarms.timer`

4) ログ確認
- `journalctl --user -u daily-jp.service -f`
- `journalctl --user -u investment-alarms.service -f`

補足:
- タイムゾーン（JST）で実行したい場合は OS 側を `Asia/Tokyo` に設定してください。
  - `sudo timedatectl set-timezone Asia/Tokyo`
  - `daily-jp.timer`: 平日 15:10 / `investment-alarms.timer`: 平日 08:05 に実行
  - オフライン時も再実行するため `Persistent=true` を設定済み
  - ユニットは hardening 設定（`ProtectSystem=full` 等）と `Restart=on-failure` を付与済み

## イベント除外（任意）
- `events_ignore.csv` に `ticker,until,reason` を記入（until は YYYY-MM-DD）。
- until 当日まで該当銘柄は除外されます（決算 T±1 などに活用）。

## メール送信（任意）
- `config.jp.yml` で `email_enabled: true`、`email_to:` を設定。
- 環境変数 `EMAIL_USER`（送信元Gmail）と `EMAIL_PASS`（アプリパスワード）を設定。
- 実行時に **件名: "DAILY-JP PLAN"** で要約が送信されます。

> ※メール連携を使えば、毎朝あなたが「デイリーJP」と送るだけで、こちらがGmailから直近の要約を取得して最終判断に落とし込めます（初回のみ連携OKの合図が必要）。

### 追加オプション（デイリー）
- `--outdir`: 出力先を上書き
- `--no-email`: メール送信を抑止
- `--dry-run`: 書き込みと通知を抑止（計算のみ）
- 休日スキップ: `holiday_skip_jp: true`（`jpholiday` があれば日本の祝日も除外）
- キャッシュ: `cache_dir`, `cache_max_age_days`（`yfinance` の取得を補助）
- サイジング: `min_notional_jpy`, `notional_cap_per_ticker_jpy`, `commission_per_trade_jpy`
- 再エントリー抑止: `reentry_cooloff_days`（`reports/picks_history.json` を参照）
- エグジット表現: `hold_days`, `trailing_stop_mult`（後者>0でトレール表記）

## 長期投資向けアラーム（追加機能）
- 対象アラーム（デフォルト値）
  - 米10年金利（US10Y）: 4.60% 上抜け（Hard）／ 3.90% 割れ（Hard）
  - VIX: 22 超（Soft）／ 28 超（Hard）
  - USD/JPY: 145.00 円割れ（円高加速）
  - SMH/XLK 比率: 50 日線割れ（終値ベース）

### 使い方（アラーム）
1. `config.alarms.yml` を必要に応じて編集（しきい値、メール宛先 等）
2. 実行: `python investment_alarms.py -c config.alarms.yml`
3. 出力: `reports_alarms/alarms_*.csv`, `reports_alarms/alarms_latest.json`

### 追加機能（アラーム）
- 状態管理: `notify_on_change_only`, `cooldown_hours`, `state_file`
- 休日スキップ（米国）: `holiday_skip_us`（`pandas-market-calendars` があれば取引所休場）
- 比率/ベースラインの拡張: `ratio_signals`, `spy_sma200_down`, `topix_t1306_sma200_down`
- デイリー要約の同報: `digest_include_daily: true`（`reports/plan_latest.json` を取り込み）
- キャッシュ: `cache_dir`, `cache_max_age_days`

### 追加オプション（アラーム）
- `--outdir`, `--no-email`, `--dry-run` をサポート

実行後、`reports/` および `reports_alarms/` に `status.html` と `metrics.prom`（Prometheus テキストフォーマット）が生成されます。

メモ:
- US10Y は Yahoo の `^TNX` を使用し、値はパーセント換算のため `Close/10` として扱います。
- VIX は `^VIX`、USDJPY は `USDJPY=X` を使用します。
- SMH/XLK は終値比率（Adj Close があれば優先）とその 50 日移動平均のデッドクロスで判定します。

### システム全体のユニットとして使う場合（任意）
1) `sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/`
2) `sudo systemctl daemon-reload`
3) `sudo systemctl enable --now daily-jp.timer investment-alarms.timer`
4) ログ: `sudo journalctl -u daily-jp.service -f`

### メール送信（任意）
- `config.alarms.yml` で `email_enabled: true`、`email_to:` を設定。
- 環境変数 `EMAIL_USER`（送信元Gmail）と `EMAIL_PASS`（アプリパスワード）を設定。
- アラームがトリガーされた内容のサマリをメール送信します（トリガーが無い場合は「No new triggers.」）。

## 開発
- 依存関係: `requirements.txt`
- テスト: `pytest -q`（ネットワークアクセス不要）
- CI: GitHub Actions（`python 3.10/3.11`）でテスト実行
