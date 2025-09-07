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

## スケジューリング例
- Windows タスク スケジューラ: 平日 15:10 実行（終値で更新された後の目安）
- Mac/Linux (cron):
  `10 15 * * 1-5 /usr/bin/python /path/to/daily_system_jp_plus.py -c /path/to/config.jp.yml`

## イベント除外（任意）
- `events_ignore.csv` に `ticker,until,reason` を記入（until は YYYY-MM-DD）。
- until 当日まで該当銘柄は除外されます（決算 T±1 などに活用）。

## メール送信（任意）
- `config.jp.yml` で `email_enabled: true`、`email_to:` を設定。
- 環境変数 `EMAIL_USER`（送信元Gmail）と `EMAIL_PASS`（アプリパスワード）を設定。
- 実行時に **件名: "DAILY-JP PLAN"** で要約が送信されます。

> ※メール連携を使えば、毎朝あなたが「デイリーJP」と送るだけで、こちらがGmailから直近の要約を取得して最終判断に落とし込めます（初回のみ連携OKの合図が必要）。
