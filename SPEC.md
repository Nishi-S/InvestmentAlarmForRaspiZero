# 実装済み機能（スペック）

## デイリー選定（日本株）
- 指標/特徴量: EMA20/50, SMA100, ATR(14), Donchian High(20), Bollinger %B, VOLR, OBV傾き(20), RSI(2), 5日/20日騰落率
- スコア合成: Trend/Momentum/Volume/Breakout/Structure の加重平均（重み調整可）
- レジーム判定: ベンチマークSMAの位置＋傾きで Risk ON/OFF（既定: `1306.T`, `regime_ma=100`）
- フィルタ: 20日平均売買代金下限、イベント除外（CSV）、主要MA割れ、レジーム別プール適用
- サイジング/出口: ATR×止め幅で株数算出、ロット丸め（1株/100株）、最低約定金額/銘柄上限、目標/損切り、保有日/トレーリング
- 再エントリー抑止: 採用履歴に基づくクールオフ日数（履歴JSON管理）

## アラーム監視（長期）
- 金利/ボラ/為替: US10Y（`^TNX/10`）上抜け/下抜け、VIX SOFT/HARD上抜け、USDJPY 下抜け
- 比率シグナル: SMH/XLK の SMA50 デッドクロス、任意のショート/ロング比率（複数定義可）
- ベースライン: SPY/1306.T の SMA200 割れチェック（任意）
- 通知抑制: 状態変化時のみ通知、クールダウン時間、状態ファイルで持続管理

## データ取得/キャッシュ
- 価格取得: `yfinance` によるダウンロード（リトライあり）
- キャッシュ: ティッカー別CSV保存・再利用（有効期限/期限切れフォールバック）
- 休日判定: 日本=`jpholiday`（無ければ平日）、米国=`pandas-market-calendars`（無ければ平日）

## 出力/可視化
- デイリー出力: `reports/ranks_*.csv`, `plan_*.csv`, `plan_latest.json`, `picks_history.json`, `metrics.prom`, `status.html`
- アラーム出力: `reports_alarms/alarms_*.csv`, `alarms_latest.json`, `metrics.prom`, `status.html`
- ステータスHTML: デイリー要約とアラーム一覧を1ページで表示

## 通知（メール）
- 条件: デイリーは有効化時送信、アラームは「変更あり」かつ有効化時送信
- 方式: Gmail SMTP（`EMAIL_USER`/`EMAIL_PASS` アプリパス推奨）
- 本文: 人間可読要約＋機械可読YAMLブロック（`jpplan`、採用/却下トップを含む）

## 実行/運用
- CLI: `-c/--config`, `--outdir`, `--no-email`, `--dry-run`
- systemd: 平日スケジュール（デイリー=15:10、アラーム=08:05）、`/etc/jpplus.env` 読込、ハードニング設定

## 設定（主な項目）
- デイリー設定: ユニバース/ベンチマーク、レジームMA、ATR/止め幅、採用数、資金/1トレードリスク、売買代金下限、ロット、イベントCSV、再エントリー抑止、スコア重み
- アラーム設定: 各しきい値、比率シグナル配列/SMA日数、ベースライン有効化、休日スキップ、状態管理（`state_file`/`cooldown_hours`）、デイリー要約同報

## メトリクス/ログ/開発
- メトリクス: ピック数/トリガー数/変更数、スクリプト実行秒
- ログ: `LOG_LEVEL`（`DEBUG/INFO/WARN/ERROR`）
- 必要要件: Python 3.10/3.11、`numpy/pandas/PyYAML/yfinance`（任意: `jpholiday`/`pandas-market-calendars`）
- テスト: `pytest` によるスコア〜採用/サイジング、メール本文、休日判定、アラーム評価の検証

