#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import datetime as dt
from typing import Optional


def _read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def generate_status_html(out_path: str, daily_json_path: Optional[str], alarms_json_path: Optional[str]) -> None:
    daily = _read_json(daily_json_path) if daily_json_path else None
    alarms = _read_json(alarms_json_path) if alarms_json_path else None
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    html = []
    html.append("<html><head><meta charset='utf-8'><title>Status</title>")
    html.append("<style>body{font-family:sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px} h2{margin-top:1.2em}</style>")
    html.append("</head><body>")
    html.append(f"<h1>Investment Status</h1><div>Generated: {esc(now)}</div>")

    html.append("<h2>Daily Plan</h2>")
    if not daily or not daily.get("top"):
        html.append("<div>No picks.</div>")
    else:
        html.append("<table><tr><th>Ticker</th><th>Shares</th><th>Close</th><th>ATR</th><th>Stop</th><th>Target</th><th>Score</th></tr>")
        for r in daily["top"]:
            html.append("<tr>" + "".join([
                f"<td>{esc(str(r.get('ticker')))}</td>",
                f"<td>{esc(str(r.get('shares')))}</td>",
                f"<td>{esc(str(r.get('close_ref')))}</td>",
                f"<td>{esc(str(r.get('atr')))}</td>",
                f"<td>{esc(str(r.get('stop_ref')))}</td>",
                f"<td>{esc(str(r.get('target_ref')))}</td>",
                f"<td>{esc(str(r.get('score_total')))}</td>",
            ]) + "</tr>")
        html.append("</table>")

    html.append("<h2>Alarms</h2>")
    if not alarms or not alarms.get("all"):
        html.append("<div>No alarms data.</div>")
    else:
        html.append("<table><tr><th>Name</th><th>Severity</th><th>Description</th><th>Triggered</th><th>Value</th><th>Threshold</th><th>Time</th></tr>")
        for r in alarms["all"]:
            html.append("<tr>" + "".join([
                f"<td>{esc(str(r.get('name','')))}</td>",
                f"<td>{esc(str(r.get('severity','')))}</td>",
                f"<td>{esc(str(r.get('desc','')))}</td>",
                f"<td>{esc(str(r.get('triggered','')))}</td>",
                f"<td>{esc(str(r.get('value','')))}</td>",
                f"<td>{esc(str(r.get('threshold','')))}</td>",
                f"<td>{esc(str(r.get('ts','')))}</td>",
            ]) + "</tr>")
        html.append("</table>")

    html.append("</body></html>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

