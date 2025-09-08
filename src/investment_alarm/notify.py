#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional


def send_email(email_enabled: bool, email_to: Optional[str], subject: str, body: str) -> None:
    if not email_enabled or not email_to:
        return
    user = os.environ.get("EMAIL_USER")
    pwd = os.environ.get("EMAIL_PASS")
    if not user or not pwd:
        logging.warning("EMAIL_USER / EMAIL_PASS が未設定のためメール送信をスキップ")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = email_to
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(user, pwd)
            smtp.sendmail(user, [email_to], msg.as_string())
        logging.info("メール送信完了: %s", email_to)
    except Exception as e:
        logging.warning("メール送信に失敗: %s", e)


__all__ = ["send_email"]

