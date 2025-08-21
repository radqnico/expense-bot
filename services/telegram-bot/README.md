Telegram Bot (Dockerized)

Overview
- Minimal Telegram bot using python-telegram-bot with polling.
- Built as a standalone Docker image (compose will use the prebuilt image later).

Build
- docker build -t bot-spese-telegram:dev .

Run
- docker run --rm \
    -e TELEGRAM_BOT_TOKEN=YOUR_TOKEN_HERE \
    --name bot-spese-telegram \
    bot-spese-telegram:dev

Environment
- TELEGRAM_BOT_TOKEN: Telegram bot token from BotFather (required).

Notes
- Uses long polling; no public URL/webhook needed.
- Logs to stdout/stderr.

