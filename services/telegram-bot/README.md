Telegram Bot (Dockerized)

Overview
- Minimal Telegram bot using python-telegram-bot with polling.
- Built as a standalone Docker image (compose will use the prebuilt image later).

Build
- docker build -t bot-spese-telegram:dev .

Run
- docker run --rm \
    -e TELEGRAM_BOT_TOKEN=YOUR_TOKEN_HERE \
    -e BOT_NAME="RADQ Expenses Tracker" \
    -e BOT_SHORT_DESCRIPTION="Track expenses easily with RADQ." \
    -e BOT_DESCRIPTION="A simple expenses tracker bot. Use /help to see commands." \
    -e OLLAMA_HOST=http://host.docker.internal:11434 \
    -e OLLAMA_MODEL=qwen2.5:0.5b \
    --name bot-spese-telegram \
    bot-spese-telegram:dev

Environment
- TELEGRAM_BOT_TOKEN: Telegram bot token from BotFather (required).
- BOT_NAME: Public bot name to show in profile.
- BOT_SHORT_DESCRIPTION: Short description (shown in profile).
- BOT_DESCRIPTION: Full description (shown in profile).
- OLLAMA_HOST: Ollama API base URL (default in compose: http://ollama:11434).
- OLLAMA_MODEL: Small model name, e.g. `qwen2.5:0.5b`.

Notes
- Uses long polling; no public URL/webhook needed.
- Logs to stdout/stderr.
- Ensure the chosen model is pulled in your Ollama: `ollama pull qwen2.5:0.5b`.
