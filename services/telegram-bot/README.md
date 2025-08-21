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
    -e OLLAMA_HOSTS=http://host.docker.internal:11434 \
    -e OLLAMA_MODEL=qwen2.5:0.5b \
    -e OLLAMA_PULL_ON_START=true \
    -e DB_HOST=postgres -e DB_PORT=5432 -e DB_NAME=appdb -e DB_USER=app -e DB_PASSWORD=app \
    --name bot-spese-telegram \
    bot-spese-telegram:dev

Environment
- TELEGRAM_BOT_TOKEN: Telegram bot token from BotFather (required).
- BOT_NAME: Public bot name to show in profile.
- BOT_SHORT_DESCRIPTION: Short description (shown in profile).
- BOT_DESCRIPTION: Full description (shown in profile).
- OLLAMA_HOSTS: One or more Ollama API base URLs, comma-separated. Each can be
  either a bare `host:port` or full URL. The bot pings all on each message and
  round-robins across available instances. (default in compose: http://ollama:11434)
- OLLAMA_MODEL: Small model name, e.g. `qwen2.5:0.5b`.
- OLLAMA_PULL_ON_START: Pull the model on startup (true/false).
- DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD: Postgres connection for persistence.

Commands
- /help: Show available commands.
- /status: Backend health + queue info.
- /last [n]: Show last n entries (default 5).
- /sum [today|week|month|all]: Sum amounts for period (default month).
- /undo: Delete last entry.
- /export [period|YYYY-MM]: Download CSV for this chat. Period can be today/week/month/all or a specific month in YYYY-MM.
- /report day|week|month|year: Sends charts (cumulative net, daily income/expenses, top expenses) and a PDF summary for the period.
- /reset day|month|all: Delete entries for the given period.
- /month YYYY-MM: Monthly summary (income, expenses, net, by day).

Notes
- Uses long polling; no public URL/webhook needed.
- Logs to stdout/stderr.
- Ensure the chosen model is pulled in your Ollama: `ollama pull qwen2.5:0.5b`.
  - If `OLLAMA_PULL_ON_START=true`, the bot will try to pull automatically (with retries).
