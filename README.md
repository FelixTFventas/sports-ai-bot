# Sports AI Bot

Bot de Telegram en Python para publicar pronosticos de futbol en grupo.

## Mercados iniciales

- Over 2.5
- BTTS (ambos marcan)

## Ligas iniciales

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

## Fuente historica inicial

- `football-data.co.uk`

## Fuente inicial de fixtures proximos

- `ESPN scoreboard API` sin clave para los proximos dias

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env
```

## Comandos utiles

Descargar historicos:

```bash
python -m sports_ai_bot.main fetch-data
```

Construir dataset de entrenamiento:

```bash
python -m sports_ai_bot.main build-dataset
```

Construir features para fixtures proximos:

```bash
python -m sports_ai_bot.main build-fixtures
```

Entrenar modelos:

```bash
python -m sports_ai_bot.main train
```

Vista previa de mensaje de Telegram sin enviar:

```bash
python -m sports_ai_bot.main preview-message
```

Validar configuracion:

```bash
python -m sports_ai_bot.main check-config
```

Ejecutar bot de Telegram:

```bash
python -m sports_ai_bot.main run-bot
```

## Configuracion de Telegram

1. Crea el bot con `@BotFather` y copia `TELEGRAM_BOT_TOKEN`.
2. Mete el bot en tu grupo y envia al menos un mensaje.
3. Obtiene el `TELEGRAM_CHAT_ID` del grupo.
4. Copia `.env.example` a `.env` y completa las variables.

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=-100...
POST_HOUR_LOCAL=09:00
BOT_LANGUAGE=es
```

5. Verifica la configuracion:

```bash
python -m sports_ai_bot.main check-config
```

6. Arranca el bot:

```bash
python -m sports_ai_bot.main run-bot
```

## Comandos del bot

- `/start`
- `/help`
- `/today`
- `/over`
- `/btts`
- `/top`
- `/publishnow`
- `/performance`

## Estado actual

Esta primera base deja listo:

- descarga de historicos por liga
- generacion de features iniciales
- generacion de features para fixtures futuros usando la temporada actual del mismo proveedor
- entrenamiento de modelos para Over 2.5 y BTTS
- mensajes locales con picks calculados
- estructura del bot de Telegram y scheduler diario
