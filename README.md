# Sports AI Bot

Bot de Telegram en Python para publicar picks de futbol en un grupo usando modelos estadisticos propios.

## Resumen

El proyecto descarga historicos, construye features, entrena modelos para mercados de goles y genera mensajes listos para publicar en Telegram.

Mercados soportados:

- Over 2.5
- BTTS

Ligas incluidas:

- Premier League
- La Liga
- Serie A
- Bundesliga
- Ligue 1

Fuentes de datos:

- Historicos: `football-data.co.uk`
- Fixtures proximos: `ESPN scoreboard API`

## Caracteristicas

- Descarga automatica de historicos por liga
- Construccion de dataset de entrenamiento
- Generacion de features para fixtures futuros
- Entrenamiento de modelos para `Over 2.5` y `BTTS`
- Generacion local de mensajes para Telegram
- Comandos interactivos del bot
- Scheduler diario para publicacion automatica

## Estructura

Archivos principales:

- `src/sports_ai_bot/main.py`
- `src/sports_ai_bot/bot/telegram_bot.py`
- `src/sports_ai_bot/collect/historical.py`
- `src/sports_ai_bot/collect/fixtures.py`
- `src/sports_ai_bot/features/build.py`
- `src/sports_ai_bot/train/train_models.py`
- `src/sports_ai_bot/predict/pipeline.py`
- `src/sports_ai_bot/explain/messages.py`
- `src/sports_ai_bot/utils/config.py`

## Requisitos

- Python `3.11+`
- Un bot de Telegram creado con `@BotFather`
- Un `TELEGRAM_CHAT_ID` valido para el grupo o chat destino

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
copy .env.example .env
```

## Configuracion

Completa `C:\Users\JUNIOR\sports-ai-bot\.env` con tus valores reales:

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
POST_HOUR_LOCAL=09:00
BOT_LANGUAGE=es
```

Notas:

- `TELEGRAM_CHAT_ID` de grupos suele empezar por `-100`
- `POST_HOUR_LOCAL` define la hora diaria de publicacion automatica

## Comandos del proyecto

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

Vista previa del mensaje sin enviar:

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

Tests:

```bash
python -m pytest
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

## Flujo recomendado

1. Ejecutar `fetch-data`
2. Ejecutar `build-dataset`
3. Ejecutar `train`
4. Ejecutar `build-fixtures`
5. Revisar `preview-message`
6. Validar con `check-config`
7. Ejecutar `run-bot`

## Despliegue

El proyecto puede ejecutarse localmente o desplegarse en una plataforma que mantenga un proceso Python activo.

Para despliegue remoto, el comando principal es:

```bash
python -m sports_ai_bot.main run-bot
```

La plataforma elegida debe permitir:

- Variables de entorno
- Proceso persistente en Python
- Acceso saliente a Internet para Telegram y fuentes de datos

## Estado actual

La base actual deja listo:

- pipeline de datos historicos
- features para entrenamiento y fixtures futuros
- modelos para `Over 2.5` y `BTTS`
- mensajes locales para Telegram sin dependencias de IA externa
- bot funcional con scheduler diario

## Seguridad

- No subas `.env` al repositorio
- No pegues tokens de Telegram en issues o commits
- Revoca cualquier credencial que haya sido expuesta
