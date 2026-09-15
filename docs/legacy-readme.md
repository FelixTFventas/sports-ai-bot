# Sports AI Bot

> DOCUMENTACION HISTORICA: conserva las instrucciones y cambios anteriores a la
> reorganizacion. No usar como guia operativa actual. Consulta ../README.md.

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
TELEGRAM_ADMIN_IDS=
DATA_DIR=./data
BOT_TIMEZONE=America/Bogota
POST_HOUR_LOCAL=09:00
BOT_LANGUAGE=es
THE_ODDS_API_KEY=
THE_ODDS_API_REGION=eu
THE_ODDS_API_BOOKMAKER=bet365
THE_ODDS_API_EXTRA_BOOKMAKERS=pinnacle
CORNERS_PICK_SELECTION=Over
CORNERS_PICK_POINT=9.5
CORNERS_PICK_MIN_PRICE=1.65
CORNERS_PICK_MAX_PRICE=2.15
```

Notas:

- `TELEGRAM_CHAT_ID` de grupos suele empezar por `-100`
- `TELEGRAM_ADMIN_IDS` contiene IDs numericos positivos de usuarios autorizados, separados por comas (ejemplo: `123456789,987654321`). No son nombres de usuario ni IDs de grupos. Vacio deshabilita `/publishnow`, no el scheduler.
- `DATA_DIR` contiene datos y registros SQLite; usa una ruta absoluta y persistente en el servidor.
- `POST_HOUR_LOCAL` define la hora diaria de publicacion automatica
- `BOT_TIMEZONE` define la zona horaria de esa hora y de la deduplicacion diaria.
- `THE_ODDS_API_EXTRA_BOOKMAKERS` permite agregar respaldo como `pinnacle`
- `CORNERS_PICK_SELECTION` y `CORNERS_PICK_POINT` controlan la linea experimental de corners
- `CORNERS_PICK_MIN_PRICE` y `CORNERS_PICK_MAX_PRICE` filtran el rango de cuota para corners

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
- `/forebettop`
- `/publishnow`
- `/performance`

## Flujo recomendado

### Publicacion segura

- `/publishnow` requiere un administrador autorizado y permite una nueva publicacion manual tras 60 segundos desde la finalizacion de la anterior.
- El scheduler y `python -m sports_ai_bot.main send-today` comparten un registro diario por chat y fecha local. Un envio manual no sustituye el diario.
- SQLite impide publicaciones simultaneas que compartan `DATA_DIR`. El scheduler espera hasta aproximadamente cinco minutos si otra publicacion ocupa el registro.
- Los fallos anteriores al envio permiten volver a intentar. Un timeout o fallo despues de iniciar el envio deja el lote como incierto: no se reenvia automaticamente ni se presupone que Telegram no lo recibio.
- Revisa Telegram y los logs ante un envio incierto. No borres los registros para forzar un reenvio: podrias duplicar mensajes. Aun no hay una herramienta de reconciliacion.
- Los mensajes largos se dividen sin truncar. Las operaciones del pipeline se ejecutan fuera del hilo principal, serializando escrituras de las rutas del bot y de publicacion; esto no habilita procesamiento paralelo general de comandos.
- ESPN solo aporta Over 2.5 cuando precio y linea explicita 2.5 pertenecen a la misma cotizacion. Formatos ambiguos se descartan; esto puede reducir la cobertura.

### Preparacion de datos

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
- Un volumen persistente para `DATA_DIR`, incluidos `publications.sqlite3`, `publication-lock.sqlite3` y `work-lock.sqlite3`.

Al migrar, detiene el bot local antes de trasladar los datos y arrancar el servidor. Ejecuta una sola instancia de polling por token. No ejecutes PC y servidor simultaneamente con copias independientes del registro: SQLite no coordina esas copias. Evita sistemas de archivos de red sin bloqueos SQLite fiables. Los comandos de mantenimiento que escriben datos deben ejecutarse con el bot detenido.

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
