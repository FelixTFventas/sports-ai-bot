# Sports AI Bot

Bot local de futbol con probabilidades estimadas, cuotas rastreables y evaluacion
temporal. **No garantiza ganancias ni dispone de seguimiento automatico verificado
de casas colombianas.** El funcionamiento actual usa cuotas introducidas manualmente.

## Estado real

- BetPlay, Wplay, Rushbet, Codere y Betano: importacion manual con dominio local,
  hora de observacion, mercado y cuota. No se sustituyen por operaciones de otros paises.
- Mercados activos: **Over 2.5 / Over** y **BTTS / Si**, tiempo reglamentario de 90
  minutos mas descuento, sin prorroga. El importador no verifica la pagina ni sus reglas;
  quien introduce la cuota debe comprobarlas.
- Colombia (`liga_colombia`) e internacional: historicos locales autorizados.
  Admitir una liga no implica tener datos ni modelo validado para ella.
- Los modelos antiguos quedan fuera del flujo nuevo. No se borran datos, modelos,
  cambios locales ni registros de publicaciones.
- Corners y 1X2 no forman parte de las recomendaciones activas. Forebet esta separado
  de los picks propios: solo consulta privada bajo demanda y se etiqueta experimental.

## Instalacion local

Python 3.11 o superior; scikit-learn >=1.6. Desde la carpeta del proyecto:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m sports_ai_bot.main coverage
.venv\Scripts\python.exe -m sports_ai_bot.main check-config
```

La configuracion esta en `.env`; `.env.example` documenta sus campos. No sobrescribir
un `.env` existente ni compartir tokens. No se necesita una API de cuotas para el flujo
manual. `check-config` no imprime claves.

## Datos y entrenamiento

Importar un CSV autorizado con `Date,HomeTeam,AwayTeam,FTHG,FTAG`; `HC,AC` son opcionales.
`Date` es el inicio del partido en ISO 8601 con zona; los resultados deben ser finales
del tiempo reglamentario. No importar partidos anulados, abandonados o en juego como finales.
No inventar ceros para corners desconocidos.

```powershell
.venv\Scripts\python.exe -m sports_ai_bot.main import-history "historicos.csv" --league liga_colombia --source-url "https://proveedor-autorizado.example/resultados" --permission-note "Descripcion del permiso de uso obtenido"
.venv\Scripts\python.exe -m sports_ai_bot.main build-dataset
.venv\Scripts\python.exe -m sports_ai_bot.main train
```

El dominio de ejemplo no es un proveedor. La nota es una declaracion del usuario,
no una comprobacion juridica. Los archivos se identifican por SHA-256 y guardan
procedencia en `data/imports`. Los originales no se borran.

Football-Data publica restricciones que requieren revisar la autorizacion para este
uso. `fetch-data` esta bloqueado por defecto; activar `HISTORICAL_DOWNLOAD_AUTHORIZED`
solo contando con permiso compatible. No se ha verificado una descarga automatica
autorizada de historicos colombianos. Los datos preexistentes no acreditan permiso.

El entrenamiento usa regresion logistica, imputacion aprendida solo en train,
calibracion sigmoid y tramos temporales separados 60/20/20. No selecciona modelos
sobre el test final. Incluye dos ventanas exploratorias dentro de train.

Un target pasa el gate global con 200/100/200 muestras train/calibration/test,
20 muestras de cada clase por tramo y Brier/log loss mejores que prevalencia de train
y constante 0.5. Cada liga requiere ademas 100/50/100 muestras y 10 de cada clase
en test, superando ambos baselines locales. Se registran bins de calibracion por liga.
Esto es un filtro empirico fuera de muestra, **no prueba de rentabilidad ni intervalo
de confianza**. Las ligas ausentes no heredan validacion global.

## Cuotas y picks

Plantillas de columnas vacias en `examples/`. CSV de cuotas:

```csv
league,home_team,away_team,kickoff,bookmaker,market,selection,line,odd,observed_at,source_url
```

Usar identificadores `betplay`, `wplay`, `rushbet`, `codere`, `betano`. Dominios admitidos:
`betplay.com.co`, `wplay.co`, `rushbet.co`, `codere.com.co`, `betano.co` y sus subdominios.
Esta lista es validacion de entrada, no certificacion de licencia ni de disponibilidad.
`kickoff` y `observed_at` requieren zona horaria, por ejemplo `2026-09-15T19:00:00-05:00`.
Over 2.5 usa seleccion `Over`, linea `2.5`; BTTS usa `Si` y linea vacia.
La observacion no puede estar en el futuro ni ser posterior al inicio.

```powershell
.venv\Scripts\python.exe -m sports_ai_bot.main import-quotes "cuotas.csv"
.venv\Scripts\python.exe -m sports_ai_bot.main quotes
.venv\Scripts\python.exe -m sports_ai_bot.main generate
.venv\Scripts\python.exe -m sports_ai_bot.main generate --observation
.venv\Scripts\python.exe -m sports_ai_bot.main status
```

La importacion es atomica y omite snapshots identicos. Solo se considera la ultima
observacion por casa/evento/mercado, nunca el mejor precio antiguo. Caducidad por
defecto: 30 minutos. Ambos equipos necesitan cinco partidos generales y cinco en su
condicion local/visitante; ultimo partido a no mas de 45 dias del proximo encuentro.

`generate` exige modelo y liga validados, esquema y hash del artefacto correctos,
edge >=3 puntos porcentuales y EV teorico >=3%. Selecciona un pick por evento y un
maximo de cinco. No recomienda stakes ni combinadas. Puede devolver cero picks e
indica si faltan cuotas, datos, modelos o valor. Una cuota manual no garantiza que
siga disponible: verificarla en la casa antes de decidir.

`--observation` admite modelos v2 experimentales y registra el analisis separado.
No alimenta la cache publica ni Telegram. No permite usar artefactos antiguos sin
trazabilidad. No se ha realizado validacion prospectiva con apuestas reales.

## Telegram

```powershell
.venv\Scripts\python.exe -m sports_ai_bot.main run-bot
```

| Comando | Uso |
|---|---|
| `/picks` | Cache vigente de picks validados, sin descargas |
| `/cuotas` | Observaciones manuales vigentes |
| `/forebet` | Consulta privada manual de datos experimentales Forebet |
| `/rendimiento` | Resultados de predicciones registradas |
| `/help` | Ayuda |
| `/publishnow` | Publicacion al chat configurado, solo administradores |

Se recalcula el analisis local cada 15 minutos. **Esto no actualiza cuotas ni
historicos.** La publicacion diaria envia un solo resumen a `POST_HOUR_LOCAL` en
`BOT_TIMEZONE`. Los antiguos comandos de picks son aliases y corners sigue desactivado.
Forebet no se incluye en el scheduler ni en `/publishnow`. La CLI antigua de
investigacion se retiro.

`/forebet` abre temporalmente Chrome o Edge en modo visible, consulta directamente las
paginas publicas de Over/Under 2.5 y BTTS, prioriza Primera A y completa con eventos
globales. Cloudflare rechaza clientes HTTP y navegadores headless, por lo que esta
funcion esta orientada al equipo local con sesion grafica; no intenta eludir bloqueos.
La cache predeterminada dura 45 minutos. Cada observacion se conserva separada en
SQLite, no se mezcla con modelos validados ni con cuotas manuales y nunca se publica
automaticamente. Las cuotas mostradas son referencias sin casa identificada.

`TELEGRAM_ADMIN_IDS` contiene IDs numericos separados por comas. Vacio deniega
publicaciones manuales. Se conservan bloqueo entre procesos, deduplicacion diaria y
tratamiento conservador de envios inciertos: no se reintentan automaticamente.

## Resultados y persistencia

```powershell
.venv\Scripts\python.exe -m sports_ai_bot.main import-history "resultados.csv" --league liga_colombia --source-url "https://proveedor-autorizado.example/resultados" --permission-note "Permiso obtenido"
.venv\Scripts\python.exe -m sports_ai_bot.main update-results
.venv\Scripts\python.exe -m sports_ai_bot.main report-performance
```

`import-history` liquida despues de importar. `update-results` solo usa datos locales;
no descarga resultados. Los mercados no liquidables permanecen pendientes. No se
simulan anulaciones, pushes ni resultados faltantes. Las cuotas invalidas se excluyen
de ROI. Los picks nuevos usan una unidad de evaluacion, no una recomendacion de stake.
El resumen separa experimental, validado y antiguo sin clasificar; conserva primera
cuota por seleccion/modelo/estrategia y presenta desgloses por modelo.

- `analysis.sqlite3`: snapshots inmutables de cuotas y predicciones, cache y fecha de
  publicacion confirmada. SQLite no equivale a un registro inviolable.
- `predictions/picks_*.csv`: predicciones y resultados liquidados; se conserva
  trazabilidad adicional al reescribir. El informe mide registros, no demuestra envios.
- `reports/training_summary.json`: esquema, hashes, muestras, baselines y validacion.
- Registro de publicacion anterior: conservarlo para evitar reenvios tras reinicios.

Mantener **todo `DATA_DIR` en almacenamiento persistente**. Para una copia coherente,
detener el bot y los comandos que escriben, respaldar el directorio completo y luego
reiniciar. No borrar SQLite para resolver un envio incierto. Render requiere un disco
persistente que puede tener coste; el despliegue remoto no se activa con este cambio.

## Verificacion

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src tests
```

Las pruebas usan datos sinteticos y no prueban rentabilidad ni conexion real a las
casas. Consulta `docs/sources.md` para la evidencia y limites de cobertura y
`docs/legacy-readme.md` para la documentacion anterior conservada.
