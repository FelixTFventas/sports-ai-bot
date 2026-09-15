# Fuentes y limites

Revision documental: 2026-09-14. No se usaron claves privadas ni tecnicas para eludir
autenticacion. "No verificado" no significa "inexistente".

| Solicitud | Evidencia | Decision |
|---|---|---|
| BetPlay, Wplay, Rushbet | No aparecen en catalogo consultado de The Odds API | Manual |
| Codere Colombia | Catalogo lista Codere IT, no acredita Colombia | Manual |
| Betano Colombia | Catalogo lista UK/Ontario, no acredita Colombia | Manual |
| Liga colombiana | No aparece liga domestica en catalogo consultado | Importacion local |
| Cuotas historicas The Odds API | Requieren plan de pago | Fuera de alcance gratuito |
| Football-Data | CSV accesible no equivale a permiso para entrenar este bot | Descarga bloqueada por defecto |
| RSSSF Colombia | Documento de campeones con permiso de copia atribuido | Referencia, no feed de partidos |
| Forebet | Paginas publicas y robots.txt sin rutas bloqueadas; terminos restringen copia/distribucion | Consulta privada manual, sin publicacion automatica |

Documentos consultados:

- https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
- https://the-odds-api.com/sports-odds-data/sports-apis.html
- https://the-odds-api.com/liveapi/guides/v4/
- https://the-odds-api.com/terms-and-conditions.html
- https://www.football-data.co.uk/data.php
- https://www.rsssf.com/tablesc/colchamp.html
- https://www.coljuegos.gov.co/
- https://www.forebet.com/en/terms-of-use
- https://www.forebet.com/robots.txt

RSSSF permite copiar el documento revisado con reconocimiento a Juan Pablo Andres /
RSSSF. No se extiende este permiso a todas sus paginas ni a cualquier uso.
No se certificaron licencias vigentes de cada operador en Colombia, ni se verificaron
los terminos individuales de captura de sus paginas. La entrada manual requiere
acceso legitimo y uso compatible con sus condiciones. No hay scraping automatico de
esas paginas. La whitelist de dominios del importador solo reduce errores de atribucion.
