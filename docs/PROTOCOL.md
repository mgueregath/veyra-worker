# Protocolo worker ↔ API de Veyra

El worker **siempre inicia** la conexión (pull): no abre puertos hacia la API y funciona detrás de NAT. Todas las rutas
cuelgan de `/api/worker` y llevan `Authorization: Bearer <token del worker>`. Un token 401/403 hace que el worker espere
y reintente (no se cae). Los cuerpos son JSON.

## Carpeta compartida

API y worker montan el MISMO volumen en la MISMA ruta (`/data`). Las rutas de los trabajos son absolutas dentro de él:
la API deja ahí los datasets exportados y lee de ahí los pesos y resultados. El worker rechaza cualquier ruta que,
resuelta (siguiendo enlaces simbólicos), quede fuera de `VEYRA_DATA_ROOT`.

## Endpoints

| Método y ruta | Cuerpo | Respuesta |
|---|---|---|
| `POST /register` | `{name, kinds[], max_batch, max_live, capabilities}` (`check_only: true` solo valida el token) | `{worker_id}` |
| `POST /heartbeat` (cada ~5 s) | `{running_jobs[], free_batch, free_live}` | `{cancel_jobs[]}` |
| `POST /jobs/claim` | `{kinds[], wait}` | `200 {id, kind, payload}` o `204` (la API puede retener la petición hasta `wait` s) |
| `POST /jobs/{id}/progress` | campos libres + `log` (últimos 60 kB) | `{cancel: bool}` |
| `POST /jobs/{id}/finish` | `{status: done\|failed\|cancelled, result, error, log}` | `{ok}` |

Responsabilidades de la API:
- Al `register`, los trabajos que figuraban «en curso» para ese worker y no estén en `running_jobs` (el worker se reinició) se marcan como fallidos.
- Si dejan de llegar latidos (p. ej. 60 s) los trabajos del worker se marcan como fallidos y el worker como inactivo.
- Cancelar = incluir el id en `cancel_jobs` del siguiente latido. El worker termina el proceso y cierra con `cancelled`.
- `claim` solo debe entregar tipos incluidos en `kinds` (el worker informa solo los que tiene cupo).

## Tipos de trabajo

### `train`
`payload`: `{data, weights, epochs, imgsz, batch, project, name, seed, out}` (`weights`: ruta compartida o `yolov8{n,s,m,l,x}.pt`).
`progress`: `{progress: {epoch, epochs, metrics}}`. `result`: `{map50, map50_95, precision, recall, weights}`.

### `predict`
`payload`: `{weights, images[], out, imgsz, conf}`. Escribe en `out` el JSON `{ruta: [[x, y, w, h, puntaje, clase], …]}`.
`result`: `{out, images}`.

### `live`
`payload`: `{source, detector, weights, classes[], threshold, imgsz, loop, no_realtime, record_dir}`. `source` es una URL
`rtsp(s)://` / `http(s)://` o un archivo compartido (`file:`, `concat:`, `data:` se rechazan).
Cuando el proceso está listo envía `progress {endpoint, token, pid}` (el token es propio de la sesión). El trabajo queda
«en curso» mientras la sesión viva. La API reenvía `GET {endpoint}/stream.mjpg?token=…` al navegador y usa `/stats`,
`/frame.jpg` y `POST /control` del proceso directamente. Cancelar el trabajo detiene la sesión; `done` = terminó la fuente.
