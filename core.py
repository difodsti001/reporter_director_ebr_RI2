"""
core.py

Lógica de negocio del asistente de Reporte Institucional 2 (F4 - EBR 2026,
Sesiones de aprendizaje V1 -> V2).

Dos bases de datos, dos pools separados:
- pool_datos:  base con public."base_ebr_IA" y public.vw_processing_brechas_ebr
- pool_cache:  base con f4_reportes_directivo y f4_consultas_directivo
               (tablas compartidas con RI1; se distinguen por tipo_reporte)

Alcance de esta versión (según los datos disponibles hoy en la base):
- SÍ: cobertura por bandas (adenda), distribución de V2 por aspecto, avance
  V1->V2 en puntos porcentuales, énfasis, lectura por nivel (con supresión
  de grupos pequeños), contraste, preguntas y síntesis para la RTC 2.
- NO (la base no tiene tipo_cambio, BC1-BC4, brechas_v1_persistentes ni
  progresion_global_referencial, y los documentos prohíben reconstruirlos):
  dificultades que persisten, naturaleza de los cambios, progresión formal,
  logros consolidados y prácticas transferibles. El reporte lo declara.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import structlog
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from llm_client import generar_texto_narrativo


# =============================================================================
# Logging estructurado
# =============================================================================

def configurar_logging(nivel: str = "INFO") -> None:
    """Llamar UNA VEZ al arrancar la aplicación, antes de loguear nada."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, nivel.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, nivel.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger(__name__)


# =============================================================================
# Excepciones de negocio
# =============================================================================

class DirectivoNoEncontradoError(Exception):
    """El userid no resuelve a ninguna IE en base_ebr_IA."""


class NoEsDirectivoError(Exception):
    """El userid existe en base_ebr_IA pero su cargo no es DIRECTIVO: el
    reporte institucional solo lo ve el directivo."""


class SinDatosDisponiblesError(Exception):
    """No hay cmids configurados para las evidencias V1/V2."""


class ReporteNoExisteError(Exception):
    """No hay reporte guardado todavía para esta IE."""


# =============================================================================
# Configuración de conexión
# =============================================================================

_KEEPALIVE_PARAMS = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=3"


def _con_keepalives(dsn: str) -> str:
    separador = "&" if "?" in dsn else "?"
    return f"{dsn}{separador}{_KEEPALIVE_PARAMS}"


DSN_DATOS = _con_keepalives(os.environ["DSN_DATOS"])
DSN_CACHE = _con_keepalives(os.environ["DSN_CACHE"])

_MAX_IDLE = int(os.environ.get("POOL_MAX_IDLE_SECONDS", "300"))

_POOL_KWARGS = dict(
    min_size=1,
    max_size=10,
    kwargs={"row_factory": dict_row},
    open=False,
    max_idle=_MAX_IDLE,
    check=AsyncConnectionPool.check_connection,
)

pool_datos = AsyncConnectionPool(conninfo=DSN_DATOS, **_POOL_KWARGS)
pool_cache = AsyncConnectionPool(conninfo=DSN_CACHE, **_POOL_KWARGS)


async def abrir_pools() -> None:
    """Llamar en el evento startup de FastAPI."""
    await pool_datos.open()
    await pool_cache.open()
    await _asegurar_schema_cache()


async def cerrar_pools() -> None:
    """Llamar en el evento shutdown de FastAPI."""
    await pool_datos.close()
    await pool_cache.close()


async def _asegurar_schema_cache() -> None:
    """Crea las tablas de caché si no existen (entornos nuevos). En
    producción las tablas ya existen -- compartidas con RI1, con la columna
    tipo_reporte y las restricciones UNIQUE (cod_modular, tipo_reporte) y
    UNIQUE (cod_modular, userid_directivo, tipo_reporte) -- y esto no
    modifica nada."""
    ddl = """
        CREATE TABLE IF NOT EXISTS f4_reportes_directivo (
            id SERIAL PRIMARY KEY,
            cod_modular VARCHAR NOT NULL,
            estado VARCHAR NOT NULL DEFAULT 'generado',
            n_docentes_evaluados INTEGER NOT NULL,
            reporte_json JSONB,
            fecha_generacion TIMESTAMP NOT NULL DEFAULT now(),
            fecha_actualizacion TIMESTAMP NOT NULL DEFAULT now(),
            tipo_reporte VARCHAR,
            CONSTRAINT chk_estado CHECK (estado IN ('generado', 'muestra_insuficiente', 'sin_datos')),
            CONSTRAINT chk_reporte_json_si_generado
                CHECK (estado <> 'generado' OR reporte_json IS NOT NULL),
            CONSTRAINT uq_cod_modular_tipo_reporte UNIQUE (cod_modular, tipo_reporte)
        );

        CREATE TABLE IF NOT EXISTS f4_consultas_directivo (
            id SERIAL PRIMARY KEY,
            cod_modular VARCHAR NOT NULL,
            userid_directivo INTEGER NOT NULL,
            courseid INTEGER NOT NULL,
            cmid INTEGER NOT NULL,
            id_reporte_directivo INTEGER NOT NULL REFERENCES f4_reportes_directivo(id),
            fecha_consulta TIMESTAMP NOT NULL DEFAULT now(),
            tipo_reporte VARCHAR,
            CONSTRAINT uq_cod_modular_userid_tipo UNIQUE (cod_modular, userid_directivo, tipo_reporte)
        );

        CREATE INDEX IF NOT EXISTS idx_consultas_cod_modular ON f4_consultas_directivo (cod_modular);
        CREATE INDEX IF NOT EXISTS idx_consultas_userid ON f4_consultas_directivo (userid_directivo);
        CREATE INDEX IF NOT EXISTS idx_consultas_reporte ON f4_consultas_directivo (id_reporte_directivo);
    """
    async with pool_cache.connection() as conn:
        await conn.execute(ddl)
        await conn.commit()


# =============================================================================
# Configuración de negocio
# =============================================================================

TIPO_REPORTE = "RI2"

CARGOS_DOCENTE = ("DOCENTE", "JERARQUICO")
# Solo el directivo puede ver el reporte institucional (base_ebr_IA.cargo).
CARGO_DIRECTIVO = "DIRECTIVO"
# Solo participan en el programa los docentes que aprobaron el Curso 1
# (base_ebr_IA.estado_c1): quienes cursan el Curso 2. Es la definición de
# "docente activo" (N) del RI2; la base no tiene un campo de retirados/bajas.
# La aprobación del Curso 2 (estado_c2) NO excluye a nadie de N: solo explica
# por qué un docente no tiene comparación válida (se informa como categoría
# aparte en la cobertura, sin restarlo de la población de referencia).
ESTADO_C1_APROBADO = "APROBADO"
ESTADO_C2_APROBADO = "APROBADO"

NIVELES_EDUCATIVOS = ("Inicial", "Primaria", "Secundaria")
NIVELES_RUBRICA = ("inicio", "en_desarrollo", "logrado", "destacado")
NIVELES_FAVORABLES = ("logrado", "destacado")

# Mínimo de comparaciones válidas (y de docentes en N) para no entrar en
# COB_CONFIDENCIAL (adenda RE-05), y tamaño mínimo de un grupo (nivel
# educativo) para mostrar su desglose.
UMBRAL_MUESTRA_INSUFICIENTE = 5
UMBRAL_GRUPO_MINIMO_NIVEL = 5

# Versión lógica del cálculo/estructura del reporte guardado en la caché: si
# cambia, los reportes guardados con otra versión se regeneran.
VERSION_CALCULO = 9

# Adenda F4-ADENDA-EMISION-RI-EBRJEC26 v1.0, sección 4.
# (N mínimo, N máximo inclusive o None=sin techo, código de banda, % exigido)
BANDAS_COBERTURA = (
    (5, 20, "B1", 85),
    (21, 35, "B2", 80),
    (36, 50, "B3", 70),
    (51, 70, "B4", 65),
    (71, None, "B5", 60),
)


def _clasificar_cobertura(n_evidencias_validas: int, n_docentes_activos: int) -> dict:
    """Clasifica la cobertura de una IE. Para RI2, n_evidencias_validas es el
    número de comparaciones V1->V2 válidas y n_docentes_activos (N) es el
    universo: los docentes que aprobaron el Curso 1.

    Retorna {"banda_aplicada", "umbral_requerido", "umbral_alcanzado",
    "estado_cobertura"}.

    N < 5 (banda B0): no hay banda numérica y el estado es SIEMPRE
    COB_CONFIDENCIAL, aunque todos tengan comparación válida (adenda RE-05).

    N >= 5: umbral_requerido = máx(5; ⌈% de la banda × N⌉) (adenda RE-02,
    redondeo hacia arriba). Con menos de 5 comparaciones válidas,
    COB_CONFIDENCIAL; entre 5 y el umbral, COB_BAJO_UMBRAL (reporte con
    advertencia de cobertura); desde el umbral, COB_SUFICIENTE."""
    if n_docentes_activos < UMBRAL_MUESTRA_INSUFICIENTE:
        return {
            "banda_aplicada": "B0",
            "umbral_requerido": None,
            "umbral_alcanzado": False,
            "estado_cobertura": "COB_CONFIDENCIAL",
        }

    banda_aplicada, umbral_requerido = "B0", UMBRAL_MUESTRA_INSUFICIENTE
    for n_min, n_max, codigo, pct in BANDAS_COBERTURA:
        if n_min <= n_docentes_activos and (n_max is None or n_docentes_activos <= n_max):
            banda_aplicada = codigo
            # round(..., 9) evita que el error de coma flotante (ej.
            # 0.85*20 = 17.000000000000004) suba el techo una unidad de más.
            umbral_requerido = max(
                UMBRAL_MUESTRA_INSUFICIENTE,
                math.ceil(round(pct / 100 * n_docentes_activos, 9)),
            )
            break

    if n_evidencias_validas < UMBRAL_MUESTRA_INSUFICIENTE:
        estado_cobertura = "COB_CONFIDENCIAL"
    elif n_evidencias_validas < umbral_requerido:
        estado_cobertura = "COB_BAJO_UMBRAL"
    else:
        estado_cobertura = "COB_SUFICIENTE"

    return {
        "banda_aplicada": banda_aplicada,
        "umbral_requerido": umbral_requerido,
        "umbral_alcanzado": n_evidencias_validas >= umbral_requerido,
        "estado_cobertura": estado_cobertura,
    }


# =============================================================================
# Cursos y cmids (el curso es el mismo; el cmid identifica la entrega)
# =============================================================================

CMIDS_V1_POR_CURSO: dict[int, list[int]] = {
    2918: [206189],
    2926: [207613],
    2931: [207631],
    2934: [207649],
    2937: [207667],
    2940: [207685],
    2943: [207703],
    2946: [207721],
    2949: [207739],
    2952: [207757],
    2955: [207775],
    2958: [207793],
    2961: [207811],
    2964: [207829],
    2967: [207847],
    2970: [207865],
    2973: [207883],
    2976: [207901],
    2979: [207919],
    2982: [207937],
    2985: [207955],
    2988: [207973],
    2991: [207991],
    2994: [208010],
}

CMIDS_V2_POR_CURSO: dict[int, list[int]] = {
    2918: [206728],
    2926: [216633],
    2931: [216644],
    2934: [216654],
    2937: [216664],
    2940: [216674],
    2943: [216684],
    2946: [216694],
    2949: [216704],
    2952: [216714],
    2955: [216724],
    2958: [216734],
    2961: [216744],
    2964: [216754],
    2967: [216764],
    2970: [217767],
    2973: [217756],
    2976: [217746],
    2979: [217736],
    2982: [217722],
    2985: [217712],
    2988: [217702],
    2991: [217688],
    2994: [217674],
}


def _aplanar_cmids(por_curso: dict[int, list[int]]) -> list[int]:
    return [cmid for cmids in por_curso.values() for cmid in cmids]


def _mapa_cmid_a_version() -> dict[int, str]:
    mapa = {cmid: "V1" for cmid in _aplanar_cmids(CMIDS_V1_POR_CURSO)}
    mapa.update({cmid: "V2" for cmid in _aplanar_cmids(CMIDS_V2_POR_CURSO)})
    return mapa


# =============================================================================
# Referencia de aspectos (denominaciones visibles de RI2 -- Modelo y
# Especificación del Producto Visible, tabla "Resultado general de V2")
# =============================================================================

CRITERIOS: dict[str, dict[str, str]] = {
    "C1": {
        "nombre": "Coherencia entre propósito y desafíos",
        "que_evalua": (
            "Que la sesión responda a los desafíos identificados en el "
            "diagnóstico previo, no a un formato genérico."
        ),
    },
    "C2": {
        "nombre": "Situación significativa y desafío de aprendizaje",
        "que_evalua": (
            "Que la situación de aprendizaje esté orientada al desarrollo "
            "de competencias y pensamiento crítico, no solo a completar "
            "actividades."
        ),
    },
    "C3": {
        "nombre": "Demanda cognitiva e involucramiento activo",
        "que_evalua": (
            "Que las actividades propuestas exijan análisis, toma de "
            "decisiones o argumentación, no solo participación."
        ),
    },
    "C4": {
        "nombre": "Mediación y evaluación formativa",
        "que_evalua": (
            "Que las estrategias de mediación y los criterios de "
            "evaluación formativa sean coherentes entre sí y con el "
            "propósito."
        ),
    },
    "C5": {
        "nombre": "Uso pedagógico de recursos y condiciones de aprendizaje",
        "que_evalua": (
            "Que los recursos y condiciones de aprendizaje elegidos "
            "respondan a una decisión pedagógica, no a disponibilidad."
        ),
    },
}


def _nombre_criterio(criterio_id: str) -> str:
    """'C1' -> 'Coherencia entre propósito y desafíos'. Nunca antepone el
    código: C1-C5 no deben mostrarse al directivo."""
    info = CRITERIOS.get(criterio_id)
    return info["nombre"] if info else criterio_id


# =============================================================================
# Acceso a datos
# =============================================================================

def _armar_clave_ie(cod_institucional: Optional[str], nombre_ie: Optional[str],
                    ugel: Optional[str], region: Optional[str], distrito: Optional[str]) -> str:
    """Clave de la IE en la caché. Preferentemente el cod_institucional; si
    es null o vacío, la clave compuesta nombre|ugel|región|distrito."""
    cod = (cod_institucional or "").strip()
    if cod:
        return cod
    return "|".join([nombre_ie or "", ugel or "", region or "", distrito or ""])


def _filtro_ie_sql(ie: dict) -> tuple[str, dict]:
    """Condición SQL (sobre base_ebr_IA) que identifica a los docentes de la
    IE: por cod_institucional si el directivo lo tiene; si no, por las 4
    columnas nombre_ie, ugel, región y distrito."""
    cod = (ie.get("cod_institucional") or "").strip()
    if cod:
        return "TRIM(cod_institucional) = %(cod_institucional)s", {"cod_institucional": cod}
    return (
        'nombre_ie = %(nombre_ie)s AND ugel IS NOT DISTINCT FROM %(ugel)s '
        'AND "región" = %(region)s AND distrito IS NOT DISTINCT FROM %(distrito)s',
        {
            "nombre_ie": ie["nombre_ie"],
            "ugel": ie.get("ugel"),
            "region": ie["region"],
            "distrito": ie.get("distrito"),
        },
    )


def _elegir_fila_directivo(rows: list[dict], userid: int) -> dict:
    """De las filas de base_ebr_IA del userid elige la del directivo.
    Lanza DirectivoNoEncontradoError si el userid no existe y
    NoEsDirectivoError si existe pero su cargo no es DIRECTIVO."""
    if not rows:
        log.warning("directivo_no_encontrado", userid=userid)
        raise DirectivoNoEncontradoError(f"userid {userid} no tiene IE asociada")
    for r in rows:
        if (r.get("cargo") or "").strip().upper() == CARGO_DIRECTIVO:
            return r
    log.warning("acceso_denegado_no_es_directivo", userid=userid)
    raise NoEsDirectivoError(f"userid {userid} no tiene cargo {CARGO_DIRECTIVO}")


async def resolver_ie(userid: int) -> dict:
    """Busca la IE del directivo por userid y valida que su cargo en
    base_ebr_IA sea DIRECTIVO. Retorna la clave de la IE ("clave_ie":
    cod_institucional o, si no hay, nombre|ugel|región|distrito) más los
    campos necesarios para filtrar a sus docentes."""
    query = """
        SELECT cod_institucional, nombre_ie, ugel, "región" AS region, distrito, cargo
        FROM public."base_ebr_IA"
        WHERE userid = %(userid)s
    """
    async with pool_datos.connection() as conn:
        rows = await (await conn.execute(query, {"userid": userid})).fetchall()

    row = _elegir_fila_directivo(rows, userid)
    cod = (row["cod_institucional"] or "").strip()
    clave_ie = _armar_clave_ie(cod, row["nombre_ie"], row["ugel"], row["region"], row["distrito"])
    log.info(
        "ie_resuelta",
        userid=userid,
        clave_ie=clave_ie,
        por_cod_institucional=bool(cod),
        nombre_ie=row["nombre_ie"],
    )
    return {
        "clave_ie": clave_ie,
        "cod_institucional": cod or None,
        "nombre_ie": row["nombre_ie"],
        "ugel": row["ugel"],
        "region": row["region"],
        "distrito": row["distrito"],
    }


def _normalizar_nivel(valor: Optional[str]) -> Optional[str]:
    """La base mezcla mayúsculas/minúsculas ('secundaria' y 'Secundaria') y
    algún texto 'NULL'. Devuelve el nombre canónico o None."""
    if not valor:
        return None
    limpio = valor.strip().lower()
    for canonico in NIVELES_EDUCATIVOS:
        if canonico.lower() == limpio:
            return canonico
    return None


async def obtener_docentes_activos(ie: dict) -> tuple[dict[int, Optional[str]], set[int]]:
    """Docentes activos de la IE: los que aprobaron el Curso 1 (estado_c1).

    Retorna ({userid: nivel_educativo}, {userid de los que NO aprobaron el
    Curso 2}). El primer dict es N (n_docentes_activos): el universo de
    cálculo y de donde se toman las evidencias V1 y V2. El conjunto es solo
    informativo: esos docentes siguen dentro de N, y si no tienen comparación
    válida se informan como "no aprobó el Curso 2"."""
    filtro, params = _filtro_ie_sql(ie)
    query = f"""
        SELECT DISTINCT userid, nivel_educativo,
               UPPER(TRIM(estado_c1)) AS estado_c1, UPPER(TRIM(estado_c2)) AS estado_c2
        FROM public."base_ebr_IA"
        WHERE {filtro}
          AND UPPER(TRIM(cargo)) = ANY(%(cargos)s)
    """
    async with pool_datos.connection() as conn:
        rows = await (await conn.execute(query, {**params, "cargos": list(CARGOS_DOCENTE)})).fetchall()

    activos: dict[int, Optional[str]] = {}
    c2_aprobado: set[int] = set()
    for r in rows:
        if r["estado_c1"] != ESTADO_C1_APROBADO:
            continue
        uid = int(r["userid"])
        # Si un userid figurara dos veces con niveles distintos, se
        # conserva el primero no nulo.
        if activos.get(uid) is None:
            activos[uid] = _normalizar_nivel(r["nivel_educativo"])
        if r["estado_c2"] == ESTADO_C2_APROBADO:
            c2_aprobado.add(uid)
    return activos, set(activos) - c2_aprobado


# La vista en producción devuelve "en_proceso" aunque el resto del pipeline
# usa "en_desarrollo". Se normaliza en el único punto donde se leen filas.
_ALIAS_NIVEL_OBTENIDO = {"en_proceso": "en_desarrollo"}


async def obtener_filas_vista(user_ids: list[int]) -> list[dict]:
    """Filas de la vista para los docentes dados, de las evidencias V1 y V2
    (se distinguen por cmid). Sin filtrar por status."""
    cmids = _aplanar_cmids(CMIDS_V1_POR_CURSO) + _aplanar_cmids(CMIDS_V2_POR_CURSO)
    if not cmids:
        raise SinDatosDisponiblesError("No hay cmids de V1/V2 configurados")
    if not user_ids:
        return []

    query = """
        SELECT run_id, user_id, cmid, status, completed_at, criterion_index, nivel_obtenido
        FROM public.vw_processing_brechas_ebr
        WHERE cmid = ANY(%(cmids)s)
          AND user_id = ANY(%(user_ids)s)
    """
    async with pool_datos.connection() as conn:
        rows = await (await conn.execute(
            query, {"cmids": cmids, "user_ids": list(user_ids)}
        )).fetchall()

    for row in rows:
        row["nivel_obtenido"] = _ALIAS_NIVEL_OBTENIDO.get(row["nivel_obtenido"], row["nivel_obtenido"])
    return rows


# =============================================================================
# Emparejamiento V1 -> V2 y clasificación de la evidencia
# =============================================================================

def seleccionar_evidencia_por_version(filas: list[dict]) -> dict[int, dict[str, dict]]:
    """De las filas crudas de la vista arma, por docente y versión, UNA
    evidencia: {user_id: {"V1": ev, "V2": ev}} con ev = {"status", "completa",
    "niveles": {criterion_index: nivel}}.

    Un docente puede tener varios intentos (run_id) para el mismo cmid. Se
    elige el último intento exitoso y completo; si no hay, el último exitoso
    (queda como incompleto: su carga en F2 puede estar en curso); si no hay
    ninguno exitoso, el último intento sea cual sea su estado."""
    mapa = _mapa_cmid_a_version()
    corridas: dict[tuple, dict] = {}
    for f in filas:
        version = mapa.get(f["cmid"])
        if version is None:
            continue
        clave = (f["user_id"], version, f["run_id"])
        c = corridas.setdefault(clave, {"status": f["status"], "completed_at": f["completed_at"], "filas": []})
        c["filas"].append(f)

    def _orden(c: dict):
        return (c["completed_at"] is not None, c["completed_at"] or datetime.min)

    def _completa(c: dict) -> bool:
        return c["status"] == "success" and all(
            r["criterion_index"] is not None and r["nivel_obtenido"] is not None for r in c["filas"]
        )

    por_usuario_version: dict[tuple, list[dict]] = {}
    for (uid, version, _run), c in corridas.items():
        por_usuario_version.setdefault((uid, version), []).append(c)

    resultado: dict[int, dict[str, dict]] = {}
    for (uid, version), lista in por_usuario_version.items():
        completas = [c for c in lista if _completa(c)]
        exitosas = [c for c in lista if c["status"] == "success"]
        elegida = max(completas or exitosas or lista, key=_orden)
        completa = _completa(elegida)
        if elegida["status"] == "success" and not completa:
            log.warning("evidencia_con_registro_incompleto", user_id=uid, version=version)
        resultado.setdefault(uid, {})[version] = {
            "status": elegida["status"],
            "completa": completa,
            "niveles": {
                r["criterion_index"]: r["nivel_obtenido"]
                for r in elegida["filas"]
                if r["criterion_index"] is not None and r["nivel_obtenido"] is not None
            } if completa else {},
        }
    return resultado


def clasificar_comparaciones(docentes: dict[int, Optional[str]], evidencias: dict[int, dict[str, dict]],
                             no_aprobaron_c2: Optional[set[int]] = None) -> dict:
    """Clasifica a cada docente activo según el estado de su comparación:

      valida             V1 y V2 exitosas y completas -> entra a los cálculos.
      sin_referencia_v1  V2 exitosa pero sin V1 válida -> solo cobertura.
      no_valida          V2 con validation_failed (el documento no
                         corresponde al producto) -> solo cobertura.
      no_aprobo_c2       sin V2 al corte y sin haber aprobado el Curso 2 ->
                         solo cobertura (sigue contando en N).
      sin_entrega_v2     sin V2 al corte (sin fila, 'failed', en curso) pero
                         con el Curso 2 aprobado.

    Los cinco grupos suman n_docentes_activos."""
    no_aprobaron_c2 = no_aprobaron_c2 or set()
    comparaciones = []
    n_sin_ref = n_no_validas = n_sin_entrega = n_no_aprobo_c2 = 0

    for uid, nivel_educativo in docentes.items():
        ev = evidencias.get(uid, {})
        v1, v2 = ev.get("V1"), ev.get("V2")

        if v2 and v2["status"] == "validation_failed":
            n_no_validas += 1
        elif v2 and v2["status"] == "success" and v2["completa"]:
            if v1 and v1["status"] == "success" and v1["completa"]:
                comparaciones.append({
                    "user_id": uid,
                    "nivel_educativo": nivel_educativo,
                    "v1": v1["niveles"],
                    "v2": v2["niveles"],
                })
            else:
                n_sin_ref += 1
        elif uid in no_aprobaron_c2:
            n_no_aprobo_c2 += 1
        else:
            n_sin_entrega += 1

    return {
        "comparaciones": comparaciones,
        "n_docentes_activos": len(docentes),
        "n_no_aprobaron_c2": n_no_aprobo_c2,
        "n_comparaciones_validas": len(comparaciones),
        "n_sin_referencia_v1": n_sin_ref,
        "n_no_validas": n_no_validas,
        "n_sin_entrega_v2": n_sin_entrega,
    }


async def calcular_estado_ie(ie: dict) -> dict:
    """Pipeline de datos: docentes activos -> filas V1/V2 -> comparaciones."""
    docentes, no_aprobaron_c2 = await obtener_docentes_activos(ie)
    filas = await obtener_filas_vista(list(docentes))
    evidencias = seleccionar_evidencia_por_version(filas)
    estado = clasificar_comparaciones(docentes, evidencias, no_aprobaron_c2)
    log.info(
        "estado_ie_calculado",
        clave_ie=ie["clave_ie"],
        n_docentes_activos=estado["n_docentes_activos"],
        n_no_aprobaron_c2=estado["n_no_aprobaron_c2"],
        n_comparaciones_validas=estado["n_comparaciones_validas"],
        n_sin_referencia_v1=estado["n_sin_referencia_v1"],
        n_no_validas=estado["n_no_validas"],
        n_sin_entrega_v2=estado["n_sin_entrega_v2"],
    )
    return estado


# =============================================================================
# Cálculos agregados (funciones puras, sobre las comparaciones válidas)
# =============================================================================

def calcular_distribucion_v2(comparaciones: list[dict]) -> list[dict]:
    """Distribución de nivel_v2 por aspecto en los 4 niveles de la rúbrica,
    más los agrupados de lectura "Inicio + En desarrollo" y "Logrado +
    Destacado" (no son niveles nuevos). Población: comparaciones válidas."""
    por_criterio: dict[int, dict[str, int]] = {}
    for cmp_ in comparaciones:
        for criterio, nivel in cmp_["v2"].items():
            if nivel not in NIVELES_RUBRICA:
                continue
            conteos = por_criterio.setdefault(criterio, {n: 0 for n in NIVELES_RUBRICA})
            conteos[nivel] += 1

    resultado = []
    for criterio, conteos in sorted(por_criterio.items()):
        total = sum(conteos.values())
        if total == 0:
            continue
        pct = {n: round(100 * conteos[n] / total) for n in NIVELES_RUBRICA}
        resultado.append({
            "criterio_id": f"C{criterio}",
            "n": total,
            "n_inicio": conteos["inicio"],
            "n_en_desarrollo": conteos["en_desarrollo"],
            "n_logrado": conteos["logrado"],
            "n_destacado": conteos["destacado"],
            "pct_inicio": pct["inicio"],
            "pct_en_desarrollo": pct["en_desarrollo"],
            "pct_logrado": pct["logrado"],
            "pct_destacado": pct["destacado"],
            "pct_inicio_en_desarrollo": pct["inicio"] + pct["en_desarrollo"],
            "pct_logrado_destacado": pct["logrado"] + pct["destacado"],
        })
    return resultado


def calcular_avance_por_criterio(comparaciones: list[dict]) -> list[dict]:
    """Avance respecto de la línea de base (Ficha RI2 7.2; Instrumento
    Parte D): avance = % favorable V2 - % favorable V1, en puntos
    porcentuales, ambos calculados sobre LA MISMA población (las
    comparaciones válidas que tienen nivel en V1 y en V2 para ese aspecto).
    "Favorable" = Logrado + Destacado."""
    criterios = sorted({c for cmp_ in comparaciones for c in cmp_["v2"]})
    resultado = []
    for criterio in criterios:
        pares = [
            (cmp_["v1"][criterio], cmp_["v2"][criterio])
            for cmp_ in comparaciones
            if cmp_["v1"].get(criterio) in NIVELES_RUBRICA and cmp_["v2"].get(criterio) in NIVELES_RUBRICA
        ]
        n = len(pares)
        if n == 0:
            continue
        f1 = sum(1 for v1, _ in pares if v1 in NIVELES_FAVORABLES)
        f2 = sum(1 for _, v2 in pares if v2 in NIVELES_FAVORABLES)
        pct_v1 = round(100 * f1 / n, 1)
        pct_v2 = round(100 * f2 / n, 1)
        resultado.append({
            "criterio_id": f"C{criterio}",
            "nombre": _nombre_criterio(f"C{criterio}"),
            "n": n,
            "n_favorable_v1": f1,
            "n_favorable_v2": f2,
            "pct_favorable_v1": pct_v1,
            "pct_favorable_v2": pct_v2,
            "avance_pp": round(pct_v2 - pct_v1, 1),
        })
    return resultado


def calcular_lectura_por_nivel(comparaciones: list[dict]) -> dict:
    """Sección 7: avance por aspecto dentro de cada nivel educativo, en el
    orden fijo Inicial, Primaria, Secundaria y sin rankings. Un nivel solo se
    muestra si tiene al menos UMBRAL_GRUPO_MINIMO_NIVEL comparaciones válidas:
    con menos casos se podría inferir el resultado de un docente. Los
    docentes sin nivel registrado no entran en esta sección (sí en el resto
    del reporte). Retorna también si algún nivel quedó sin mostrar."""
    por_nivel: dict[str, list[dict]] = {}
    for cmp_ in comparaciones:
        if cmp_["nivel_educativo"] in NIVELES_EDUCATIVOS:
            por_nivel.setdefault(cmp_["nivel_educativo"], []).append(cmp_)

    niveles = [
        {
            "nivel_educativo": nivel,
            "n": len(por_nivel[nivel]),
            "avance_por_aspecto": calcular_avance_por_criterio(por_nivel[nivel]),
        }
        for nivel in NIVELES_EDUCATIVOS
        if len(por_nivel.get(nivel, [])) >= UMBRAL_GRUPO_MINIMO_NIVEL
    ]
    return {"niveles": niveles, "hay_niveles_no_mostrados": len(niveles) < len(por_nivel)}


# -----------------------------------------------------------------------------
# Formato de cifras (coma decimal, como en el modelo del reporte)
# -----------------------------------------------------------------------------

def _fmt_num(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def _fmt_pct(x: float) -> str:
    return f"{_fmt_num(x)} %"


def _fmt_pp(x: float) -> str:
    signo = "+" if x > 0 else ("-" if x < 0 else "")
    return f"{signo}{_fmt_num(abs(x))} p.p."


# -----------------------------------------------------------------------------
# Textos armados en Python (las cifras nunca las escribe el LLM)
# -----------------------------------------------------------------------------

FUENTES_CONTRASTE_POR_ASPECTO = {
    "C1": "las evidencias de implementación y las matrices de análisis",
    "C2": "las evidencias de implementación y las evidencias de aprendizaje de los estudiantes",
    "C3": "las sesiones implementadas y las evidencias de aprendizaje de los estudiantes",
    "C4": "las matrices de análisis y las evidencias de seguimiento",
    "C5": "las evidencias de implementación y las evidencias de aula",
}


def _primera_oracion_interpretacion(c: dict, prudente: bool = False) -> str:
    """Primera oración de la interpretación de un aspecto, con el conteo
    EXACTO (no derivado de un porcentaje redondeado). Solo Python escribe
    esta oración; el LLM únicamente redacta la explicación que va después."""
    if c["pct_logrado_destacado"] >= c["pct_inicio_en_desarrollo"]:
        n_grupo, pct_grupo, resultado = c["n_logrado"] + c["n_destacado"], c["pct_logrado_destacado"], "Logrado o Destacado"
    else:
        n_grupo, pct_grupo, resultado = c["n_inicio"] + c["n_en_desarrollo"], c["pct_inicio_en_desarrollo"], "Inicio o En desarrollo"
    ambito = "revisadas" if prudente else "válidas"
    return f"En {n_grupo} de las {c['n']} comparaciones {ambito} ({pct_grupo}%) se observa {resultado} en la Sesión V2."


def _segunda_oracion_avance(a: Optional[dict]) -> str:
    if not a:
        return ""
    if a["avance_pp"] == 0:
        return (
            "Respecto de la Sesión V1, la proporción de resultados Logrado o Destacado se mantuvo en "
            f"{_fmt_pct(a['pct_favorable_v2'])} en esa misma población."
        )
    return (
        "Respecto de la Sesión V1, la proporción de resultados Logrado o Destacado pasó de "
        f"{_fmt_pct(a['pct_favorable_v1'])} a {_fmt_pct(a['pct_favorable_v2'])} "
        f"({_fmt_pp(a['avance_pp'])}) en esa misma población."
    )


def _oracion_contraste(criterio_id: str, prudente: bool) -> str:
    fuente = FUENTES_CONTRASTE_POR_ASPECTO.get(criterio_id, "las evidencias de implementación")
    if prudente:
        return (
            "Esto describe únicamente las comparaciones revisadas y debe tomarse como una hipótesis, "
            f"lo que conviene verificar con {fuente}."
        )
    return f"Para comprender mejor esta situación, conviene contrastarla con {fuente}."


def construir_hallazgos_panorama(estado: dict, avance: list[dict], distribucion: list[dict], prudente: bool) -> list[dict]:
    """Sección 1: hallazgos principales + información que los sustenta.
    Determinista. No jerarquiza aspectos como "el que requiere mayor
    atención" (en bajo umbral la adenda lo prohíbe expresamente)."""
    n, total = estado["n_comparaciones_validas"], estado["n_docentes_activos"]
    pct = round(100 * n / total, 1) if total else 0.0
    ambito = "En las comparaciones revisadas" if prudente else "En las comparaciones analizadas"
    hallazgos = [{
        "hallazgo": "Cobertura del análisis",
        "informacion": f"{n} de {total} docentes cuentan con comparación válida V1→V2 ({_fmt_pct(pct)}).",
    }]

    con_avance = sorted([a for a in avance if a["avance_pp"] > 0], key=lambda a: -a["avance_pp"])
    if con_avance:
        a = con_avance[0]
        hallazgos.append({
            "hallazgo": "Cambio favorable destacado",
            "informacion": (
                f"{ambito}, la proporción favorable en {a['nombre'][0].lower() + a['nombre'][1:]} "
                f"aumentó de {_fmt_pct(a['pct_favorable_v1'])} a {_fmt_pct(a['pct_favorable_v2'])} "
                f"({_fmt_pp(a['avance_pp'])})."
            ),
        })

    sin_avance = [a for a in avance if a["avance_pp"] <= 0]
    if sin_avance:
        nombres = "; ".join(a["nombre"] for a in sin_avance)
        hallazgos.append({
            "hallazgo": "Aspectos en los que la proporción favorable no aumentó",
            "informacion": f"{ambito}, la proporción favorable se mantuvo o disminuyó en: {nombres}.",
        })

    if not prudente:
        # Aspectos con presencia relevante de Inicio / En desarrollo en V2,
        # listados en el orden oficial de los aspectos (sin ordenar por %).
        por_consolidar = [c for c in distribucion if c["pct_inicio_en_desarrollo"] >= 40]
        if por_consolidar:
            nombres = "; ".join(_nombre_criterio(c["criterio_id"]) for c in por_consolidar)
            hallazgos.append({
                "hallazgo": "Aspectos que requieren consolidación",
                "informacion": f"En la Sesión V2, 40 % o más de las comparaciones se ubica en Inicio o En desarrollo en: {nombres}.",
            })

    return hallazgos


def calcular_enfasis_lectura(avance_por_id: dict[str, dict], prudente: bool) -> list[dict]:
    """Sección 4: énfasis del programa, sustentados SOLO en los indicadores
    disponibles. No se crean indicadores independientes (ni porcentajes) para
    énfasis que la rúbrica no evalúa directamente."""
    ambito = "las comparaciones revisadas" if prudente else "las comparaciones analizadas"
    enfasis = []

    c3 = avance_por_id.get("C3")
    if c3:
        enfasis.append({
            "enfasis": "Involucramiento activo",
            "texto_permitido": (
                f"En {ambito}, la proporción de Sesiones con resultado favorable en demanda cognitiva e "
                f"involucramiento activo pasó de {_fmt_pct(c3['pct_favorable_v1'])} a "
                f"{_fmt_pct(c3['pct_favorable_v2'])} ({_fmt_pp(c3['avance_pp'])}). Conviene analizar cómo "
                "sostener estas oportunidades en diferentes experiencias."
            ),
        })

    c2 = avance_por_id.get("C2")
    if c2 or c3:
        partes = []
        if c2:
            partes.append(f"situación significativa y desafío de aprendizaje ({_fmt_pp(c2['avance_pp'])})")
        if c3:
            partes.append(f"demanda cognitiva e involucramiento activo ({_fmt_pp(c3['avance_pp'])})")
        enfasis.append({
            "enfasis": "Pensamiento crítico",
            "texto_permitido": (
                "Este énfasis no cuenta con un indicador independiente. Se lee a partir de los aspectos "
                f"relacionados: {' y '.join(partes)}. El desafío es asegurar profundidad y consistencia, "
                "no solo presencia de actividades."
            ),
        })

    enfasis.append({
        "enfasis": "Comprensión lectora",
        "texto_permitido": (
            "Este reporte no presenta un indicador de comprensión lectora porque las Sesiones no la evalúan "
            "como un aspecto independiente. Su análisis debe contrastarse con evidencias específicas de "
            "aprendizaje de los estudiantes."
        ),
    })

    enfasis.append({
        "enfasis": "Resolución de situaciones y problemas",
        "texto_permitido": (
            "Este énfasis no cuenta con un indicador independiente. Los aspectos de situación significativa y "
            "de demanda cognitiva aportan elementos para analizar la autenticidad y la complejidad de las "
            "situaciones propuestas."
        ),
    })
    return enfasis


def calcular_aspectos_contraste(avance: list[dict], distribucion: list[dict], prudente: bool) -> list[dict]:
    """Sección 8 -- "Lo que conviene contrastar con otras evidencias".
    Determinista; formato de la tabla del modelo: lo que muestra el
    reporte / otras evidencias a considerar / para comprender mejor."""
    filas = []
    cierre = lambda fuente: f" Lo que conviene verificar con {fuente}." if prudente else ""

    con_avance = sorted([a for a in avance if a["avance_pp"] > 0], key=lambda a: -a["avance_pp"])
    if con_avance:
        a = con_avance[0]
        fuente = "las sesiones implementadas y las evidencias recogidas durante el proceso"
        filas.append({
            "hallazgo": (
                f"{a['nombre']}: la proporción favorable pasó de {_fmt_pct(a['pct_favorable_v1'])} a "
                f"{_fmt_pct(a['pct_favorable_v2'])} ({_fmt_pp(a['avance_pp'])})." + cierre(fuente)
            ),
            "fuente_sugerida": "Sesiones implementadas y evidencias recogidas durante el proceso",
            "pregunta": "¿Qué decisiones parecen haber favorecido la evolución observada?",
        })

    pendientes = sorted([c for c in distribucion if c["pct_inicio_en_desarrollo"] > 0], key=lambda c: -c["pct_inicio_en_desarrollo"])
    if pendientes:
        c = pendientes[0]
        fuente = "la matriz de análisis y las evidencias de aula"
        filas.append({
            "hallazgo": (
                f"{_nombre_criterio(c['criterio_id'])}: {c['n_inicio'] + c['n_en_desarrollo']} de {c['n']} "
                f"comparaciones ({c['pct_inicio_en_desarrollo']}%) se ubica en Inicio o En desarrollo en la Sesión V2."
                + cierre(fuente)
            ),
            "fuente_sugerida": "Matriz de análisis y evidencias de aula",
            "pregunta": "¿Qué dificultades aparecen cuando se intenta trasladar el aprendizaje a la práctica?",
        })

    sin_avance = sorted([a for a in avance if a["avance_pp"] <= 0], key=lambda a: a["avance_pp"])
    if sin_avance:
        a = sin_avance[0]
        fuente = "las versiones de planificación y el análisis de implementación"
        filas.append({
            "hallazgo": (
                f"{a['nombre']}: la proporción favorable "
                + ("se mantuvo" if a["avance_pp"] == 0 else f"varió {_fmt_pp(a['avance_pp'])}")
                + f" ({_fmt_pct(a['pct_favorable_v1'])} en V1 y {_fmt_pct(a['pct_favorable_v2'])} en V2)."
                + cierre(fuente)
            ),
            "fuente_sugerida": "Versiones de planificación y análisis de implementación",
            "pregunta": "¿Qué condiciones ayudan a sostener y profundizar las decisiones favorables en la nueva versión?",
        })
    return filas


PREGUNTAS_RTC = (
    "¿Qué cambios reconocemos en nuestras decisiones pedagógicas cuando comparamos la versión inicial y la versión posterior de nuestras experiencias?",
    "¿Qué evidencias de la implementación influyeron realmente en las decisiones que decidimos modificar?",
    "¿En qué aspectos observamos cambios favorables que todavía necesitamos sostener de manera más consistente?",
    "¿Qué aspectos identificados al inicio continúan presentes y qué necesitamos comprender mejor sobre ellos?",
    "¿Qué podemos aprender de las experiencias que muestran mejoras más consistentes sin convertirlas en recetas para otros contextos?",
    "¿Qué nos muestran los cambios realizados sobre nuestra manera de utilizar evidencias para tomar decisiones pedagógicas?",
)

# Adenda 8.2: en bajo umbral las preguntas se formulan como indagación.
PREGUNTAS_RTC_BAJO_UMBRAL = (
    "¿Qué evidencias adicionales necesitamos para confirmar los cambios que reconocemos entre la versión inicial y la versión posterior de nuestras experiencias?",
    "¿Qué evidencias de la implementación nos permitirían verificar si las decisiones modificadas respondieron a lo ocurrido en el aula?",
    "¿Qué información complementaria necesitamos para saber en qué aspectos los cambios favorables se sostienen de manera consistente?",
    "¿Qué otras fuentes nos ayudarían a comprender mejor los aspectos que todavía muestran resultados por consolidar?",
    "¿Qué evidencias adicionales necesitamos para confirmar si estas observaciones se extienden al conjunto de docentes de la institución?",
)

# Aspectos del Reporte Institucional 2 que la base de datos aún no permite
# calcular (se informan como no disponibles, nunca se reconstruyen).
SECCIONES_NO_DISPONIBLES = (
    "dificultades_persistentes",
    "naturaleza_de_los_cambios",
    "logros_consolidados",
)


def construir_advertencia_cobertura(n: int, umbral: int, n_docentes: int, pct: float) -> str:
    """Adenda 8.1 -- texto literal para RI 2."""
    return (
        f"Este reporte se elaboró con {n} de las {umbral} comparaciones válidas entre la Versión 1 y la "
        f"Versión 2 requeridas para representar a su institución ({n} de {n_docentes} docentes participantes; "
        f"{_fmt_num(pct)} %). Los resultados describen únicamente las comparaciones revisadas y no pueden "
        "generalizarse al conjunto de docentes. Por ello, los hallazgos de este reporte deben tratarse solo "
        "como hipótesis por contrastar con otras fuentes de su institución —evidencias de implementación, "
        "matrices de análisis, evidencias de seguimiento y acompañamiento— antes de utilizarse en la RTC 2."
    )


def construir_texto_confidencial(n: int, n_docentes: int) -> str:
    """Adenda 8.3 -- texto literal para COB_CONFIDENCIAL."""
    return (
        f"Su institución cuenta con {n} evidencias válidas de {n_docentes} docentes participantes. Para "
        "proteger la confidencialidad de los resultados de cada docente, el reporte no presenta "
        "distribuciones, porcentajes ni lectura por aspecto cuando se dispone de menos de cinco evidencias "
        "válidas o cuando la institución tiene menos de cinco docentes participantes. Le sugerimos promover "
        "la entrega de las evidencias pendientes y apoyar su Diagnóstico en otras fuentes de su institución."
    )


# =============================================================================
# Redacción del texto narrativo (LLM)
# =============================================================================

_PATRON_CIFRA_O_CODIGO = re.compile(r"\d|\b[CB][1-5]\b")


def _quitar_oraciones_con_cifras(texto: str) -> str:
    """El LLM no debe escribir cifras ni códigos internos (las cifras las
    escribe Python). Por si lo hace igual, se descartan esas oraciones."""
    oraciones = re.split(r"(?<=[.!?])\s+", (texto or "").strip())
    return " ".join(o for o in oraciones if o and not _PATRON_CIFRA_O_CODIGO.search(o)).strip()


def _explicacion_fallback(avance: Optional[dict]) -> str:
    if not avance:
        return "Esta lectura se reporta a partir de los datos disponibles."
    return "Este es un cambio descriptivo entre las dos versiones y no permite atribuir causas por sí solo."


def _sintesis_fallback(v: dict) -> str:
    base = (
        f"Las comparaciones entre las Sesiones V1 y V2 de {v.get('nombre_ie') or 'la institución'} reúnen "
        f"{v['n_comparaciones_validas']} de {v['n_docentes_activos']} docentes participantes. "
    )
    if v["prudente"]:
        return base + (
            "Lo observado en las comparaciones revisadas debe tratarse como hipótesis por contrastar con las "
            "evidencias de implementación, las matrices de análisis y las evidencias de seguimiento durante la RTC 2."
        )
    return base + (
        "Estos resultados describen cambios y no explican sus causas; conviene contrastarlos durante la RTC 2 con "
        "las evidencias de implementación, las matrices de análisis y las evidencias de seguimiento."
    )


def _construir_prompt_secciones_narrativas(v: dict) -> str:
    """Prompt para las DOS piezas que requieren redacción pedagógica: la
    explicación de cada aspecto (sección 3) y la síntesis (sección 10).
    Todo lo demás se arma en Python. Las cifras y la oración de contraste
    las escribe Python; el LLM no debe incluir ninguna cifra."""
    prudente = v["prudente"]
    avance_por_id = {a["criterio_id"]: a for a in v["avance_por_aspecto"]}

    def _tendencia(a: Optional[dict]) -> str:
        if not a:
            return "sin dato de avance"
        return "aumentó" if a["avance_pp"] > 0 else ("se mantuvo" if a["avance_pp"] == 0 else "disminuyó")

    aspectos_texto = "\n".join(
        f"- id \"{c['criterio_id']}\" · {_nombre_criterio(c['criterio_id'])} (evalúa: "
        f"{CRITERIOS.get(c['criterio_id'], {}).get('que_evalua', '')}). "
        f"La proporción favorable (Logrado+Destacado) entre V1 y V2 {_tendencia(avance_por_id.get(c['criterio_id']))}. "
        f"El sistema ya redactó, por fuera de tu texto, estas oraciones con las cifras: "
        f"\"{_primera_oracion_interpretacion(c, prudente)} {_segunda_oracion_avance(avance_por_id.get(c['criterio_id']))}\""
        for c in v["distribucion_v2_por_aspecto"]
    )
    con_avance = [a["nombre"] for a in v["avance_por_aspecto"] if a["avance_pp"] > 0]
    sin_avance = [a["nombre"] for a in v["avance_por_aspecto"] if a["avance_pp"] <= 0]
    ids = ", ".join(f'"{c["criterio_id"]}"' for c in v["distribucion_v2_por_aspecto"])

    if prudente:
        modo = (
            "MODO PRUDENTE (cobertura bajo el umbral): todo lo que escribas es una HIPÓTESIS sobre las "
            "comparaciones revisadas. Usa siempre \"en las comparaciones revisadas\". PROHIBIDO escribir "
            "\"la institución presenta\", \"la mayoría\" o \"los docentes de la IE\" y cualquier "
            "generalización al conjunto de docentes. No digas que un aspecto \"es el que requiere mayor atención\"."
        )
    else:
        modo = (
            "MODO REGULAR: describe lo que muestran las comparaciones; no atribuyas los cambios a una causa "
            "única ni presentes hipótesis como hechos."
        )

    return f"""
Eres un redactor pedagógico para el sistema educativo peruano. Redactas piezas de un reporte institucional
dirigido al equipo directivo de una IE, a partir de datos YA CALCULADOS. No recalcules ni inventes datos.

{modo}

Contexto: el reporte compara las Sesiones de aprendizaje V1 y V2 de los docentes con comparación válida
({v['n_comparaciones_validas']} de {v['n_docentes_activos']} docentes). Es insumo para la RTC 2.

Aspectos evaluados:
{aspectos_texto}

Aspectos cuya proporción favorable aumentó (orden oficial): {', '.join(con_avance) or '(ninguno)'}
Aspectos cuya proporción favorable se mantuvo o disminuyó (orden oficial): {', '.join(sin_avance) or '(ninguno)'}

Responde ÚNICAMENTE con un objeto JSON válido, sin texto antes ni después, con esta forma exacta:
{{
  "interpretaciones": {{ {ids}: "explicación pedagógica" }},
  "sintesis_institucional": "texto de síntesis"
}}

Para cada "interpretaciones.<id>": escribe 1 o 2 oraciones de explicación pedagógica de lo que esa situación
significa para las decisiones de planificación, apoyada en lo que "evalúa" ese aspecto y coherente con que la
proporción favorable aumentó, se mantuvo o disminuyó. NO repitas ni reescribas las oraciones con cifras que el
sistema ya redactó, NO escribas ninguna cifra, porcentaje ni número, y NO cierres con una oración de contraste
con otras fuentes (el sistema la agrega). Empieza directo con la explicación (ej. "Esto sugiere que...").

Para "sintesis_institucional": un solo párrafo (de 4 a 6 oraciones) que (1) reconozca, sin cifras, los
aspectos en los que la proporción favorable aumentó (si los hay), (2) mencione en conjunto y en el orden dado,
sin jerarquizarlos, los aspectos en los que se mantuvo o disminuyó (si los hay), (3) aclare que estos
resultados describen cambios y no explican sus causas y (4) cierre indicando que la RTC 2 es la oportunidad
de contrastar esta información con las evidencias de implementación, las matrices de análisis y las
evidencias de seguimiento. Si ambas listas están vacías, limítate a la cobertura y al cierre.

Reglas estrictas, sin excepción:
- NUNCA uses las palabras "patrón", "patrones", "alerta", "alertas", "semaforización", "nivel global" ni "clúster".
- NUNCA muestres códigos internos (C1-C5, B1-B5, BC1-BC4, V1V2_...) ni nombres de campos: usa el nombre del aspecto.
- NUNCA uses lenguaje causal ("esto provoca", "la causa es", "debido a que", "gracias a").
- NUNCA formules recomendaciones ni menciones a docentes individuales; el destinatario es el directivo y todo es agregado.
- NUNCA sugieras acciones, estrategias ni próximos pasos concretos; el reporte describe, la decisión es del directivo.
- NUNCA presentes un aspecto como "el más urgente", "el prioritario" ni compares aspectos entre sí.
- NUNCA menciones tipo de cambio (mejora real, mejora formal, retroceso), dificultades persistentes, logros consolidados
  ni progresión formal: esa información no está disponible en este reporte.
- Usa "propósito de aprendizaje", nunca "objetivos de aprendizaje".
- Prosa fluida, sin markdown, sin viñetas, sin títulos ni etiquetas, sin placeholders entre corchetes.
""".strip()


def _completar_narrativas(variables: dict, data: dict) -> dict:
    """Arma el texto final de cada aspecto: oración de cifras (Python) +
    oración de avance (Python) + explicación (LLM, sin cifras) + contraste
    (Python). Completa con fallback solo los campos que falten."""
    prudente = variables["prudente"]
    avance_por_id = {a["criterio_id"]: a for a in variables["avance_por_aspecto"]}
    crudas = data.get("interpretaciones") or {}
    faltantes = []

    interpretaciones = {}
    for c in variables["distribucion_v2_por_aspecto"]:
        cid = c["criterio_id"]
        a = avance_por_id.get(cid)
        explicacion = _quitar_oraciones_con_cifras(crudas.get(cid) or "")
        if not explicacion:
            explicacion = _explicacion_fallback(a)
            faltantes.append(f"interpretaciones.{cid}")
        interpretaciones[cid] = " ".join(filter(None, [
            _primera_oracion_interpretacion(c, prudente),
            _segunda_oracion_avance(a),
            explicacion,
            _oracion_contraste(cid, prudente),
        ]))

    sintesis = _quitar_oraciones_con_cifras(data.get("sintesis_institucional") or "")
    if not sintesis:
        sintesis = _sintesis_fallback(variables)
        faltantes.append("sintesis_institucional")

    if faltantes:
        log.warning(
            "llm_narrativas_incompletas",
            clave_ie=variables.get("clave_ie"),
            campos_completados_con_fallback=faltantes,
        )
    return {"interpretaciones": interpretaciones, "sintesis_institucional": sintesis}


def _narrativas_fallback(variables: dict) -> dict:
    """Degradación total (el LLM no devolvió JSON parseable en 2 intentos)."""
    return _completar_narrativas(variables, {})


async def redactar_narrativas(variables: dict) -> dict:
    """Llama al LLM UNA vez por reporte (con un reintento). Si falla dos
    veces, degrada a texto plantillado determinista."""
    prompt = _construir_prompt_secciones_narrativas(variables)

    for intento in (1, 2):
        inicio = time.monotonic()
        try:
            crudo = await generar_texto_narrativo(prompt, formato_json=True)
            data = json.loads(crudo)
            if not isinstance(data, dict):
                raise ValueError("la respuesta del LLM no es un objeto JSON")
        except Exception:
            log.error(
                "llm_narrativas_fallidas",
                clave_ie=variables.get("clave_ie"),
                intento=intento,
                duracion_ms=round((time.monotonic() - inicio) * 1000),
                exc_info=True,
            )
            if intento == 2:
                return _narrativas_fallback(variables)
            continue

        log.info(
            "llm_narrativas_generadas",
            clave_ie=variables.get("clave_ie"),
            duracion_ms=round((time.monotonic() - inicio) * 1000),
        )
        return _completar_narrativas(variables, data)


# =============================================================================
# Orquestador de generación
# =============================================================================

def _campos_cobertura(estado: dict, clasificacion: dict) -> dict:
    n, total = estado["n_comparaciones_validas"], estado["n_docentes_activos"]
    return {
        "n_docentes_activos": total,
        "n_no_aprobaron_c2": estado.get("n_no_aprobaron_c2", 0),
        "n_comparaciones_validas": n,
        "n_sin_referencia_v1": estado["n_sin_referencia_v1"],
        "n_no_validas": estado["n_no_validas"],
        "n_sin_entrega_v2": estado["n_sin_entrega_v2"],
        "pct_cobertura": round(100 * n / total, 1) if total else 0.0,
        "muestra_suficiente": n >= UMBRAL_MUESTRA_INSUFICIENTE,
        "banda_aplicada": clasificacion["banda_aplicada"],
        "umbral_requerido": clasificacion["umbral_requerido"],
        "umbral_alcanzado": clasificacion["umbral_alcanzado"],
        "estado_cobertura": clasificacion["estado_cobertura"],
        "advertencia_hipotesis": clasificacion["estado_cobertura"] == "COB_BAJO_UMBRAL",
    }


async def generar_reporte(ie: dict, estado: dict) -> dict:
    """Genera el Reporte Institucional 2 para la IE a partir del estado ya
    calculado (calcular_estado_ie). Según estado_cobertura:

      COB_CONFIDENCIAL  solo estado de cobertura (adenda 8.3): sin
                        distribuciones, porcentajes, frecuencias ni lectura
                        por aspecto, y sin llamar al LLM.
      COB_BAJO_UMBRAL   reporte completo con advertencia de cobertura,
                        hipótesis y redacción prudente (adenda 8.1 y 8.2).
      COB_SUFICIENTE    reporte regular."""
    ahora_utc = datetime.now(timezone.utc)
    clasificacion = _clasificar_cobertura(estado["n_comparaciones_validas"], estado["n_docentes_activos"])
    cobertura = _campos_cobertura(estado, clasificacion)
    estado_cob = cobertura["estado_cobertura"]

    base = {
        "reporte_id": f"{ie['clave_ie']}-{ahora_utc.strftime('%Y%m%dT%H%M%S')}",
        "clave_ie": ie["clave_ie"],
        "institucion_id": ie["clave_ie"],
        "nombre_ie": ie["nombre_ie"],
        "tipo_reporte": TIPO_REPORTE,
        "hito_asociado": "HITO_2_DIRECTIVO",
        "version_evidencia": "V1_V2",
        "origen_version": "SOLICITUD",
        "fecha_solicitud": ahora_utc.isoformat(),
        "fecha_generacion": ahora_utc.isoformat(),
        "identificacion_individual_incluida": False,
        "version_calculo": VERSION_CALCULO,
        **cobertura,
    }

    if estado_cob == "COB_CONFIDENCIAL":
        # Ni siquiera el porcentaje de cobertura (adenda 6 y 13).
        base.pop("pct_cobertura")
        base.pop("n_sin_referencia_v1")
        base["texto_confidencial"] = construir_texto_confidencial(
            estado["n_comparaciones_validas"], estado["n_docentes_activos"]
        )
        log.info("reporte_confidencial", clave_ie=ie["clave_ie"], n=estado["n_comparaciones_validas"])
        return base

    # La advertencia y la redacción de hipótesis aplican SOLO cuando la
    # cobertura queda por debajo del umbral de la banda (COB_BAJO_UMBRAL).
    # Con cobertura completa (todo el universo con comparación válida) o
    # suficiente no hay nada que tratar como hipótesis.
    prudente = estado_cob == "COB_BAJO_UMBRAL"
    comparaciones = estado["comparaciones"]

    distribucion = calcular_distribucion_v2(comparaciones)
    avance = calcular_avance_por_criterio(comparaciones)
    avance_por_id = {a["criterio_id"]: a for a in avance}

    variables = {
        "clave_ie": ie["clave_ie"],
        "nombre_ie": ie["nombre_ie"],
        "n_comparaciones_validas": estado["n_comparaciones_validas"],
        "n_docentes_activos": estado["n_docentes_activos"],
        "prudente": prudente,
        "distribucion_v2_por_aspecto": distribucion,
        "avance_por_aspecto": avance,
    }
    narrativas = await redactar_narrativas(variables)

    distribucion_visible = [{**c, "nombre": _nombre_criterio(c["criterio_id"])} for c in distribucion]
    # Adenda 8.2: en COB_BAJO_UMBRAL se omite la lectura por nivel. En los
    # demás casos se evalúa, y los niveles con menos de 5 casos se suprimen.
    lectura_por_nivel = (
        None if estado_cob == "COB_BAJO_UMBRAL" else calcular_lectura_por_nivel(comparaciones)
    )

    base.update({
        "advertencia_cobertura": (
            construir_advertencia_cobertura(
                estado["n_comparaciones_validas"],
                clasificacion["umbral_requerido"],
                estado["n_docentes_activos"],
                cobertura["pct_cobertura"],
            ) if prudente else None
        ),
        "panorama_hallazgos": construir_hallazgos_panorama(estado, avance, distribucion, prudente),
        "distribucion_v2_por_aspecto": distribucion_visible,
        "interpretaciones_por_aspecto": narrativas["interpretaciones"],
        "avance_por_aspecto": avance,
        "enfasis_lectura": calcular_enfasis_lectura(avance_por_id, prudente),
        "secciones_no_disponibles": list(SECCIONES_NO_DISPONIBLES),
        # None = no se muestra (bajo umbral, adenda 8.2); {"niveles": []} =
        # se evaluó pero ningún nivel tiene tamaño suficiente.
        "lectura_por_nivel": lectura_por_nivel,
        "aspectos_contraste": calcular_aspectos_contraste(avance, distribucion, prudente),
        "preguntas_rtc": list(PREGUNTAS_RTC_BAJO_UMBRAL if prudente else PREGUNTAS_RTC),
        "sintesis_institucional": narrativas["sintesis_institucional"],
        "titulo_sintesis": (
            "Hipótesis para el análisis de la RTC 2" if prudente
            else "Síntesis del análisis para la RTC 2"
        ),
    })
    return base


# =============================================================================
# Caché (tablas compartidas con RI1, distinguidas por tipo_reporte)
# =============================================================================

async def obtener_reporte_guardado(clave_ie: str) -> Optional[dict]:
    """Fila de f4_reportes_directivo de esta IE para RI2 (o None)."""
    query = """
        SELECT id, cod_modular, estado, n_docentes_evaluados, reporte_json,
               fecha_generacion, fecha_actualizacion
        FROM f4_reportes_directivo
        WHERE cod_modular = %(clave)s AND tipo_reporte = %(tipo)s
    """
    async with pool_cache.connection() as conn:
        return await (await conn.execute(query, {"clave": clave_ie, "tipo": TIPO_REPORTE})).fetchone()


async def guardar_estado_reporte(
    clave_ie: str,
    estado: str,
    n_docentes_evaluados: int,
    reporte_json: Optional[dict] = None,
) -> int:
    """Upsert de la fila de RI2 para esta IE (una por clave + tipo_reporte).
    n_docentes_evaluados guarda las comparaciones válidas V1->V2."""
    query = """
        INSERT INTO f4_reportes_directivo (cod_modular, tipo_reporte, estado, n_docentes_evaluados, reporte_json)
        VALUES (%(clave)s, %(tipo)s, %(estado)s, %(n)s, %(reporte)s)
        ON CONFLICT (cod_modular, tipo_reporte) DO UPDATE SET
            estado = EXCLUDED.estado,
            n_docentes_evaluados = EXCLUDED.n_docentes_evaluados,
            reporte_json = COALESCE(EXCLUDED.reporte_json, f4_reportes_directivo.reporte_json),
            fecha_actualizacion = now()
        RETURNING id
    """
    async with pool_cache.connection() as conn:
        row = await (await conn.execute(query, {
            "clave": clave_ie,
            "tipo": TIPO_REPORTE,
            "estado": estado,
            "n": n_docentes_evaluados,
            "reporte": Jsonb(reporte_json) if reporte_json is not None else None,
        })).fetchone()
        await conn.commit()
    return row["id"]


async def registrar_consulta(
    clave_ie: str, userid_directivo: int, courseid: int, cmid: int, id_reporte_directivo: int
) -> None:
    """Asocia un directivo con el reporte RI2 de su IE (un registro por
    clave + directivo + tipo_reporte; cada consulta actualiza la fecha)."""
    query = """
        INSERT INTO f4_consultas_directivo
            (cod_modular, userid_directivo, courseid, cmid, id_reporte_directivo, tipo_reporte)
        VALUES (%(clave)s, %(userid)s, %(courseid)s, %(cmid)s, %(id_reporte)s, %(tipo)s)
        ON CONFLICT (cod_modular, userid_directivo, tipo_reporte) DO UPDATE SET
            courseid = EXCLUDED.courseid,
            cmid = EXCLUDED.cmid,
            id_reporte_directivo = EXCLUDED.id_reporte_directivo,
            fecha_consulta = now()
    """
    async with pool_cache.connection() as conn:
        await conn.execute(query, {
            "clave": clave_ie,
            "userid": userid_directivo,
            "courseid": courseid,
            "cmid": cmid,
            "id_reporte": id_reporte_directivo,
            "tipo": TIPO_REPORTE,
        })
        await conn.commit()


# =============================================================================
# Puntos de entrada
# =============================================================================

def _reporte_vigente(guardado: Optional[dict], estado: dict) -> bool:
    """El reporte guardado sigue vigente mientras no cambien las
    comparaciones válidas ni N (adenda RE-08 y CP21: N se recalcula en cada
    solicitud). Si cambia alguno, se genera una versión nueva."""
    if guardado is None or guardado["estado"] != "generado" or not guardado["reporte_json"]:
        return False
    esperado = _clasificar_cobertura(estado["n_comparaciones_validas"], estado["n_docentes_activos"])
    return (
        guardado["reporte_json"].get("version_calculo") == VERSION_CALCULO
        and guardado["reporte_json"].get("estado_cobertura") == esperado["estado_cobertura"]
        and guardado["n_docentes_evaluados"] == estado["n_comparaciones_validas"]
        and guardado["reporte_json"].get("n_docentes_activos") == estado["n_docentes_activos"]
        and guardado["reporte_json"].get("n_no_aprobaron_c2") == estado.get("n_no_aprobaron_c2", 0)
    )


async def consultar_reporte_existente(userid: int) -> dict:
    """
    GET /reporte_ri2/consultar -- endpoint rápido de solo lectura (nunca
    toca el LLM). Retorna {"accion": "mostrar" | "generar", "reporte": ... |
    None, + campos de cobertura}:

      - Hay reporte vigente -> accion="mostrar".
      - Estado confidencial (N < 5 o menos de 5 comparaciones válidas) ->
        accion="mostrar" con solo el estado de cobertura.
      - No hay reporte, o cambiaron las comparaciones válidas / N ->
        accion="generar": el frontend muestra el estado de cobertura y, solo
        cuando el directivo pulsa "Solicitar Reporte Institucional" (adenda
        RE-01), llama a POST /reporte_ri2/generar.
    """
    inicio = time.monotonic()
    log.info("reporte_consultado", userid=userid)

    ie = await resolver_ie(userid)
    estado = await calcular_estado_ie(ie)
    guardado = await obtener_reporte_guardado(ie["clave_ie"])
    clasificacion = _clasificar_cobertura(estado["n_comparaciones_validas"], estado["n_docentes_activos"])

    n_validas, n_activos = estado["n_comparaciones_validas"], estado["n_docentes_activos"]
    campos = {
        "nombre_ie": ie["nombre_ie"],
        "n_comparaciones_validas": n_validas,
        "n_docentes_activos": n_activos,
        "n_no_aprobaron_c2": estado["n_no_aprobaron_c2"],
        "n_sin_referencia_v1": estado["n_sin_referencia_v1"],
        "n_no_validas": estado["n_no_validas"],
        "n_sin_entrega_v2": estado["n_sin_entrega_v2"],
        "pct_cobertura": round(100 * n_validas / n_activos, 1) if n_activos else 0.0,
        "minimo_comparaciones": UMBRAL_MUESTRA_INSUFICIENTE,
        **clasificacion,
    }
    duracion_ms = round((time.monotonic() - inicio) * 1000)

    if _reporte_vigente(guardado, estado):
        log.info("consulta_mostrar_vigente", clave_ie=ie["clave_ie"], duracion_ms=duracion_ms)
        return {"accion": "mostrar", "reporte": guardado["reporte_json"], **campos}

    if clasificacion["estado_cobertura"] == "COB_CONFIDENCIAL":
        # Solo se emite el estado de cobertura: no hay contenido que generar
        # ni LLM, así que no hace falta confirmación ni guardar nada.
        log.info("consulta_confidencial", clave_ie=ie["clave_ie"], duracion_ms=duracion_ms)
        return {"accion": "mostrar", "reporte": await generar_reporte(ie, estado), **campos}

    log.info(
        "consulta_debe_generar",
        clave_ie=ie["clave_ie"],
        motivo="sin_registro_previo" if guardado is None else "cambio_en_comparaciones_o_N",
        duracion_ms=duracion_ms,
    )
    return {"accion": "generar", "reporte": None, **campos}


async def obtener_o_generar_reporte(userid: int, courseid: int, cmid: int) -> dict:
    """
    POST /reporte_ri2/generar -- pipeline completo:
      1. Resuelve la IE del directivo (cod_institucional o clave compuesta).
      2. Calcula el estado actual (docentes aprobados del Curso 1, V1/V2).
      3. Si el reporte guardado sigue vigente, lo sirve; si no, genera uno
         nuevo (con LLM salvo COB_CONFIDENCIAL) y lo guarda.
      4. Registra la asociación directivo-reporte (courseid/cmid:
         trazabilidad).
    """
    inicio = time.monotonic()
    log.info("reporte_solicitado", userid=userid, courseid=courseid, cmid=cmid)

    ie = await resolver_ie(userid)
    estado = await calcular_estado_ie(ie)
    guardado = await obtener_reporte_guardado(ie["clave_ie"])

    if _reporte_vigente(guardado, estado):
        log.info("cache_hit", clave_ie=ie["clave_ie"], n=estado["n_comparaciones_validas"])
        id_reporte = guardado["id"]
        reporte = guardado["reporte_json"]
    else:
        log.info("cache_miss_o_reintento", clave_ie=ie["clave_ie"])
        reporte = await generar_reporte(ie, estado)
        id_reporte = await guardar_estado_reporte(
            ie["clave_ie"], "generado", estado["n_comparaciones_validas"], reporte
        )

    await registrar_consulta(ie["clave_ie"], userid, courseid, cmid, id_reporte)

    log.info(
        "reporte_completado",
        userid=userid,
        clave_ie=ie["clave_ie"],
        estado_cobertura=reporte.get("estado_cobertura"),
        duracion_ms=round((time.monotonic() - inicio) * 1000),
    )
    return reporte
