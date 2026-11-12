"""
Pruebas del Reporte Institucional 2 (F4 - EBR 2026, V1 -> V2).

Cubren las funciones puras de core.py y la generación completa con el LLM
simulado (nunca se abren los pools ni se llama a Azure OpenAI). Los casos
CP13-CP18, CP21, CP24 y CP25 vienen de la Adenda F4-ADENDA-EMISION-RI-EBRJEC26
(sección 14); los ejemplos numéricos del avance, del Modelo y Especificación
del Producto Visible RI2.
"""

import asyncio
import json
import re
from datetime import datetime

import pytest

import core


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cmp(uid, v1, v2, nivel="Primaria"):
    """Comparación válida con el mismo nivel en los 5 aspectos (v1/v2 son
    str) o con un dict {criterio: nivel}."""
    def _niveles(x):
        return dict(x) if isinstance(x, dict) else {c: x for c in range(1, 6)}
    return {"user_id": uid, "nivel_educativo": nivel, "v1": _niveles(v1), "v2": _niveles(v2)}


def _fila(uid, version, criterio, nivel, status="success", run="r1", completed=None, cmid=None):
    if cmid is None:
        cmid = core.CMIDS_V1_POR_CURSO[2918][0] if version == "V1" else core.CMIDS_V2_POR_CURSO[2918][0]
    return {
        "run_id": f"{uid}-{version}-{run}", "user_id": uid, "cmid": cmid, "status": status,
        "completed_at": completed, "criterion_index": criterio, "nivel_obtenido": nivel,
    }


def _filas_evidencia(uid, version, nivel, **kw):
    return [_fila(uid, version, c, nivel, **kw) for c in range(1, 6)]


def _estado(comparaciones, n_docentes, sin_ref=0, no_validas=0, no_aprobaron_c1=0):
    n = len(comparaciones)
    return {
        "comparaciones": comparaciones,
        "n_docentes_activos": n_docentes,
        "n_no_aprobaron_c1": no_aprobaron_c1,
        "n_comparaciones_validas": n,
        "n_sin_referencia_v1": sin_ref,
        "n_no_validas": no_validas,
        "n_sin_entrega_v2": n_docentes - n - sin_ref - no_validas,
    }


IE = {"clave_ie": "IE-TEST", "cod_institucional": "IE-TEST", "nombre_ie": "IE de prueba",
      "ugel": "UGEL X", "region": "REGION", "distrito": "D"}


def _correr(coro):
    return asyncio.run(coro)


@pytest.fixture
def llm_ok(monkeypatch):
    """LLM simulado que devuelve un JSON válido para cualquier aspecto."""
    async def fake(prompt, formato_json=False):
        ids = re.findall(r'id "(C[1-5])"', prompt)
        return json.dumps({
            "interpretaciones": {i: "Esto sugiere que las decisiones de planificación se ajustaron." for i in ids},
            "sintesis_institucional": "Las comparaciones muestran cambios en varios aspectos. Estos resultados no explican sus causas.",
        })
    monkeypatch.setattr(core, "generar_texto_narrativo", fake)


@pytest.fixture
def llm_roto(monkeypatch):
    async def fake(prompt, formato_json=False):
        raise RuntimeError("LLM caído")
    monkeypatch.setattr(core, "generar_texto_narrativo", fake)


# ---------------------------------------------------------------------------
# Configuración de cmids
# ---------------------------------------------------------------------------

def test_cmids_v1_v2_completos_y_sin_cruce():
    assert core.CMIDS_V1_POR_CURSO.keys() == core.CMIDS_V2_POR_CURSO.keys()
    assert len(core.CMIDS_V2_POR_CURSO) == 24
    v1, v2 = set(core._aplanar_cmids(core.CMIDS_V1_POR_CURSO)), set(core._aplanar_cmids(core.CMIDS_V2_POR_CURSO))
    assert len(v2) == 24 and not (v1 & v2)
    mapa = core._mapa_cmid_a_version()
    assert mapa[206189] == "V1" and mapa[206728] == "V2" and mapa[217674] == "V2"


# ---------------------------------------------------------------------------
# Cobertura por bandas (adenda)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("validas,n,banda,umbral,estado", [
    (5, 96, "B5", 58, "COB_BAJO_UMBRAL"),         # CP13
    (28, 40, "B3", 28, "COB_SUFICIENTE"),         # CP14
    (27, 40, "B3", 28, "COB_BAJO_UMBRAL"),        # CP15
    (10, 12, "B1", 11, "COB_BAJO_UMBRAL"),        # CP16 (⌈10,2⌉ = 11)
    (4, 30, "B2", 24, "COB_CONFIDENCIAL"),        # CP17
    (4, 4, "B0", None, "COB_CONFIDENCIAL"),       # N < 5: siempre confidencial, aun con 100 % de cobertura
    (3, 4, "B0", None, "COB_CONFIDENCIAL"),
    (1, 1, "B0", None, "COB_CONFIDENCIAL"),
    (0, 0, "B0", None, "COB_CONFIDENCIAL"),
    (1, 56, "B4", 37, "COB_CONFIDENCIAL"),        # N >= 5 con < 5 comparaciones válidas
    (4, 56, "B4", 37, "COB_CONFIDENCIAL"),
    (6, 60, "B4", 39, "COB_BAJO_UMBRAL"),         # CP24
    (46, 96, "B5", 58, "COB_BAJO_UMBRAL"),        # CP25
    (17, 20, "B1", 17, "COB_SUFICIENTE"),         # 85 % de 20 = 17 exacto (sin error de flotante)
    (5, 5, "B1", 5, "COB_SUFICIENTE"),
])
def test_clasificar_cobertura(validas, n, banda, umbral, estado):
    r = core._clasificar_cobertura(validas, n)
    assert (r["banda_aplicada"], r["umbral_requerido"], r["estado_cobertura"]) == (banda, umbral, estado)
    assert r["umbral_alcanzado"] == (estado == "COB_SUFICIENTE")


def test_cp21_cambio_de_banda_por_retiro():
    assert core._clasificar_cobertura(0, 51)["umbral_requerido"] == 34   # B4: ⌈0,65 × 51⌉
    r = core._clasificar_cobertura(0, 50)
    assert (r["banda_aplicada"], r["umbral_requerido"]) == ("B3", 35)    # B3: 0,70 × 50


# ---------------------------------------------------------------------------
# IE: clave y filtro
# ---------------------------------------------------------------------------

def test_clave_ie_prefiere_cod_institucional_y_cae_a_clave_compuesta():
    assert core._armar_clave_ie(" 0123456 ", "IE", "UGEL", "REG", "DIST") == "0123456"
    assert core._armar_clave_ie(None, "IE", "UGEL", "REG", "DIST") == "IE|UGEL|REG|DIST"
    assert core._armar_clave_ie("  ", "TRILCE", "UGEL ANDAHUAYLAS", "APURIMAC", None) == "TRILCE|UGEL ANDAHUAYLAS|APURIMAC|"


def test_filtro_ie_por_codigo_o_por_cuatro_columnas():
    sql, params = core._filtro_ie_sql({"cod_institucional": " 123 ", "nombre_ie": "X"})
    assert "cod_institucional" in sql and "nombre_ie" not in sql and params == {"cod_institucional": "123"}

    sql, params = core._filtro_ie_sql({"cod_institucional": None, "nombre_ie": "X", "ugel": "U", "region": "R", "distrito": None})
    assert all(k in sql for k in ("nombre_ie", "ugel", "región", "distrito"))
    assert params == {"nombre_ie": "X", "ugel": "U", "region": "R", "distrito": None}


# ---------------------------------------------------------------------------
# Emparejamiento V1 -> V2
# ---------------------------------------------------------------------------

def test_seleccion_toma_ultimo_intento_exitoso_y_completo():
    filas = (
        _filas_evidencia(1, "V2", "inicio", run="a", completed=datetime(2026, 10, 1))
        + _filas_evidencia(1, "V2", "logrado", run="b", completed=datetime(2026, 10, 5))
        + _filas_evidencia(1, "V2", "destacado", status="failed", run="c", completed=datetime(2026, 10, 9))
    )
    ev = core.seleccionar_evidencia_por_version(filas)[1]["V2"]
    assert ev["status"] == "success" and ev["completa"]
    assert set(ev["niveles"].values()) == {"logrado"}      # ni el viejo ni el 'failed' posterior


def test_seleccion_marca_incompleta_la_carga_en_curso():
    filas = _filas_evidencia(1, "V2", "logrado")
    filas[2]["nivel_obtenido"] = None
    ev = core.seleccionar_evidencia_por_version(filas)[1]["V2"]
    assert ev["status"] == "success" and not ev["completa"] and ev["niveles"] == {}


def test_clasificacion_de_comparaciones_suma_n_docentes():
    docentes = {1: "Primaria", 2: "Inicial", 3: "Secundaria", 4: "Primaria", 5: "Primaria", 6: "Primaria"}
    filas = []
    filas += _filas_evidencia(1, "V1", "en_desarrollo") + _filas_evidencia(1, "V2", "logrado")            # válida
    filas += _filas_evidencia(2, "V2", "logrado")                                                         # sin V1
    filas += _filas_evidencia(3, "V1", "logrado") + _filas_evidencia(3, "V2", "inicio", status="validation_failed")  # V2 no válida
    filas += _filas_evidencia(4, "V1", "logrado") + _filas_evidencia(4, "V2", "inicio", status="failed")  # V2 failed
    filas += _filas_evidencia(5, "V1", "logrado") + _filas_evidencia(5, "V2", "logrado", status="success")
    filas = [f for f in filas if not (f["user_id"] == 5 and f["criterion_index"] == 3)] + [_fila(5, "V2", 3, None)]  # V2 incompleta
    filas += _filas_evidencia(99, "V1", "logrado") + _filas_evidencia(99, "V2", "logrado")                # no es docente activo

    est = core.clasificar_comparaciones(docentes, core.seleccionar_evidencia_por_version(filas), n_no_aprobaron_c1=7)

    assert est["n_docentes_activos"] == 6 and est["n_no_aprobaron_c1"] == 7   # los no aprobados no entran a la suma
    assert est["n_comparaciones_validas"] == 1 and est["comparaciones"][0]["user_id"] == 1
    assert est["n_sin_referencia_v1"] == 1
    assert est["n_no_validas"] == 1
    assert est["n_sin_entrega_v2"] == 3      # failed + incompleta + sin ninguna fila (6)
    assert (est["n_comparaciones_validas"] + est["n_sin_referencia_v1"] + est["n_no_validas"] + est["n_sin_entrega_v2"]) == 6


def test_v2_valida_con_v1_fallida_es_sin_referencia():
    docentes = {1: "Primaria"}
    filas = _filas_evidencia(1, "V1", "logrado", status="validation_failed") + _filas_evidencia(1, "V2", "logrado")
    est = core.clasificar_comparaciones(docentes, core.seleccionar_evidencia_por_version(filas))
    assert est["n_sin_referencia_v1"] == 1 and est["n_comparaciones_validas"] == 0


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------

def test_avance_ejemplo_del_modelo_demanda_cognitiva():
    """Modelo RI2: demanda cognitiva e involucramiento activo, 20 trayectorias,
    35,0 % -> 60,0 % = +25,0 p.p."""
    comps = []
    for i in range(20):
        v1 = "logrado" if i < 7 else "inicio"          # 7/20 = 35 %
        v2 = "destacado" if i < 12 else "en_desarrollo"  # 12/20 = 60 %
        comps.append(_cmp(i, {3: v1}, {3: v2}))
    a = core.calcular_avance_por_criterio(comps)[0]
    assert a["criterio_id"] == "C3" and a["n"] == 20
    assert (a["pct_favorable_v1"], a["pct_favorable_v2"], a["avance_pp"]) == (35.0, 60.0, 25.0)
    assert (a["n_favorable_v1"], a["n_favorable_v2"]) == (7, 12)


def test_avance_usa_la_misma_poblacion_en_v1_y_v2():
    comps = [_cmp(1, {1: "logrado"}, {1: "logrado"}), _cmp(2, {1: "inicio"}, {1: "destacado"}),
             _cmp(3, {1: "inicio"}, {})]       # sin nivel V2 en C1 -> no entra a ese aspecto
    a = core.calcular_avance_por_criterio(comps)[0]
    assert a["n"] == 2 and a["pct_favorable_v1"] == 50.0 and a["pct_favorable_v2"] == 100.0 and a["avance_pp"] == 50.0


def test_avance_negativo_y_nulo():
    comps = [_cmp(1, {1: "logrado", 2: "logrado"}, {1: "inicio", 2: "logrado"})]
    por_id = {a["criterio_id"]: a for a in core.calcular_avance_por_criterio(comps)}
    assert por_id["C1"]["avance_pp"] == -100.0 and por_id["C2"]["avance_pp"] == 0.0


def test_distribucion_v2_y_agrupados():
    comps = [_cmp(i, "inicio", v2) for i, v2 in enumerate(["inicio", "en_desarrollo", "logrado", "destacado", "logrado"])]
    c = core.calcular_distribucion_v2(comps)[0]
    assert c["n"] == 5 and (c["pct_inicio"], c["pct_en_desarrollo"], c["pct_logrado"], c["pct_destacado"]) == (20, 20, 40, 20)
    assert (c["pct_inicio_en_desarrollo"], c["pct_logrado_destacado"]) == (40, 60)


def test_lectura_por_nivel_suprime_niveles_con_menos_de_5_casos():
    comps = [_cmp(i, "inicio", "logrado", "Primaria") for i in range(10)] + \
            [_cmp(100, "inicio", "destacado", "Inicial"), _cmp(101, "inicio", "logrado", "Secundaria")] + \
            [_cmp(200 + i, "inicio", "logrado", "Secundaria") for i in range(4)]
    r = core.calcular_lectura_por_nivel(comps)
    assert [(n["nivel_educativo"], n["n"]) for n in r["niveles"]] == [("Primaria", 10), ("Secundaria", 5)]
    assert r["hay_niveles_no_mostrados"] is True      # Inicial (1 caso) no se muestra


def test_lectura_por_nivel_sin_niveles_suprimidos_ni_docentes_sin_nivel():
    comps = [_cmp(i, "inicio", "logrado", "Primaria") for i in range(5)] + [_cmp(50, "inicio", "logrado", None)]
    r = core.calcular_lectura_por_nivel(comps)
    assert [n["n"] for n in r["niveles"]] == [5] and r["hay_niveles_no_mostrados"] is False


@pytest.mark.parametrize("crudo,esperado", [
    ("secundaria", "Secundaria"), ("Secundaria", "Secundaria"), (" PRIMARIA ", "Primaria"),
    ("Inicial", "Inicial"), ("NULL", None), (None, None), ("", None),
])
def test_normalizar_nivel_educativo(crudo, esperado):
    assert core._normalizar_nivel(crudo) == esperado


# ---------------------------------------------------------------------------
# Acceso: solo el directivo
# ---------------------------------------------------------------------------

def test_solo_el_directivo_puede_ver_el_reporte():
    fila_dir = {"cargo": " directivo ", "nombre_ie": "IE"}
    fila_doc = {"cargo": "DOCENTE", "nombre_ie": "IE"}
    assert core._elegir_fila_directivo([fila_doc, fila_dir], 1) is fila_dir
    with pytest.raises(core.NoEsDirectivoError):
        core._elegir_fila_directivo([fila_doc], 1)
    with pytest.raises(core.DirectivoNoEncontradoError):
        core._elegir_fila_directivo([], 1)


# ---------------------------------------------------------------------------
# Textos armados en Python
# ---------------------------------------------------------------------------

def test_primera_oracion_usa_conteo_exacto_y_adapta_ambito():
    c = {"n": 5, "n_inicio": 1, "n_en_desarrollo": 0, "n_logrado": 3, "n_destacado": 1,
         "pct_logrado_destacado": 80, "pct_inicio_en_desarrollo": 20}
    assert core._primera_oracion_interpretacion(c) == "En 4 de las 5 comparaciones válidas (80%) se observa Logrado o Destacado en la Sesión V2."
    assert "comparaciones revisadas" in core._primera_oracion_interpretacion(c, prudente=True)
    c2 = {"n": 5, "n_inicio": 1, "n_en_desarrollo": 2, "n_logrado": 2, "n_destacado": 0,
          "pct_logrado_destacado": 40, "pct_inicio_en_desarrollo": 60}
    assert "En 3 de las 5" in core._primera_oracion_interpretacion(c2) and "Inicio o En desarrollo" in core._primera_oracion_interpretacion(c2)


def test_formato_de_cifras():
    assert core._fmt_pct(83.333) == "83,3 %"
    assert core._fmt_pp(25.0) == "+25,0 p.p." and core._fmt_pp(-3.5) == "-3,5 p.p." and core._fmt_pp(0.0) == "0,0 p.p."


def test_segunda_oracion_avance():
    a = {"pct_favorable_v1": 35.0, "pct_favorable_v2": 60.0, "avance_pp": 25.0}
    assert "pasó de 35,0 % a 60,0 % (+25,0 p.p.)" in core._segunda_oracion_avance(a)
    assert "se mantuvo en 60,0 %" in core._segunda_oracion_avance({**a, "pct_favorable_v1": 60.0, "avance_pp": 0.0})
    assert core._segunda_oracion_avance(None) == ""


def test_se_descartan_oraciones_del_llm_con_cifras_o_codigos():
    texto = "Esto sugiere un ajuste. En 4 de 5 casos se observa mejora. El aspecto C3 cambió. Conviene mirar la práctica."
    assert core._quitar_oraciones_con_cifras(texto) == "Esto sugiere un ajuste. Conviene mirar la práctica."


def test_completar_narrativas_antepone_cifras_de_python_y_rellena_huecos():
    comps = [_cmp(i, "inicio", "logrado") for i in range(5)]
    dist = core.calcular_distribucion_v2(comps)
    avance = core.calcular_avance_por_criterio(comps)
    variables = {"clave_ie": "x", "nombre_ie": "IE", "n_comparaciones_validas": 5, "n_docentes_activos": 5,
                 "prudente": False, "distribucion_v2_por_aspecto": dist, "avance_por_aspecto": avance}
    data = {"interpretaciones": {"C1": "En 1 de 5 el LLM inventó una cifra. Esto sugiere un cambio sustantivo."}}
    out = core._completar_narrativas(variables, data)
    c1 = out["interpretaciones"]["C1"]
    assert c1.startswith("En 5 de las 5 comparaciones válidas (100%)")
    assert "inventó" not in c1 and "Esto sugiere un cambio sustantivo." in c1
    assert c1.endswith("conviene contrastarla con las evidencias de implementación y las matrices de análisis.")
    assert out["interpretaciones"]["C2"] and out["sintesis_institucional"]     # huecos -> fallback


# ---------------------------------------------------------------------------
# Generación completa (LLM simulado)
# ---------------------------------------------------------------------------

def _comparaciones_sinteticas(n=20):
    """n comparaciones: 60 % Primaria, 40 % Secundaria; V1 inicio -> V2 logrado
    en la mitad."""
    comps = []
    for i in range(n):
        nivel = "Primaria" if i < int(n * 0.6) else "Secundaria"
        comps.append(_cmp(i, "inicio", "logrado" if i % 2 == 0 else "en_desarrollo", nivel))
    return comps


def test_reporte_confidencial_no_expone_cifras_ni_llama_al_llm(llm_roto):
    estado = _estado(_comparaciones_sinteticas(4), n_docentes=30, no_validas=1, no_aprobaron_c1=12)
    r = _correr(core.generar_reporte(IE, estado))
    assert r["estado_cobertura"] == "COB_CONFIDENCIAL"
    assert r["n_no_aprobaron_c1"] == 12     # conteo de cobertura: sí se informa
    assert "Su institución cuenta con 4 evidencias válidas de 30 docentes participantes" in r["texto_confidencial"]
    prohibidas = {"distribucion_v2_por_aspecto", "avance_por_aspecto", "pct_cobertura", "panorama_hallazgos",
                  "interpretaciones_por_aspecto", "lectura_por_nivel", "n_sin_referencia_v1", "enfasis_lectura"}
    assert not (prohibidas & r.keys())
    assert r["identificacion_individual_incluida"] is False and r["tipo_reporte"] == "RI2"


def test_menos_de_5_docentes_es_confidencial_aunque_todos_tengan_comparacion(llm_roto):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(4), n_docentes=4, no_aprobaron_c1=0)))
    assert r["estado_cobertura"] == "COB_CONFIDENCIAL" and r["banda_aplicada"] == "B0"
    assert "Su institución cuenta con 4 evidencias válidas de 4 docentes participantes" in r["texto_confidencial"]
    assert not ({"distribucion_v2_por_aspecto", "avance_por_aspecto", "lectura_por_nivel", "pct_cobertura"} & r.keys())


def test_menos_de_5_docentes_sin_cobertura_total_es_confidencial(llm_roto):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(3), n_docentes=4)))
    assert r["estado_cobertura"] == "COB_CONFIDENCIAL" and "distribucion_v2_por_aspecto" not in r


def test_cobertura_completa_muestra_lectura_por_nivel_si_los_grupos_alcanzan(llm_ok):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(10), n_docentes=10)))   # 6 Primaria + 4 Secundaria
    assert r["estado_cobertura"] == "COB_SUFICIENTE"
    # Primaria (6) se muestra; Secundaria (4) se suprime por tener menos de 5 casos
    assert [(n["nivel_educativo"], n["n"]) for n in r["lectura_por_nivel"]["niveles"]] == [("Primaria", 6)]
    assert r["lectura_por_nivel"]["hay_niveles_no_mostrados"] is True


def test_reporte_bajo_umbral_con_advertencia_y_lenguaje_prudente(llm_ok):
    estado = _estado(_comparaciones_sinteticas(10), n_docentes=40, sin_ref=3)   # umbral 28
    r = _correr(core.generar_reporte(IE, estado))
    assert r["estado_cobertura"] == "COB_BAJO_UMBRAL" and r["advertencia_hipotesis"] and r["umbral_requerido"] == 28
    assert r["advertencia_cobertura"].startswith("Este reporte se elaboró con 10 de las 28 comparaciones válidas")
    assert "RTC 2" in r["advertencia_cobertura"]
    assert r["lectura_por_nivel"] is None                         # adenda 8.2
    assert r["titulo_sintesis"] == "Hipótesis para el análisis de la RTC 2"
    assert all("¿Qué evidencias adicionales" in r["preguntas_rtc"][0] for _ in [0])
    texto = " ".join(r["interpretaciones_por_aspecto"].values()) + r["sintesis_institucional"]
    for prohibido in ("la institución presenta", "la mayoría", "los docentes de la IE"):
        assert prohibido not in texto
    assert "lo que conviene verificar con" in r["interpretaciones_por_aspecto"]["C1"]
    assert not any(h["hallazgo"] == "Aspectos que requieren consolidación" for h in r["panorama_hallazgos"])


def test_reporte_suficiente_completo(llm_ok):
    estado = _estado(_comparaciones_sinteticas(20), n_docentes=24, sin_ref=1, no_validas=1)   # B2: umbral 20
    r = _correr(core.generar_reporte(IE, estado))
    assert r["estado_cobertura"] == "COB_SUFICIENTE" and r["advertencia_cobertura"] is None
    assert r["pct_cobertura"] == 83.3 and r["n_sin_entrega_v2"] == 2
    assert [c["criterio_id"] for c in r["distribucion_v2_por_aspecto"]] == ["C1", "C2", "C3", "C4", "C5"]
    assert r["avance_por_aspecto"][0]["avance_pp"] == 50.0
    assert r["titulo_sintesis"] == "Síntesis del análisis para la RTC 2"
    assert {n["nivel_educativo"] for n in r["lectura_por_nivel"]["niveles"]} == {"Primaria", "Secundaria"}
    assert set(r["secciones_no_disponibles"]) == set(core.SECCIONES_NO_DISPONIBLES)
    assert r["preguntas_rtc"] == list(core.PREGUNTAS_RTC)


def test_reporte_con_llm_caido_degrada_a_texto_plantillado(llm_roto):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(20), n_docentes=24)))
    assert r["estado_cobertura"] == "COB_SUFICIENTE"
    assert r["interpretaciones_por_aspecto"]["C3"].startswith("En 10 de las 20 comparaciones válidas")
    assert r["sintesis_institucional"]


def test_texto_visible_nunca_muestra_codigos_internos_ni_palabras_prohibidas(llm_ok):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(20), n_docentes=24)))
    visibles = [
        *r["interpretaciones_por_aspecto"].values(), r["sintesis_institucional"], *r["preguntas_rtc"],
        *(h["informacion"] for h in r["panorama_hallazgos"]),
        *(e["texto_permitido"] for e in r["enfasis_lectura"]),
        *(f["hallazgo"] + f["pregunta"] + f["fuente_sugerida"] for f in r["aspectos_contraste"]),
        *(c["nombre"] for c in r["distribucion_v2_por_aspecto"]),
    ]
    texto = " ".join(visibles)
    assert not re.search(r"\b(C[1-5]|B[1-5]|BC[1-4]|SV1_B[1-5])\b", texto)
    assert not re.search(r"\b(patrón|patrones|alerta|alertas|semaforización|nivel global)\b", texto, re.I)


def test_el_reporte_no_incluye_nunca_datos_individuales(llm_ok):
    r = _correr(core.generar_reporte(IE, _estado(_comparaciones_sinteticas(20), n_docentes=24)))
    serializado = json.dumps(r, ensure_ascii=False)
    assert "user_id" not in serializado and "participante_id" not in serializado
    assert r["identificacion_individual_incluida"] is False


# ---------------------------------------------------------------------------
# Caché
# ---------------------------------------------------------------------------

def test_reporte_vigente_solo_si_no_cambian_comparaciones_ni_N():
    estado = _estado(_comparaciones_sinteticas(10), n_docentes=40)
    guardado = {"estado": "generado", "n_docentes_evaluados": 10,
                "reporte_json": {"n_docentes_activos": 40, "n_no_aprobaron_c1": 0, "estado_cobertura": "COB_BAJO_UMBRAL", "version_calculo": core.VERSION_CALCULO}}
    assert core._reporte_vigente(guardado, estado)
    assert not core._reporte_vigente(None, estado)
    assert not core._reporte_vigente({**guardado, "n_docentes_evaluados": 9}, estado)                 # nueva evidencia
    assert not core._reporte_vigente({**guardado, "reporte_json": {"n_docentes_activos": 41, "n_no_aprobaron_c1": 0, "estado_cobertura": "COB_BAJO_UMBRAL", "version_calculo": core.VERSION_CALCULO}}, estado)  # N cambió (CP21)
    assert not core._reporte_vigente({**guardado, "estado": "sin_datos"}, estado)
    con_no_aprobados = _estado(_comparaciones_sinteticas(10), n_docentes=40, no_aprobaron_c1=5)
    assert not core._reporte_vigente(guardado, con_no_aprobados)                                      # cambió el conteo de no aprobados
    assert core._reporte_vigente({**guardado, "reporte_json": {"n_docentes_activos": 40, "n_no_aprobaron_c1": 5, "estado_cobertura": "COB_BAJO_UMBRAL", "version_calculo": core.VERSION_CALCULO}}, con_no_aprobados)


def test_reporte_guardado_con_otra_version_de_calculo_se_regenera():
    estado = _estado(_comparaciones_sinteticas(10), n_docentes=40, no_aprobaron_c1=0)
    json_ok = {"n_docentes_activos": 40, "n_no_aprobaron_c1": 0, "estado_cobertura": "COB_BAJO_UMBRAL",
               "version_calculo": core.VERSION_CALCULO}
    guardado = {"estado": "generado", "n_docentes_evaluados": 10, "reporte_json": json_ok}
    assert core._reporte_vigente(guardado, estado)
    assert not core._reporte_vigente({**guardado, "reporte_json": {**json_ok, "version_calculo": core.VERSION_CALCULO - 1}}, estado)
    assert not core._reporte_vigente({**guardado, "reporte_json": {**json_ok, "estado_cobertura": "COB_SUFICIENTE"}}, estado)


# ---------------------------------------------------------------------------
# Consulta previa a la solicitud (adenda RE-01)
# ---------------------------------------------------------------------------

def _stub_consulta(monkeypatch, estado, guardado=None):
    async def _ie(userid):
        return IE

    async def _estado_ie(ie):
        return estado

    async def _guardado(clave):
        return guardado

    monkeypatch.setattr(core, "resolver_ie", _ie)
    monkeypatch.setattr(core, "calcular_estado_ie", _estado_ie)
    monkeypatch.setattr(core, "obtener_reporte_guardado", _guardado)


def test_consulta_confidencial_muestra_solo_estado_de_cobertura(monkeypatch, llm_roto):
    _stub_consulta(monkeypatch, _estado(_comparaciones_sinteticas(4), n_docentes=4))
    r = _correr(core.consultar_reporte_existente(1))
    assert r["accion"] == "mostrar" and r["estado_cobertura"] == "COB_CONFIDENCIAL"
    assert r["reporte"]["estado_cobertura"] == "COB_CONFIDENCIAL" and "texto_confidencial" in r["reporte"]


def test_consulta_sin_reporte_pide_solicitud_con_el_estado_de_cobertura(monkeypatch):
    _stub_consulta(monkeypatch, _estado(_comparaciones_sinteticas(10), n_docentes=40, no_aprobaron_c1=6, sin_ref=2))
    r = _correr(core.consultar_reporte_existente(1))
    assert r["accion"] == "generar" and r["reporte"] is None            # no se genera sin solicitud
    assert (r["estado_cobertura"], r["banda_aplicada"], r["umbral_requerido"]) == ("COB_BAJO_UMBRAL", "B3", 28)
    assert r["n_docentes_activos"] == 40 and r["n_no_aprobaron_c1"] == 6 and r["pct_cobertura"] == 25.0
