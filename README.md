# Proyecto Oráculo — Análisis y Diseño de Algoritmos (2026-B)
## Subproyecto ADAG1 · Grupo GP01

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JulianGomezOrtiz/ADA-2026-2-GP01/blob/main/Evaluacion_Inicial.ipynb)

**Curso:** Análisis y Diseño de Algoritmos (2026-B)  
**Grupo:** GP01 (ADAG1)  
**Integrantes:**
- Julián Gómez Ortiz

**Repositorio:** [https://github.com/JulianGomezOrtiz/ADA-2026-2-GP01](https://github.com/JulianGomezOrtiz/ADA-2026-2-GP01)  
**Rama:** `main`  
**Entrega:** Primera Entrega Consolidada (Semana 3)

---

## 1. Descripción del Proyecto

El **Proyecto Oráculo** aborda la optimización combinatoria de directivas de prompt y parámetros de decodificación para un modelo de lenguaje neuronal (`Qwen3-1.7B`), con el objetivo de maximizar el cumplimiento simultáneo de restricciones verificables de formato, estilo, longitud y gramática en un conjunto curado de problemas (evaluación estricta todo-o-nada).

### Espacio de Búsqueda
El espacio de búsqueda abarca $8^5 = 32.768$ combinaciones de directivas discretas para temperatura $0.0$, correspondientes a cinco ranuras del catálogo:
- `rol`: directiva de prefijo y actitud inicial.
- `estrategia`: instrucción de proceso deliberativo previo a la respuesta.
- `formato`: estructura macro (Markdown, párrafos, viñetas).
- `estilo`: caracteres, mayúsculas y puntuación.
- `cierre`: sufijo y recordatorios de restricciones.

---

## 2. Productos de la Primera Entrega

En concordancia con la guía oficial de la primera entrega, el repositorio contiene los cuatro productos consolidados:

| Producto | Archivo(s) | Descripción |
|---|---|---|
| **1. Informe Académico y Formalización** | `Documento.ipynb` | Formalización matemática completa del problema, variables, entradas, salidas, restricciones, función objetivo y marco conceptual. |
| **2. Demostración de NP-Dureza** | `Documento.ipynb` (Sección 4) | Formulación en versión de decisión (`CONFIG-OPT-DECISION`), pertenencia a NP, reducción polinomial formal desde 3-SAT (en ambos sentidos $\iff$) y análisis de complejidad de muestreo (PAC/Hoeffding) para la parte estocástica. |
| **3. Búsqueda y Repositorio Base** | `Busqueda_Inicial.ipynb` y `entrega.json` | Registro de 34 evaluaciones experimentales con ascenso por coordenadas, trazas de fallos, análisis de opciones descartadas y generación del JSON oficial. |
| **4. Evaluación Oficial y Manual Técnico** | `Evaluacion_Inicial.ipynb` y `MANUAL_TECNICO.md` | Notebook preparado para ejecución desatendida por el docente sobre datos privados, y manual técnico exhaustivo de instalación, arquitectura, flujos y pruebas. |

---

## 3. Resultado de la Búsqueda y Configuración Oficial

Durante la campaña de optimización mediante **Ascenso por Coordenadas (Coordinate Ascent)** sobre el lote de 100 instancias de búsqueda, se registraron 34 evaluaciones antes de alcanzar el límite presupuestal de GPU en Colab:

- **Precisión Inicial (Base):** 9.0% (Eval 1)
- **Mejor Precisión Observada:** **16.0%** (Alcanzada en la **Evaluación 27**)
- **Incremento Logrado:** +7 puntos porcentuales netos (+77.7% relativo).

La mejor configuración encontrada, registrada en `entrega.json`, es:
```json
{
 "grupo": "GP01",
 "config": {
  "rol": 5,
  "estrategia": 2,
  "formato": 0,
  "estilo": 6,
  "cierre": 0,
  "temperatura": 0.0
 },
 "semana": 3
}
```

> **Aviso de Optimalidad:** El 16.0% se reporta como el mejor resultado empíricamente observado bajo el presupuesto explorado, y no como una garantía matemática de optimalidad global.

---

## 4. Instrucciones de Ejecución Rápida

### 4.1. Evaluación Oficial para el Docente (`Evaluacion_Inicial.ipynb`)
1. Abrir `Evaluacion_Inicial.ipynb` en Google Colab con entorno GPU T4 habilitado.
2. Seleccionar **Entorno de ejecución ▸ Ejecutar todas**.
3. El notebook:
   - Carga directamente `entrega.json`.
   - Detecta si el docente subió un archivo de prueba privado (ej. `datos_privados.json`); de lo contrario, evalúa sobre el conjunto de validación curado.
   - Imprime la precisión obtenida y el diagnóstico de restricciones sin exigir cambios en el código.

### 4.2. Prueba de Humo Local (Smoke Test sin GPU)
Para comprobar la sintaxis, catálogo y armado de prompts en cualquier entorno Python:
```python
from oraculo import armar, CATALOGO

config = {"rol": 5, "estrategia": 2, "formato": 0, "estilo": 6, "cierre": 0, "temperatura": 0.0}
instancia = {"prompt": "Write a short summary.", "restriccion": "length_constraints:number_of_words"}
prompt_generado = armar(config, instancia)
print("Prompt armado exitosamente:\n", prompt_generado[:200])
```

---

## 5. Estructura de Archivos del Repositorio

```text
ADA-2026-2-GP01/
├── README.md                 # Este documento
├── MANUAL_TECNICO.md         # Manual técnico detallado (entorno, arquitectura, flujos y errores)
├── Documento.ipynb           # Informe académico consolidado
├── Busqueda_Inicial.ipynb    # Bitácora experimental completa (34 evaluaciones y trazas)
├── Evaluacion_Inicial.ipynb  # Notebook oficial de evaluación automática para la profesora
├── entrega.json              # Configuración ganadora serializada
├── oraculo.py                # Módulo de caja negra del oráculo
├── ayudas.py                 # Funciones auxiliares de carga y particionado
└── datos_visibles.json       # Conjunto de 450 instancias curadas
```

Para más detalles sobre la arquitectura del software, manejo de excepciones (`OutOfMemoryError`, incidencia de `langdetect`) y supuestos de diseño, consulte [MANUAL_TECNICO.md](MANUAL_TECNICO.md).
