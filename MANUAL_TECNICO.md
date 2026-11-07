# MANUAL TÉCNICO DE LA SOLUCIÓN BASE
## Proyecto Oráculo — Análisis y Diseño de Algoritmos (2026-B)
**Subproyecto:** ADAG1 — Proyecto Oráculo  
**Grupo:** GP01  
**Integrantes:** Julián Gómez Ortiz  
**Repositorio GitHub:** `https://github.com/JulianGomezOrtiz/ADA-2026-2-GP01`  
**Rama:** `main`  
**Versión de Entrega:** Primera Entrega Consolidada (Semana 3)

---

## 1. Entorno de Ejecución y Dependencias

### 1.1. Lenguaje y Plataforma
- **Lenguaje:** Python 3.10 o superior (recomendado Python 3.11).
- **Aceleración por Hardware:** Requiere GPU con al menos 8 GB de VRAM (entorno estándar: NVIDIA Tesla T4 de Google Colab o superior con soporte CUDA).
- **Sistemas Operativos Compatibles:** Linux (Ubuntu 20.04+, Debian 11+), Google Colab Runtime, Windows 10/11 (con soporte de CUDA/WSL2).

### 1.2. Dependencias Principales
El proyecto interactúa con modelos de Hugging Face y la suite de evaluación IFEvalG/open-instruct. Las librerías necesarias son:

| Paquete | Versión Mínima | Propósito |
|---|---|---|
| `torch` | $\ge 2.1.0$ | Motor de inferencia y cálculo de tensores |
| `transformers` | $\ge 4.40.0$ | Carga de arquitecturas de LLM y tokenizadores |
| `accelerate` | $\ge 0.28.0$ | Inferencia multi-dispositivo y mapas de memoria |
| `bitsandbytes` | $\ge 0.46.1$ | Cuantización a 4-bit para modelos mayores |
| `spacy` | $\ge 3.7.0$ | Análisis gramatical en verificadores deterministas |
| `nltk` | $\ge 3.8.1$ | Tokenización de oraciones y palabras |
| `langdetect` | $\ge 1.0.9$ | Detección de idioma en respuestas generadas |
| `immutabledict` | $\ge 4.1.0$ | Registro inmutable de instrucciones |
| `pandas` / `matplotlib` | Recientes | Tabulación y generación de curvas de progreso |

### 1.3. Instalación de Dependencias
```bash
# Instalación de librerías base
pip install -q -U "bitsandbytes>=0.46.1" transformers accelerate nltk spacy emoji langdetect immutabledict pandas matplotlib

# Modelo de lenguaje para spaCy
python -m spacy download en_core_web_sm -q

# Repositorio de verificadores open-instruct
git clone https://github.com/allenai/open-instruct /content/open-instruct
```

---

## 2. Arquitectura del Software

### 2.1. Estructura del Repositorio
```text
ADA-2026-2-GP01/
├── README.md                 # Visión general, instalación y guía de uso rápido
├── MANUAL_TECNICO.md         # Documento técnico detallado (este manual)
├── Documento.ipynb           # Informe académico (formalización, NP-dureza y optimización)
├── Busqueda_Inicial.ipynb    # Bitácora experimental (34 evaluaciones, método, trazas)
├── Evaluacion_Inicial.ipynb  # Notebook de evaluación oficial desatendida para el docente
├── entrega.json              # Configuración ganadora oficial serializada
├── oraculo.py                # Módulo central: motor de oráculo, caché y verificadores
├── ayudas.py                 # Utilidades: carga de datos, división, curvas y exportación
└── datos_visibles.json       # 450 instancias curadas del proyecto
```

### 2.2. Responsabilidad de Cada Módulo
- **`oraculo.py`:**
  - Actúa como la caja negra de evaluación.
  - Clase `Oraculo`: Orquesta la tokenización por lotes (`_generar`), control adaptativo de prefill (`_lotes` y `PRESUPUESTO_ATENCION`), decodificación sin gradientes (`_decodificar`), verificación lógica contra el registro de instrucciones (`_verificar`) y persistencia atómica en disco (`cache_oraculo.json`).
  - Diccionario `CATALOGO`: Define el espacio discreto de directivas ($8$ opciones en cada una de las 5 ranuras: `rol`, `estrategia`, `formato`, `estilo`, `cierre`).
  - Función `espacio()`: Genera el producto cartesiano de configuraciones válidas.
  - Función `armar(config, instancia)`: Concatena sintácticamente las directivas y la consulta del usuario.
- **`ayudas.py`:**
  - `cargar_modelo(alias)`: Carga el checkpoint neuronal (por defecto `qwen17b` $\to$ `Qwen/Qwen3-1.7B`), parcha el tipo de dato para GPU T4 (`float16`) y verifica la asignación en VRAM.
  - `cargar_datos(ruta)` y `dividir(datos)`: Procesa las 450 instancias de `datos_visibles.json`, separando 100 de búsqueda, 87 de validación y 263 descartadas.
  - `curva(historial)`: Genera la gráfica de progreso acumulado del oráculo.
  - `entrega(grupo, config, semana, ruta)`: Escribe y valida el archivo `entrega.json`.
- **`datos_visibles.json`:**
  - Dataset curado derivado de IF-RLVR Train. Cada instancia contiene `id`, `prompt`, `split`, `ids` de familias de verificación y `kwargs` específicos de evaluación.

---

## 3. Funcionamiento Interno del Programa

### 3.1. Flujo de Datos
```mermaid
flowchart LR
    A["entrega.json / Busqueda"] --> B["ayudas.armar()"]
    B --> C["Qwen3-1.7B (GPU)"]
    C --> D["Texto Generado"]
    D --> E["IFEval Verificadores"]
    E --> F["Resultado (precisión, trazas)"]
    F --> G["Caché JSON / Reporte"]
```

1. **Construcción del Prompt:** `armar()` concatena en orden: directivas de rol, estrategia, formato y estilo; luego el texto de la petición del usuario; y finalmente la directiva de cierre (con recordatorio de restricciones si aplica).
2. **Inferencia Neuronal:** La llamada genera tokens de forma autoregresiva con `max_new_tokens=384`. En temperatura $0.0$, el proceso es puramente determinista (greedy decoding).
3. **Verificación Lógica Todo-o-Nada:** La salida se pasa a los verificadores de `open_instruct.IFEvalG`. Cada restricción de la instancia debe evaluarse a `True`. Si alguna falla, la instancia recibe puntaje $0$ y la familia de la primera restricción incumplida se añade a `trazas` bajo la clave `violo`.
4. **Persistencia en Caché:** Para evitar re-computaciones redundantes, las respuestas y sus calificaciones se almacenan bajo la clave `hashlib.md5(modelo | config_id | instancia_id | semilla)`.

### 3.2. Algoritmo de Optimización Implementado
El algoritmo empleado es **Ascenso por Coordenadas (Coordinate Ascent)**:
- **Estado inicial:** $c_0 = (0, 0, 0, 0, 0)$ a $\tau = 0.0$.
- **Exploración univariada:** Para cada componente $r \in \{\text{rol}, \text{estrategia}, \text{formato}, \text{estilo}, \text{cierre}\}$, se exploran sus alternativas $v \in \{0, \dots, 7\}$.
- **Actualización:** Se adopta de forma voraz (greedy) cualquier candidato cuya precisión supere la mejor registrada.
- **Condición de parada:** Una ronda completa sin mejoras (óptimo local) o límite de recursos de inferencia.

---

## 4. Guía de Uso Paso a Paso

### 4.1. Evaluación Oficial para el Docente (`Evaluacion_Inicial.ipynb`)
1. Abra `Evaluacion_Inicial.ipynb` en Google Colab con entorno GPU T4.
2. Ejecute todas las celdas en secuencia.
3. El notebook:
   - Carga `entrega.json` automáticamente.
   - Detecta si existe un archivo de test privado (ej. `datos_privados.json`); si no, toma la partición de validación.
   - Evalúa y presenta el porcentaje de precisión obtenido y el resumen de fallos.

### 4.2. Reproducción de la Búsqueda (`Busqueda_Inicial.ipynb`)
1. Abra `Busqueda_Inicial.ipynb`.
2. Ejecute la celda 4 para visualizar la tabla con las 34 evaluaciones experimentales históricas.
3. Ejecute la celda 5 para trazar la curva de progreso acumulado.
4. Si dispone de GPU T4, puede continuar explorando nuevas configuraciones o invocar `oraculo.validar(config, n=30)`.

### 4.3. Formato del Archivo `entrega.json`
El archivo debe ubicarse en la raíz del repositorio y respetar estrictamente esta estructura:
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

---

## 5. Pruebas Básicas, Validación y Manejo de Errores

### 5.1. Prueba de Corrección Básica (Smoke Test)
Para validar que el entorno está correctamente configurado sin necesidad de gastar GPU, puede ejecutarse la siguiente prueba sintética en Python:
```python
import json
from oraculo import armar, CATALOGO

config = {"rol": 5, "estrategia": 2, "formato": 0, "estilo": 6, "cierre": 0, "temperatura": 0.0}
instancia_test = {
    "prompt": "Write a 50-word story about a robot.",
    "restriccion": "length_constraints:number_of_words\tlanguage:response_language"
}

prompt_armado = armar(config, instancia_test)
assert "You are a meticulous assistant" in prompt_armado
assert "Write a 50-word story about a robot" in prompt_armado
print("Prueba de armado de prompt: EXITOSA.")
```

### 5.2. Errores Conocidos y Medidas de Mitigación
1. **`OutOfMemoryError` en GPU:**
   - *Causa:* Lotes grandes con prompts de gran longitud pueden superar la VRAM de la T4.
   - *Mitigación:* El módulo `oraculo.py` captura esta excepción en `_lote()` y conmuta automáticamente a decodificación individual de a un prompt por vez (`batch_size = 1`).
2. **Incidencia `ERROR:absl:Unable to detect language for text ... due to No features in text`:**
   - *Causa:* Formatos conflictivos (como viñetas o numeración obligatoria en textos muy cortos) provocan que el modelo devuelva salidas de 0 a 2 caracteres o sólo puntuación, rompiendo la extracción léxica de `langdetect`.
   - *Mitigación:* Se descartan los formatos invasivos (formato 1 a 7), reteniendo `formato = 0` (salida limpia).
3. **Reinicio Requerido de Entorno en Colab:**
   - *Causa:* Conflicto de versiones entre librerías preinstaladas de Colab y los módulos de `transformers`/`bitsandbytes`.
   - *Mitigación:* Es imperativo reiniciar la sesión de Colab tras correr la celda 1 de instalación.

---

## 6. Supuestos y Componentes en Desarrollo
- **Fase Actual:** Primera entrega completada (formalización, reducción NP, búsqueda inicial documentada con 34 evaluaciones, evaluación base y manual técnico).
- **Alcance Pendiente (Segunda Fase):**
  - Culminación de la exploración de la ranura de cierre (`cierre = 3` con inyección de `{restricciones}`).
  - Ajuste fino y comparación con temperaturas no nulas ($\tau = 0.3$).
  - Búsqueda informada con poda basada en trazas (Beam Search o Backtracking con poda guiado por familias de fallo).

