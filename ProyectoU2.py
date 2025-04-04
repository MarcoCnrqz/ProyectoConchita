import tkinter as tk
from tkinter import filedialog, messagebox
import PyPDF2
import spacy
import nltk
import unicodedata
from nltk.corpus import cess_esp, stopwords
from nltk.metrics.distance import edit_distance
from collections import defaultdict
import json
import os

# Descargas de NLTK (solo la primera vez)
nltk.download('cess_esp')
nltk.download('stopwords')
nltk.download('wordnet')

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    messagebox.showerror(
        "Error de SpaCy",
        "No se encontró el modelo 'es_core_news_sm'.\n"
        "Instálalo con: python -m spacy download es_core_news_sm"
    )
    raise

# ------------------------------------------------------------------------
# CONFIGURACIONES GLOBALES
# ------------------------------------------------------------------------
ARCHIVO_CACHE = "cache_palabras.json"
MAX_TAMANO_CACHE = 10000

# Corpus de palabras correctas (cess_esp)
palabras_correctas = set(
    palabra.lower() for palabra in cess_esp.words() if palabra.isalpha()
)

# Stopwords
palabras_vacias = set(stopwords.words('spanish'))

# DICCIONARIO DE REEMPLAZOS FORMALES (se aplican en modo Formal y Muy formal)
reemplazos_formales = {
    "escuela": {
        "default": "institución",
        "contextos": {"educativa": "institución educativa"}
    },
    "colegio": {
        "default": "centro educativo",
        "contextos": {"privado": "institución privada"}
    },
    "chico": {
        "default": "joven",
        "contextos": {"hombre": "caballero"}
    },
    "chica": {
        "default": "joven",
        "contextos": {"mujer": "señorita"}
    },
    "bueno": {
        "default": "adecuado",
        "contextos": {"muy": "excelente"}
    },
    "malo": {
        "default": "inadecuado",
        "contextos": {"muy": "deficiente"}
    }
}

# DICCIONARIOS DE SINÓNIMOS POR MODO (basado en la LEMA)
sinonimos_muy_informal = {
    # verbos
    "hacer": "chambear",   # Ej: "hacer" -> "chambear"
    "decir": "contar",
    "ver": "wachar",
    "querer": "querer",    # lo dejamos igual
    "tener": "tener",      # lo dejamos igual para el ejemplo
    "ser": "ser",
    "estar": "estar",
    # adjetivos / adverbios
    "grande": "grandote",
    "pequeño": "chiquito",
    "bueno": "chido",
    "malo": "gacho",
    "rápido": "de volada",
    "lento": "despacito",      # sin cambio
    "mucho": "un montón",
    "poco": "poquito",
    "feliz": "contento",
    "triste": "agüitado",
    "chico": "morrillo",
    "verdad": "alch",
    "colegio": "el reclusorio de la mente",
    "escuela": "el reclusorio de la mente",
    "policia": "la chota",
    "trabajo": "el jale",
    "casa": "canton"
}

# Modo "Informal"
sinonimos_informal = {
    # verbos
    "hacer": "realizar",   # un poco más “normal”
    "decir": "decir",
    "ver": "ver",
    "querer": "querer",
    "tener": "tener",
    "ser": "ser",
    "estar": "estar",
    # adjetivos / adverbios
    "grande": "bastante grande",
    "pequeño": "pequeñito",
    "bueno": "bueno",
    "malo": "malo",
    "rápido": "rápido",
    "lento": "lento",
    "mucho": "mucho",
    "poco": "poco",
    "feliz": "contento",
    "triste": "triste",
    "chico": "chavo",
    "verdad": "neta",
    "colegio": "escuelita",
    "escuela": "escuelita",
    "casa": "mi cueva"
}

# Modo "Formal"
sinonimos_formal = {
    # verbos
    "hacer": "realizar",
    "decir": "expresar",
    "ver": "observar",
    "querer": "desear",
    "tener": "poseer",
    "ser": "ser",
    "estar": "encontrarse",
    # adjetivos / adverbios
    "grande": "considerable",
    "pequeño": "reducido",
    "bueno": "adecuado",
    "malo": "inadecuado",
    "rápido": "ágil",
    "lento": "pausado",
    "mucho": "abundantemente",
    "poco": "escasamente",
    "feliz": "contento",
    "triste": "afligido",
    "chico": "joven",
    "verdad": "verdad"
}

# Modo "Muy formal"
sinonimos_muy_formal = {
    # verbos
    "hacer": "llevar a cabo",
    "decir": "manifestar",
    "ver": "contemplar",
    "querer": "anhelar",
    "tener": "poseer",
    "ser": "constituir",
    "estar": "hallarse",
    # adjetivos / adverbios
    "grande": "magnánimo",
    "pequeño": "diminuto",
    "bueno": "excelso",
    "malo": "deficiente",
    "rápido": "rápidamente",
    "lento": "moroso",
    "mucho": "profundamente",
    "poco": "exiguamente",
    "feliz": "dichoso",
    "triste": "consternado",
    "chico": "joven",
    "verdad": "sinceramente",
    "maestros": "docentes",
    "trabajo": "laborar"
}

sinonimos_por_modo = {
    "Muy informal": sinonimos_muy_informal,
    "Informal": sinonimos_informal,
    "Formal": sinonimos_formal,
    "Muy formal": sinonimos_muy_formal
}

class CachePalabras:
    def __init__(self, archivo_cache=ARCHIVO_CACHE):
        self.archivo_cache = archivo_cache
        self.cache = defaultdict(dict)
        self.cargar_cache()
        
    def cargar_cache(self):
        if os.path.exists(self.archivo_cache):
            try:
                with open(self.archivo_cache, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.cache[k] = v
            except:
                self.cache = defaultdict(dict)
    
    def guardar_cache(self):
        with open(self.archivo_cache, 'w', encoding='utf-8') as f:
            json.dump(dict(self.cache), f, ensure_ascii=False, indent=2)
    
    def obtener(self, palabra, contexto):
        clave_contexto = self._clave_contexto(contexto)
        return self.cache.get(palabra.lower(), {}).get(clave_contexto)
    
    def establecer(self, palabra, contexto, reemplazo):
        if len(self.cache) > MAX_TAMANO_CACHE:
            self.cache.popitem()
        clave_contexto = self._clave_contexto(contexto)
        self.cache[palabra.lower()][clave_contexto] = reemplazo
        self.guardar_cache()
    
    def _clave_contexto(self, contexto):
        return "|".join(contexto)

cache_palabras = CachePalabras()

# ------------------------------------------------------------------------
# FUNCIONES DE PROCESO (LÍNEA POR LÍNEA)
# ------------------------------------------------------------------------

def limpiar_texto(texto):
    """Conserva saltos de línea, elimina tildes, pero no quita puntuación ni dígitos."""
    # Eliminamos \r y normalizamos acentos
    texto = texto.replace("\r", "")
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def corregir_ortografia_en_linea(line):
    """
    Corrige ortografía de una sola línea de texto.
    Ignora:
      - verbos
      - stopwords
      - tokens que NO sean alfabéticos (p. ej. puntuación, números, símbolos).
    Retorna una LISTA de dict con {'texto_original','texto_corregido'}.
    """
    doc = nlp(line)
    tokens_info = []

    for i, token in enumerate(doc):
        palabra_original = token.text

        # 1) Si NO es alfabético (puntuación, números, etc.), NO lo tocamos
        if not token.is_alpha:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })
            continue

        # 2) No corregimos verbos ni stopwords
        if token.pos_ == "VERB" or palabra_original.lower() in palabras_vacias:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })
            continue

        # 3) Si ya está en el corpus, se toma como correcta
        if palabra_original.lower() in palabras_correctas:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })
            continue

        # 4) Revisar caché (usamos 2 palabras antes y 2 después)
        inicio = max(0, i - 2)
        fin = min(len(doc), i + 3)
        contexto = [doc[j].text for j in range(inicio, fin)]

        cacheada = cache_palabras.obtener(palabra_original, contexto)
        if cacheada:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': cacheada
            })
            continue

        # 5) Calcular la palabra más cercana
        mejor_match = min(
            palabras_correctas,
            key=lambda w: edit_distance(w, palabra_original.lower())
        )
        distancia = edit_distance(mejor_match, palabra_original.lower())

        if distancia <= 2:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': mejor_match
            })
            cache_palabras.establecer(palabra_original, contexto, mejor_match)
        else:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })

    return tokens_info

def aplicar_reemplazos_formales_en_linea(tokens_info):
    """
    Aplica reemplazos formales con contexto.
    Retorna una lista de strings.
    """
    line_out = []
    length = len(tokens_info)

    for i, token_dict in enumerate(tokens_info):
        orig = token_dict['texto_original']
        corr = token_dict['texto_corregido']

        inicio = max(0, i - 2)
        fin = min(length, i + 3)
        palabras_alrededor = [t['texto_original'].lower() for t in tokens_info[inicio:fin]]

        if orig.lower() in reemplazos_formales:
            info = reemplazos_formales[orig.lower()]
            reemplazo_final = info["default"]
            for clave_contextual, reemplazo_especifico in info["contextos"].items():
                if clave_contextual.lower() in palabras_alrededor:
                    reemplazo_final = reemplazo_especifico
                    break
            line_out.append(reemplazo_final)
        else:
            line_out.append(corr)

    return line_out

def sinonimizar_por_modo_en_linea(line_text, modo):
    """
    Aplica sinónimos según el modo, basándose en la LEMA.
    Ignora tokens que no sean VERB, NOUN, ADJ, ADV.
    """
    sinonimos_dict = sinonimos_por_modo.get(modo, {})
    doc = nlp(line_text)
    resultado = []

    for token in doc:
        if token.pos_ in ["VERB", "NOUN", "ADJ", "ADV"]:
            lem = token.lemma_.lower()
            if lem in sinonimos_dict:
                resultado.append(sinonimos_dict[lem])
                continue
        resultado.append(token.text)

    return " ".join(resultado)

def procesar_linea(line, modo):
    """
    Procesa UNA línea completa respetando la estructura:
      - Corrección ortográfica
      - (opc) Reemplazos formales si modo es Formal o Muy formal
      - Sinonimización por modo
    """
    if not line.strip():
        return line  # Línea vacía: no procesar

    tokens_corregidos = corregir_ortografia_en_linea(line)

    if modo in ["Formal", "Muy formal"]:
        tokens_reemplazados = aplicar_reemplazos_formales_en_linea(tokens_corregidos)
    else:
        tokens_reemplazados = [t['texto_corregido'] for t in tokens_corregidos]

    texto_intermedio = " ".join(tokens_reemplazados)
    texto_final = sinonimizar_por_modo_en_linea(texto_intermedio, modo)
    return texto_final

def procesar_texto():
    """
    Lee el texto del cuadro de entrada, lo 'limpia' (sin quitar signos),
    luego lo divide en líneas y procesa cada línea por separado, 
    conservando el formato original.
    """
    modo = modo_var.get()
    texto_entrada = entrada_texto.get("1.0", tk.END)
    # Limpiamos acentos pero NO tocamos puntuaciones ni dígitos
    texto_entrada_limpio = limpiar_texto(texto_entrada)

    # Divide en líneas
    lineas = texto_entrada_limpio.splitlines(keepends=False)
    lineas_salida = []

    for line in lineas:
        nueva_linea = procesar_linea(line, modo)
        lineas_salida.append(nueva_linea)

    texto_salida_final = "\n".join(lineas_salida)
    salida_texto.delete("1.0", tk.END)
    salida_texto.insert(tk.END, texto_salida_final)

def cargar_pdf():
    """
    Carga un PDF y coloca su contenido en el cuadro de entrada,
    conservando saltos de línea entre páginas.
    """
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos PDF", "*.pdf")])
    if not ruta_archivo:
        return

    try:
        num_pag = int(pages_entry.get())
        if num_pag < 1:
            raise ValueError
    except ValueError:
        messagebox.showinfo(
            "Páginas inválidas",
            "Número de páginas no válido. Se usarán 20 por defecto."
        )
        num_pag = 20

    try:
        with open(ruta_archivo, "rb") as archivo:
            lector = PyPDF2.PdfReader(archivo)
            paginas_a_leer = min(num_pag, len(lector.pages))
            texto_cargado = []
            for i in range(paginas_a_leer):
                page_text = lector.pages[i].extract_text() or ""
                texto_cargado.append(page_text)

            # Doble salto para separar páginas
            texto_final = "\n\n".join(texto_cargado)

            entrada_texto.delete("1.0", tk.END)
            entrada_texto.insert(tk.END, texto_final)

    except Exception as e:
        messagebox.showerror("Error al leer PDF", str(e))

# ------------------------------------------------------------------------
# INTERFAZ GRÁFICA
# ------------------------------------------------------------------------
ventana = tk.Tk()
ventana.title("Transformador de Texto - Respetando Formato y Puntuación")

frame_entrada = tk.Frame(ventana)
frame_entrada.pack(pady=5)

tk.Label(frame_entrada, text="Texto original (con saltos de línea):").pack(anchor='w')
entrada_texto = tk.Text(frame_entrada, wrap=tk.WORD, width=80, height=12)
entrada_texto.pack()

frame_modo = tk.Frame(ventana)
frame_modo.pack(pady=5)

modo_var = tk.StringVar(value="Muy informal")
tk.Label(frame_modo, text="Modo de transformación:").pack(anchor='w')
modos = ["Muy informal", "Informal", "Formal", "Muy formal"]
for m in modos:
    rb = tk.Radiobutton(frame_modo, text=m, variable=modo_var, value=m)
    rb.pack(side=tk.LEFT, padx=5)

frame_botones = tk.Frame(ventana)
frame_botones.pack(pady=5)

tk.Label(frame_botones, text="Páginas a leer del PDF:").pack(side=tk.LEFT, padx=5)
pages_entry = tk.Entry(frame_botones, width=5)
pages_entry.insert(0, "2")
pages_entry.pack(side=tk.LEFT, padx=5)

btn_pdf = tk.Button(frame_botones, text="Cargar PDF", command=cargar_pdf)
btn_pdf.pack(side=tk.LEFT, padx=5)

btn_mejorar = tk.Button(frame_botones, text="Transformar Texto", command=procesar_texto)
btn_mejorar.pack(side=tk.LEFT, padx=5)

frame_salida = tk.Frame(ventana)
frame_salida.pack(pady=5)

tk.Label(frame_salida, text="Texto transformado (se conserva formato y puntuación):").pack(anchor='w')
salida_texto = tk.Text(frame_salida, wrap=tk.WORD, width=80, height=12)
salida_texto.pack()

ventana.mainloop()
