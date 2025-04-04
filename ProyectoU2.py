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

# ======================================================
# 1) Cargar el modelo spaCy COMPLETO (es_core_news_sm)
#    para asegurar que el lematizador funcione.
# ======================================================
try:
    # OJO: no deshabilitamos "attribute_ruler" ni "lemmatizer"
    # para que la lematización en español funcione correctamente
    nlp = spacy.load("es_core_news_sm")
except OSError:
    messagebox.showerror(
        "Error de SpaCy",
        "No se encontró el modelo 'es_core_news_sm'.\n"
        "Instálalo con: python -m spacy download es_core_news_sm"
    )
    raise

# ------------------------------------------------------------------------
# CONFIGURACIÓN Y ESTRUCTURAS DE DATOS
# ------------------------------------------------------------------------

ARCHIVO_CACHE = "cache_palabras.json"
MAX_TAMANO_CACHE = 10000

# Palabras correctas (tomadas del corpus cess_esp)
palabras_correctas = set(
    palabra.lower() for palabra in cess_esp.words() if palabra.isalpha()
)

# Stopwords (palabras vacías)
palabras_vacias = set(stopwords.words('spanish'))

# Diccionario de reemplazos formales con contexto
# Se aplica DESPUÉS de la corrección ortográfica.
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
        "contextos": {"muy": "excelente", "sumamente": "magnífico"}
    },
    "malo": {
        "default": "inadecuado",
        "contextos": {"muy": "deficiente"}
    }
}

# Diccionario adicional de SINÓNIMOS “más formales” usando la LEMA
# Se aplica en la última fase del pipeline.
sinonimos_formales = {
    "computadora": "ordenador",
    "hacer": "realizar",
    "decir": "manifestar",
    "ver": "observar",
    "dar": "proporcionar",
    "tener": "poseer",
    "usar": "utilizar",
    "querer": "desear",
    "necesitar": "requerir",
    "ir": "acudir",
    "estar": "hallarse",  # Ejemplo, "estaba" -> "se hallaba"
    "ser": "constituir",
    "grande": "magnánimo",
    "pequeño": "reducido",
    "rápido": "veloz",
    "lento": "pausado",
    "feliz": "contento",
    "triste": "afligido",
    "fuerte": "vigoroso",
    "débil": "frágil",
    "mucho": "abundantemente",
    "poco": "escasamente"
}

class CachePalabras:
    """Cache para almacenar palabras procesadas y sus reemplazos óptimos."""
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
            self.cache.popitem()  # Elimina un ítem (en dicts 3.7+ es FIFO)
            
        clave_contexto = self._clave_contexto(contexto)
        self.cache[palabra.lower()][clave_contexto] = reemplazo
        self.guardar_cache()
    
    def _clave_contexto(self, contexto):
        # Convierte la lista de palabras de contexto en una cadena única
        return "|".join(contexto)

cache_palabras = CachePalabras()

# ------------------------------------------------------------------------
# FUNCIONES PRINCIPALES
# ------------------------------------------------------------------------

def limpiar_texto(texto):
    """Limpia saltos de línea, espacios múltiples y elimina tildes."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = " ".join(texto.split())
    # Eliminar acentos
    texto = ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if unicodedata.category(c) != 'Mn'
    )
    return texto

def aplicar_reemplazos_formales(tokens):
    """
    Aplica el diccionario 'reemplazos_formales' para cada palabra,
    revisando 2 palabras antes y 2 después para ver si hay contextos clave.
    tokens: lista de dict con 'texto_original' y 'texto_corregido'.
    Retorna una lista de strings (texto final, token por token).
    """
    tokens_modificados = []
    length = len(tokens)

    for i, token in enumerate(tokens):
        palabra_original = token['texto_original']
        palabra_corregida = token['texto_corregido']

        # Extraemos una ventana de 2 palabras antes y 2 después
        inicio = max(0, i - 2)
        fin = min(length, i + 3)
        palabras_alrededor = [t['texto_original'].lower() for t in tokens[inicio:fin]]

        # Si la palabra original está en el diccionario de reemplazos formales
        if palabra_original.lower() in reemplazos_formales:
            info = reemplazos_formales[palabra_original.lower()]
            reemplazo_final = info["default"]  # valor por defecto

            # Buscar si hay un contexto específico
            for clave_contextual, reemplazo_especifico in info["contextos"].items():
                if clave_contextual.lower() in palabras_alrededor:
                    reemplazo_final = reemplazo_especifico
                    break
            tokens_modificados.append(reemplazo_final)
        else:
            # Si no hay reemplazo formal, usamos la palabra corregida
            tokens_modificados.append(palabra_corregida)

    return tokens_modificados

def sinonimizar_formalmente(texto):
    """
    Recorre el texto final y reemplaza ciertos verbos, adjetivos, etc.
    por sinónimos "más formales" usando un diccionario básico.
    Se basa en la lematización en español de spaCy.
    """
    doc = nlp(texto)
    resultado = []

    for token in doc:
        # Solo aplicamos sinónimos a verbos, sustantivos, adjetivos, adverbios
        if token.pos_ in ["VERB", "NOUN", "ADJ", "ADV"]:
            # lemma_ = forma base (en minúscula, con .lower())
            lem = token.lemma_.lower()
            if lem in sinonimos_formales:
                # Reemplazamos por la palabra formal
                # (No ajustamos género/número, es un acercamiento simple)
                resultado.append(sinonimos_formales[lem])
                continue
        
        # Si no hay reemplazo de sinónimo, conservamos el token original
        resultado.append(token.text)

    # Volver a unir en un string
    return " ".join(resultado)

def corregir_ortografia(texto):
    """ 
    Corrige ortografía basándose en un corpus (cess_esp).
    Omite verbos y stopwords.
    Devuelve una lista de dict con:
       - 'texto_original'
       - 'texto_corregido'
    """
    doc = nlp(texto)
    tokens_info = []

    for i, token in enumerate(doc):
        palabra_original = token.text

        # No corregimos verbos ni stopwords
        if token.pos_ == "VERB" or palabra_original.lower() in palabras_vacias:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })
            continue

        # Si ya está en el corpus, se toma como correcta
        if palabra_original.lower() in palabras_correctas:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })
            continue

        # Revisar caché (usamos 2 palabras antes y 2 después)
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

        # Si no está en el caché, calculamos la más cercana
        mejor_match = min(
            palabras_correctas,
            key=lambda w: edit_distance(w, palabra_original.lower())
        )
        distancia = edit_distance(mejor_match, palabra_original.lower())

        # Si la distancia es pequeña, se asume corrección
        if distancia <= 2:
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': mejor_match
            })
            cache_palabras.establecer(palabra_original, contexto, mejor_match)
        else:
            # Dejar igual si no hay coincidencia razonable
            tokens_info.append({
                'texto_original': palabra_original,
                'texto_corregido': palabra_original
            })

    return tokens_info

def procesar_texto():
    """
    Lee el texto de entrada, lo limpia, corrige ortografía,
    aplica reemplazos formales con contexto y luego
    realiza una sinonimización final para obtener un texto 'muy formal'.
    Muestra el resultado en el cuadro de salida.
    """
    texto = entrada_texto.get("1.0", tk.END)
    if not texto.strip():
        messagebox.showwarning("Atención", "No hay texto para procesar.")
        return

    # 1) Limpiar texto
    texto_limpio = limpiar_texto(texto)

    # 2) Corregir ortografía (devuelve info token a token)
    tokens_corregidos = corregir_ortografia(texto_limpio)

    # 3) Aplicar los reemplazos formales con contexto
    tokens_con_reemplazos = aplicar_reemplazos_formales(tokens_corregidos)

    # 4) Volver a unir en un texto intermedio
    texto_intermedio = " ".join(tokens_con_reemplazos)

    # 5) Paso extra: sinonimización formal (usando lematizador)
    texto_final = sinonimizar_formalmente(texto_intermedio)

    # Mostrar en el texto de salida
    salida_texto.delete("1.0", tk.END)
    salida_texto.insert(tk.END, texto_final)

def cargar_pdf():
    """
    Carga un archivo PDF (tantas páginas como se indique)
    y coloca el texto en el cuadro de entrada.
    """
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos PDF", "*.pdf")])
    if not ruta_archivo:
        return

    # Intentar leer el número de páginas de la interfaz
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

    # Leer el PDF
    try:
        with open(ruta_archivo, "rb") as archivo:
            lector = PyPDF2.PdfReader(archivo)
            paginas_a_leer = min(num_pag, len(lector.pages))
            texto = []
            for i in range(paginas_a_leer):
                page_text = lector.pages[i].extract_text() or ""
                texto.append(page_text)
            texto_final = " ".join(texto)

            entrada_texto.delete("1.0", tk.END)
            entrada_texto.insert(tk.END, texto_final)
    except Exception as e:
        messagebox.showerror("Error al leer PDF", str(e))

# ------------------------------------------------------------------------
# CREACIÓN DE INTERFAZ GRÁFICA (tkinter)
# ------------------------------------------------------------------------

ventana = tk.Tk()
ventana.title("Mejorador de Texto - Muy Formal")

# Frame para la entrada
frame_entrada = tk.Frame(ventana)
frame_entrada.pack(pady=5)

tk.Label(frame_entrada, text="Texto original:").pack(anchor='w')
entrada_texto = tk.Text(frame_entrada, wrap=tk.WORD, width=80, height=15)
entrada_texto.pack()

# Frame para botones y entrada de páginas
frame_botones = tk.Frame(ventana)
frame_botones.pack(pady=5)

tk.Label(frame_botones, text="Páginas a leer del PDF:").pack(side=tk.LEFT, padx=5)
pages_entry = tk.Entry(frame_botones, width=5)
pages_entry.insert(0, "20")  # valor por defecto
pages_entry.pack(side=tk.LEFT, padx=5)

btn_cargar_pdf = tk.Button(frame_botones, text="Cargar PDF", command=cargar_pdf)
btn_cargar_pdf.pack(side=tk.LEFT, padx=5)

btn_mejorar = tk.Button(frame_botones, text="Mejorar Texto", command=procesar_texto)
btn_mejorar.pack(side=tk.LEFT, padx=5)

# Frame para la salida
frame_salida = tk.Frame(ventana)
frame_salida.pack(pady=5)

tk.Label(frame_salida, text="Texto mejorado:").pack(anchor='w')
salida_texto = tk.Text(frame_salida, wrap=tk.WORD, width=80, height=15)
salida_texto.pack()

ventana.mainloop()
