import tkinter as tk
from tkinter import filedialog
import PyPDF2
import spacy
import nltk
import unicodedata #Sirve para eliminar caracteres especiales
from nltk.corpus import wordnet, cess_esp, stopwords # Importar librerías necesarias, sirve para la corrección ortográfica y sinónimos
from nltk.metrics.distance import edit_distance
from collections import defaultdict #sirve para crear un diccionario con valores por defecto
import json
import os

# Descargas necesarias 
nltk.download('cess_esp')
nltk.download('wordnet')
nltk.download('stopwords')

# Cargar modelo SpaCy 
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"]) 

# Configuración
ARCHIVO_CACHE = "cache_palabras.json"
MAX_TAMANO_CACHE = 10000
VENTANA_CONTEXTO = 3  # Número de palabras alrededor para contexto

# Corpus de palabras correctas
palabras_correctas = set(palabra.lower() for palabra in cess_esp.words() if palabra.isalpha())

# Palabras vacías y términos protegidos
palabras_vacias = set(stopwords.words('spanish'))
terminos_protegidos = {"al", "a", "de", "del", "mi", "su", "la", "el", "los", "las"}

# Reemplazos formales predefinidos con contexto
reemplazos_formales = {
    "escuela": {"default": "institución", "contextos": {"educativa": "institución educativa"}},
    "colegio": {"default": "centro educativo", "contextos": {"privado": "institución privada"}},
    "chico": {"default": "joven", "contextos": {"hombre": "caballero"}},
    "chica": {"default": "joven", "contextos": {"mujer": "señorita"}},
    "bueno": {"default": "adecuado", "contextos": {"muy": "excelente"}},
    "malo": {"default": "inadecuado", "contextos": {"muy": "deficiente"}}
}

class CachePalabras:
    """Cache para almacenar palabras ya procesadas y sus reemplazos óptimos."""
    def __init__(self, archivo_cache=ARCHIVO_CACHE):
        self.archivo_cache = archivo_cache
        self.cache = defaultdict(dict)
        self.cargar_cache()
        
    def cargar_cache(self):
        if os.path.exists(self.archivo_cache):
            try:
                with open(self.archivo_cache, 'r', encoding='utf-8') as f:
                    self.cache.update(json.load(f))
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
            self.cache.popitem()  # Eliminar entrada más antigua
            
        clave_contexto = self._clave_contexto(contexto)
        self.cache[palabra.lower()][clave_contexto] = reemplazo
        self.guardar_cache()
    
    def _clave_contexto(self, contexto):
        return "|".join(contexto)

# Inicializar cache
cache_palabras = CachePalabras()

def limpiar_texto(texto):
    """Limpia el texto de espacios innecesarios y normaliza caracteres especiales."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    texto = " ".join(texto.split())
    texto = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return texto

def corregir_ortografia(texto):
    """Corrige la ortografía basándose en un corpus de palabras correctas."""
    doc = nlp(texto)
    palabras_corregidas = []
    
    for i, token in enumerate(doc):
        if token.pos_ == "VERB":
            palabras_corregidas.append(token.text)
            continue
        
        palabra = token.text
        if palabra.lower() in palabras_correctas:
            palabras_corregidas.append(palabra)
        else:
            contexto = [doc[j].text for j in range(max(0, i-1), min(len(doc), i+2))]
            cacheada = cache_palabras.obtener(palabra, contexto)
            if cacheada:
                palabras_corregidas.append(cacheada)
                continue
                
            mejor_match = min(palabras_correctas, key=lambda w: edit_distance(w, palabra.lower()))
            distancia = edit_distance(mejor_match, palabra.lower())
            
            if distancia <= 2:
                palabras_corregidas.append(mejor_match)
                cache_palabras.establecer(palabra, contexto, mejor_match)
            else:
                palabras_corregidas.append(palabra)
                
    return " ".join(palabras_corregidas)

def procesar_texto():
    texto = entrada_texto.get("1.0", tk.END)
    if texto.strip():
        texto = limpiar_texto(texto)
        texto = corregir_ortografia(texto)
        salida_texto.delete("1.0", tk.END)
        salida_texto.insert(tk.END, texto)

def cargar_pdf():
    ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos PDF", "*.pdf")])
    if ruta_archivo:
        with open(ruta_archivo, "rb") as archivo:
            lector = PyPDF2.PdfReader(archivo)
            texto = " ".join(pagina.extract_text() or "" for pagina in lector.pages[:20])
            entrada_texto.delete("1.0", tk.END)
            entrada_texto.insert(tk.END, texto)

# Interfaz gráfica
ventana = tk.Tk()
ventana.title("Mejorador de Texto Formal")

frame = tk.Frame(ventana)
frame.pack(pady=10)

tk.Label(frame, text="Texto original:").pack()
entrada_texto = tk.Text(frame, wrap=tk.WORD, width=80, height=15)
entrada_texto.pack()

botones_frame = tk.Frame(ventana)
botones_frame.pack(pady=5)

tk.Button(botones_frame, text="Cargar PDF", command=cargar_pdf).pack(side=tk.LEFT, padx=5)
tk.Button(botones_frame, text="Mejorar Texto", command=procesar_texto).pack(side=tk.LEFT, padx=5)

tk.Label(ventana, text="Texto mejorado:").pack()
salida_texto = tk.Text(ventana, wrap=tk.WORD, width=80, height=15)
salida_texto.pack()

ventana.mainloop()
