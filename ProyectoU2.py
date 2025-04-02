import tkinter as tk
from tkinter import filedialog
import PyPDF2
import spacy
import nltk
import unicodedata
from nltk.corpus import wordnet, cess_esp, stopwords
from nltk.metrics.distance import edit_distance
from collections import defaultdict
import json
import os

# Descargas necesarias 
nltk.download('cess_esp')
nltk.download('wordnet')
nltk.download('stopwords')

# Cargar modelo SpaCy 
nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"]) 

# Configuración
CACHE_FILE = "word_cache.json"
MAX_CACHE_SIZE = 10000
CONTEXT_WINDOW = 3  # Palabras alrededor para considerar contexto

# Corpus de palabras correctas (optimizado)
correct_words = set(word.lower() for word in cess_esp.words() if word.isalpha())

# Stopwords y términos protegidos
spanish_stopwords = set(stopwords.words('spanish'))
protected_terms = {"al", "a", "de", "del", "mi", "su", "la", "el", "los", "las"}

# Reemplazos formales predefinidos con contexto
formal_replacements = {
    "escuela": {"default": "institución", "contexts": {"educativa": "institución educativa"}},
    "colegio": {"default": "centro educativo", "contexts": {"privado": "institución privada"}},
    "chico": {"default": "joven", "contexts": {"hombre": "caballero"}},
    "chica": {"default": "joven", "contexts": {"mujer": "señorita"}},
    "bueno": {"default": "adecuado", "contexts": {"muy": "excelente"}},
    "malo": {"default": "inadecuado", "contexts": {"muy": "deficiente"}}
}

class WordCache:
    """Cache para almacenar palabras ya procesadas y sus reemplazos óptimos."""
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = defaultdict(dict)
        self.load_cache()
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache.update(json.load(f))
            except:
                self.cache = defaultdict(dict)
    
    def save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.cache), f, ensure_ascii=False, indent=2)
    
    def get(self, word, context):
        context_key = self._context_to_key(context)
        return self.cache.get(word.lower(), {}).get(context_key)
    
    def set(self, word, context, replacement):
        if len(self.cache) > MAX_CACHE_SIZE:
            self.cache.popitem()  # Eliminar entrada más antigua
            
        context_key = self._context_to_key(context)
        self.cache[word.lower()][context_key] = replacement
        self.save_cache()
    
    def _context_to_key(self, context):
        return "|".join(context)

# Inicializar cache
word_cache = WordCache()

def limpiar_texto(text):
    """Elimina espacios innecesarios y normaliza caracteres especiales."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())  # Eliminar espacios extras
    text = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
    return text

def corregir_ortografia(text):
    """
    Corrige la ortografía basándose en el corpus de palabras correctas, 
    exceptuando verbos para no alterar su conjugación.
    """
    doc = nlp(text)
    corrected_tokens = []
    
    for i, token in enumerate(doc):
        # Si es verbo, dejarlo intacto
        if token.pos_ == "VERB":
            corrected_tokens.append(token.text)
            continue
        
        word = token.text
        # Si la palabra ya es correcta, se deja igual
        if word.lower() in correct_words:
            corrected_tokens.append(word)
        else:
            # Buscar en cache primero
            context = [doc[j].text for j in range(max(0, i-1), min(len(doc), i+2))]
            cached = word_cache.get(word, context)
            if cached:
                corrected_tokens.append(cached)
                continue
                
            # Encontrar el mejor match
            closest_match = min(correct_words, key=lambda w: edit_distance(w, word.lower()))
            dist = edit_distance(closest_match, word.lower())
            
            # Solo corregir si la distancia es pequeña y no parece un nombre propio
            if dist <= 2 and not (word[0].isupper() and i > 0):
                corrected_tokens.append(closest_match)
                word_cache.set(word, context, closest_match)
            else:
                corrected_tokens.append(word)
                
    return " ".join(corrected_tokens)

def get_context_words(doc, index, window_size=CONTEXT_WINDOW):
    """Obtiene palabras de contexto alrededor de un índice dado."""
    start = max(0, index - window_size)
    end = min(len(doc), index + window_size + 1)
    return [doc[i].text.lower() for i in range(start, end) if i != index and not doc[i].is_punct]

def get_formal_synonym(word, token, context_words):
    """Sustituye palabras por sinónimos más formales considerando el contexto."""
    # Verificar cache primero
    cached = word_cache.get(word, context_words)
    if cached:
        return cached
    
    # No modificar ciertas palabras
    if (word.lower() in protected_terms or 
        token.is_punct or 
        token.is_space or 
        (token.pos_ == "VERB" and "Tense" in token.morph)):
        return word
    
    # Verificar reemplazos predefinidos con contexto
    if word.lower() in formal_replacements:
        replacement_data = formal_replacements[word.lower()]
        for ctx_word in context_words:
            if ctx_word in replacement_data["contexts"]:
                replacement = replacement_data["contexts"][ctx_word]
                word_cache.set(word, context_words, replacement)
                return replacement
        replacement = replacement_data["default"]
        word_cache.set(word, context_words, replacement)
        return replacement
    
    # Buscar sinónimos formales para sustantivos y adjetivos
    if token.pos_ in ["NOUN", "ADJ"]:
        synonyms = set()
        for syn in wordnet.synsets(word, lang='spa'):
            for lemma in syn.lemmas('spa'):
                lemma_name = lemma.name().replace('_', ' ')
                if (lemma_name != word and 
                    lemma_name in correct_words and 
                    len(lemma_name.split()) == 1):
                    synonyms.add(lemma_name)
        
        # Seleccionar el sinónimo más largo (generalmente más formal)
        if synonyms:
            replacement = max(synonyms, key=lambda x: (len(x), x))
            word_cache.set(word, context_words, replacement)
            return replacement
    
    return word

def mejorar_texto(text):
    """Mejora el texto cambiando palabras para que suenen más formales y ajustando la concordancia de género."""
    doc = nlp(text)
    improved_words = []
    
    for i, token in enumerate(doc):
        context_words = get_context_words(doc, i)
        improved_word = get_formal_synonym(token.text, token, context_words)
        
        # Si el token es un sustantivo y se reemplazó por un sinónimo, verificar el artículo anterior
        if token.pos_ == "NOUN" and improved_word.lower() != token.text.lower():
            if i > 0:
                prev_token = doc[i-1]
                # Comprobar si el token anterior es un determinante que requiere ajuste
                if prev_token.pos_ == "DET" and prev_token.text.lower() in {"el", "un", "este", "ese", "aquel"}:
                    improved_noun_doc = nlp(improved_word)
                    if improved_noun_doc and improved_noun_doc[0].morph.get("Gender"):
                        gender = improved_noun_doc[0].morph.get("Gender")
                        if "Fem" in gender:
                            mapping = {"el": "la", "un": "una", "este": "esta", "ese": "esa", "aquel": "aquella"}
                            if improved_words:
                                prev_word = improved_words[-1]
                                lower_prev = prev_word.lower()
                                if lower_prev in mapping:
                                    new_article = mapping[lower_prev]
                                    if prev_word[0].isupper():
                                        new_article = new_article.capitalize()
                                    improved_words[-1] = new_article
        improved_words.append(improved_word)
    
    smoothed_text = suavizar_texto(" ".join(improved_words))
    return smoothed_text

def suavizar_texto(text):
    """Aplica técnicas de suavizado para hacer el texto más natural."""
    doc = nlp(text)
    smoothed_words = []
    
    for i, token in enumerate(doc):
        current_word = token.text
        if i > 0 and current_word.lower() == doc[i-1].text.lower():
            continue
        smoothed_words.append(current_word)
    
    smoothed_text = " ".join(smoothed_words)
    smoothed_text = smoothed_text.replace(" a el ", " al ").replace(" de el ", " del ")
    return smoothed_text

def procesar_texto():
    text = texto_entrada.get("1.0", tk.END)
    if text.strip():
        text = limpiar_texto(text)
        text = corregir_ortografia(text)
        text = mejorar_texto(text)
        texto_salida.delete("1.0", tk.END)
        texto_salida.insert(tk.END, text)

def cargar_pdf():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(page.extract_text() or "" for page in reader.pages[:20])
            texto_entrada.delete("1.0", tk.END)
            texto_entrada.insert(tk.END, text)

# Interfaz gráfica
root = tk.Tk()
root.title("Mejorador de Texto Formal Avanzado")

frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Texto original:").pack()
texto_entrada = tk.Text(frame, wrap=tk.WORD, width=80, height=15)
texto_entrada.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

tk.Button(btn_frame, text="Cargar PDF", command=cargar_pdf).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Mejorar Texto", command=procesar_texto).pack(side=tk.LEFT, padx=5)

tk.Label(root, text="Texto mejorado:").pack()
texto_salida = tk.Text(root, wrap=tk.WORD, width=80, height=15)
texto_salida.pack()

root.mainloop()
