import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import numpy as np
import os

# --- 1. SETUP ---
model_name = "gpt2-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

text = "The investor deposited money at the bank, while the duck swam to the river bank."
inputs = tokenizer(text, return_tensors="pt")

# --- 2. FARBEN & FILTER DEFINIEREN ---
finance_words = ['investor', 'deposited', 'money']
nature_words = ['duck', 'swam', 'river']
target_homonym = 'bank'

# Liste der Wörter, die wir NICHT sehen wollen (Stop Words)
filter_words = ['the', 'at', 'to', 'while', ',', '.'] # Habe auch Punkt/Komma dazu genommen

default_color = 'silver'
color_finance = 'tab:red'
color_nature = 'tab:blue'

all_tokens_raw = [tokenizer.decode([t]).strip() for t in inputs['input_ids'][0]]

# Wir bauen Listen für INDIZES, die wir behalten wollen
keep_indices = []
point_colors = []
bank_counter = 0

for i, token in enumerate(all_tokens_raw):
    clean_t = token.lower().strip(".,;!? ")
    
    # FILTER-LOGIK: Wenn das Wort in der Filter-Liste ist, überspringen wir es
    if clean_t in filter_words:
        continue
        
    # Wenn wir hier sind, behalten wir das Wort -> Index speichern
    keep_indices.append(i)
    
    # Farbe zuweisen (wie gehabt)
    if clean_t in finance_words:
        point_colors.append(color_finance)
    elif clean_t in nature_words:
        point_colors.append(color_nature)
    elif clean_t == target_homonym:
        bank_counter += 1
        if bank_counter == 1:
            point_colors.append(color_finance)
        else:
            point_colors.append(color_nature)
    else:
        point_colors.append(default_color)

# Wir erstellen eine gefilterte Liste der Token-Texte für den Plot
plot_tokens = [all_tokens_raw[i] for i in keep_indices]
num_tokens = len(plot_tokens)

print(f"Tokens nach Filterung: {plot_tokens}")

# --- 3. BERECHNUNG ---
with torch.no_grad():
    outputs = model(**inputs)
hidden_states = outputs.hidden_states

all_vectors = []
for layer_data in hidden_states:
    # WICHTIG: Wir nehmen nur die Vektoren an den Indizes, die wir behalten wollen!
    # layer_data shape: [1, seq_len, hidden_size]
    vecs = layer_data[0, keep_indices, :].numpy()
    all_vectors.append(vecs)

all_vectors_np = np.vstack(all_vectors)

# PCA berechnen (jetzt ohne den Lärm von 'the')
pca = PCA(n_components=2)
all_projected = pca.fit_transform(all_vectors_np)

# --- 4. PLOTTING MIT BESCHRIFTUNGS-LOGIK ---
num_layers = model.config.n_layer + 1
output_dir = "frames_labeled"
os.makedirs(output_dir, exist_ok=True)
trail_length = 4 

print(f"Erstelle {num_layers} Bilder mit detaillierter Beschriftung...")

for layer_idx in range(num_layers):
    plt.figure(figsize=(10, 9)) # Etwas höher für mehr Platz
    
    # --- DATEN HOLEN ---
    start_row = layer_idx * num_tokens
    current_data = all_projected[start_row : start_row + num_tokens]
    
    # Zentrierung & SymLog (wie vorher)
    mean_x = np.mean(current_data[:, 0])
    mean_y = np.mean(current_data[:, 1])
    cur_x = current_data[:, 0] - mean_x
    cur_y = current_data[:, 1] - mean_y
    
    plt.xscale('symlog', linthresh=1.0)
    plt.yscale('symlog', linthresh=1.0)
    
    # --- SCHWEIF & PUNKTE (Code wie zuvor) ---
    for token_i in range(num_tokens):
        prev_x_list = []
        prev_y_list = []
        for back_step in range(trail_length):
            past_layer = layer_idx - (back_step + 1)
            if past_layer >= 0:
                p_start = past_layer * num_tokens
                past_data = all_projected[p_start : p_start + num_tokens]
                p_x = past_data[token_i, 0] - mean_x
                p_y = past_data[token_i, 1] - mean_y
                prev_x_list.append(p_x)
                prev_y_list.append(p_y)
        if prev_x_list:
            path_x = prev_x_list[::-1] + [cur_x[token_i]]
            path_y = prev_y_list[::-1] + [cur_y[token_i]]
            c = point_colors[token_i]
            alpha_trail = 0.5 if c != default_color else 0.05
            w_trail = 2 if c != default_color else 1
            plt.plot(path_x, path_y, color=c, alpha=alpha_trail, linewidth=w_trail, zorder=1)

    z_orders = [10 if c != default_color else 2 for c in point_colors]
    for i, token in enumerate(plot_tokens):
        c = point_colors[i]
        z = z_orders[i]
        is_highlight = c != default_color
        plt.scatter(cur_x[i], cur_y[i], c=c, s=200 if is_highlight else 100, 
                    edgecolors='white', alpha=0.9 if is_highlight else 0.4, zorder=z)
        fw = 'bold' if is_highlight else 'normal'
        plt.text(cur_x[i]+0.15, cur_y[i]+0.15, token, fontsize=11, fontweight=fw, 
                 color='black' if is_highlight else 'gray', zorder=z+1)

    # --- NEU: DYNAMISCHE TITEL & FORTSCHRITT ---
    
    # 1. Titel bestimmen
    if layer_idx == 0:
        main_title = "Input Embeddings (Start)"
        sub_title = "Statische Wortbedeutung vor Kontext-Analyse"
    elif layer_idx == num_layers - 1:
        main_title = f"Finaler Layer ({layer_idx})"
        sub_title = "Vollständig kontextualisierte Bedeutung"
    else:
        main_title = f"Transformer Block {layer_idx}"
        sub_title = "Verarbeitung von Kontext & Grammatik..."

    plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.96)
    plt.title(sub_title, fontsize=10, color='gray', style='italic')

    # 2. Fortschrittsanzeige (unten rechts)
    # Berechne Prozent (0% bis 100%)
    progress = int((layer_idx / (num_layers - 1)) * 100)
    plt.figtext(0.9, 0.02, f"Modell-Tiefe: {progress}%", ha="right", fontsize=10, 
                bbox={"facecolor":"white", "alpha":0.8, "pad":5})

    # --- DESIGN ---
    max_spread_x = np.max(np.abs(cur_x))
    max_spread_y = np.max(np.abs(cur_y))
    limit = max(max_spread_x, max_spread_y) * 1.5 
    if limit < 2: limit = 2
    
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.2)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.2)
    plt.grid(True, which='both', linestyle=':', alpha=0.4)
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    
    # Legende
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_finance, label='Finance'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_nature, label='Nature')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Footer lassen
    
    filename = f"{output_dir}/labeled_frame_{layer_idx:03d}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

print(f"Fertig! Bilder in '{output_dir}'.")