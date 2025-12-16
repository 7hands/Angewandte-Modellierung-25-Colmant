import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import numpy as np
import os

# --- 1. SETUP & FILTER ---
model_name = "gpt2-medium" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)

text = "The investor deposited money at the bank, while the duck swam to the river bank."
inputs = tokenizer(text, return_tensors="pt")

finance_words = ['investor', 'deposited', 'money']
nature_words = ['duck', 'swam', 'river']
target_homonym = 'bank'
filter_words = ['the', 'at', 'to', 'while', ',', '.']

default_color = 'silver'
color_finance = 'tab:red'
color_nature = 'tab:blue'

all_tokens_raw = [tokenizer.decode([t]).strip() for t in inputs['input_ids'][0]]
keep_indices = []
point_colors = []
bank_counter = 0

for i, token in enumerate(all_tokens_raw):
    clean_t = token.lower().strip(".,;!? ")
    if clean_t in filter_words: continue
    keep_indices.append(i)
    if clean_t in finance_words: point_colors.append(color_finance)
    elif clean_t in nature_words: point_colors.append(color_nature)
    elif clean_t == target_homonym:
        bank_counter += 1
        point_colors.append(color_finance if bank_counter == 1 else color_nature)
    else: point_colors.append(default_color)

plot_tokens = [all_tokens_raw[i] for i in keep_indices]
num_tokens = len(plot_tokens)

# --- 2. PCA BERECHNUNG ---
with torch.no_grad():
    outputs = model(**inputs)
hidden_states = outputs.hidden_states

all_vectors = []
for layer_data in hidden_states:
    vecs = layer_data[0, keep_indices, :].numpy()
    all_vectors.append(vecs)
all_vectors_np = np.vstack(all_vectors)
pca = PCA(n_components=2)
all_projected = pca.fit_transform(all_vectors_np)

# --- 3. NORMALISIERUNG VORBEREITEN ---
# Wir wollen [0, 1], aber zentriert um 0.5.
# Schritt A: Wir berechnen für JEDEN Layer den Mittelpunkt und ziehen ihn ab (lokale Zentrierung)
num_layers_total = model.config.n_layer + 1
centered_data_list = []

for layer_idx in range(num_layers_total):
    start = layer_idx * num_tokens
    end = start + num_tokens
    layer_data = all_projected[start:end]
    
    # Mittelwert dieses Layers
    mean_x = np.mean(layer_data[:, 0])
    mean_y = np.mean(layer_data[:, 1])
    
    # Zentrieren
    centered = layer_data - np.array([mean_x, mean_y])
    centered_data_list.append(centered)

all_centered = np.vstack(centered_data_list)

# Schritt B: Den maximalen Abstand finden (Globales Scaling)
# Wir suchen den Punkt, der am weitesten von seinem Layer-Zentrum entfernt ist (über alle Layer hinweg)
max_abs_val = np.max(np.abs(all_centered))

# Kleiner Puffer (10%), damit Punkte nicht am Rand kleben
scale_factor = max_abs_val * 1.1 

print(f"Globaler Scaling-Faktor: {scale_factor:.2f}")

# --- 4. PLOTTING LOOP ---
output_dir = "frames_normalized_01"
os.makedirs(output_dir, exist_ok=True)
interpolation_steps = 6 
trail_length = 4
global_frame_counter = 0

print(f"Erstelle Bilder in [0, 1] Skala...")

# Wir iterieren durch die Übergänge
for layer_idx in range(num_layers_total - 1):
    
    # Wir nehmen jetzt die VOR-ZENTRIERTEN Daten
    data_start = centered_data_list[layer_idx]
    data_end = centered_data_list[layer_idx + 1]

    for step in range(interpolation_steps):
        alphas = np.linspace(0, 1, interpolation_steps, endpoint=False)
        alpha = alphas[step]
        
        # Interpolation der zentrierten Daten
        current_data_centered = data_start * (1 - alpha) + data_end * alpha
        
        # --- NORMALISIERUNG AUF [0, 1] ---
        # Formel: (Wert / Max_Range + 1) / 2
        # -Max wird zu 0.0
        # 0 wird zu 0.5 (Mitte)
        # +Max wird zu 1.0
        cur_x = (current_data_centered[:, 0] / scale_factor + 1) / 2
        cur_y = (current_data_centered[:, 1] / scale_factor + 1) / 2
        
        fig = plt.figure(figsize=(10, 9))
        
        # SCHWEIF (Trails)
        for token_i in range(num_tokens):
            prev_x_list = []
            prev_y_list = []
            for back_step in range(trail_length):
                past_layer = layer_idx - back_step
                if past_layer >= 0:
                    # Historische Daten holen (bereits zentriert!)
                    past_vec = centered_data_list[past_layer][token_i]
                    
                    # Gleiche Normalisierung anwenden wie beim aktuellen Frame
                    px = (past_vec[0] / scale_factor + 1) / 2
                    py = (past_vec[1] / scale_factor + 1) / 2
                    prev_x_list.append(px)
                    prev_y_list.append(py)
            
            if prev_x_list:
                path_x = prev_x_list[::-1] + [cur_x[token_i]]
                path_y = prev_y_list[::-1] + [cur_y[token_i]]
                c = point_colors[token_i]
                alpha_trail = 0.5 if c != default_color else 0.1
                w_trail = 2 if c != default_color else 1
                plt.plot(path_x, path_y, color=c, alpha=alpha_trail, linewidth=w_trail, zorder=1)

        # PUNKTE
        z_orders = [10 if c != default_color else 2 for c in point_colors]
        for i, token in enumerate(plot_tokens):
            c = point_colors[i]
            z = z_orders[i]
            is_highlight = c != default_color
            plt.scatter(cur_x[i], cur_y[i], c=c, s=200 if is_highlight else 100, 
                        edgecolors='white', alpha=0.9 if is_highlight else 0.4, zorder=z)
            fw = 'bold' if is_highlight else 'normal'
            plt.text(cur_x[i]+0.015, cur_y[i]+0.015, token, fontsize=11, fontweight=fw, 
                     color='black' if is_highlight else 'gray', zorder=z+1)

        # TITEL & INFO
        main_title = f"Übergang: Layer {layer_idx} → {layer_idx+1}"
        if layer_idx == 0 and step == 0: main_title = "Start: Input Embeddings"
        trans_progress = int(alpha * 100)
        
        plt.suptitle(main_title, fontsize=16, fontweight='bold', y=0.96)
        plt.title(f"Linear Normalisiert [0, 1] ({trans_progress}%)", fontsize=10, color='gray')

        total_progress = int((global_frame_counter / ((num_layers_total-1)*interpolation_steps)) * 100)
        plt.figtext(0.9, 0.02, f"Gesamt: {total_progress}%", ha="right", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.8, "pad":5})

        # --- ACHSEN FEST AUF [0, 1] SETZEN ---
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Fadenkreuz genau in der Mitte (bei 0.5)
        plt.axhline(0.5, c='k', lw=0.5, alpha=0.2)
        plt.axvline(0.5, c='k', lw=0.5, alpha=0.2)
        
        # Grid
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xlabel("Normalisierte Dimension 1")
        plt.ylabel("Normalisierte Dimension 2")
        
        filename = f"{output_dir}/norm_frame_{global_frame_counter:04d}.png"
        plt.savefig(filename, dpi=120)
        plt.close(fig)
        global_frame_counter += 1
        print(f"Bild {global_frame_counter} erstellt...", end='\r')

# --- 5. DAS FINALE BILD (Layer 24) ---
final_idx = num_layers_total - 1
data_final = centered_data_list[final_idx]

# Normalisieren
cur_x = (data_final[:, 0] / scale_factor + 1) / 2
cur_y = (data_final[:, 1] / scale_factor + 1) / 2

fig = plt.figure(figsize=(10, 9))
# Schweif für final (gleiche Logik wie oben)
for token_i in range(num_tokens):
    prev_x_list = []
    prev_y_list = []
    for back_step in range(trail_length):
        past_layer = final_idx - back_step
        if past_layer >= 0:
            past_vec = centered_data_list[past_layer][token_i]
            prev_x_list.append((past_vec[0]/scale_factor + 1)/2)
            prev_y_list.append((past_vec[1]/scale_factor + 1)/2)
    if prev_x_list:
        path_x = prev_x_list[::-1] + [cur_x[token_i]]
        path_y = prev_y_list[::-1] + [cur_y[token_i]]
        c = point_colors[token_i]
        plt.plot(path_x, path_y, color=c, alpha=0.5 if c!=default_color else 0.1, lw=2, zorder=1)

for i, token in enumerate(plot_tokens):
    c = point_colors[i]
    is_hl = c != default_color
    plt.scatter(cur_x[i], cur_y[i], c=c, s=200 if is_hl else 100, edgecolors='white', alpha=0.9, zorder=10)
    plt.text(cur_x[i]+0.015, cur_y[i]+0.015, token, fontweight='bold' if is_hl else 'normal', fontsize=11, zorder=11)

plt.suptitle(f"Finales Ergebnis: Layer {final_idx}", fontsize=16, fontweight='bold', y=0.96)
plt.title("Vollständig kontextualisiert (Linear [0,1])", fontsize=10, color='gray')
plt.xlim(0, 1); plt.ylim(0, 1)
plt.axhline(0.5, c='k', lw=0.5, alpha=0.2); plt.axvline(0.5, c='k', lw=0.5, alpha=0.2)
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig(f"{output_dir}/norm_frame_{global_frame_counter:04d}.png", dpi=120)
plt.close(fig)

print("\nFertig! Alle Bilder sind im Bereich [0, 1] normalisiert.")