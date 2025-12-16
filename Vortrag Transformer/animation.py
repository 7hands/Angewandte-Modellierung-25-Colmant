import imageio.v3 as iio
import glob
import os

# --- EINSTELLUNGEN ---
# Der Name der Bilder, die wir suchen (das * ist ein Platzhalter)
image_pattern = "norm_frame_*.png"
output_video_name = "transformer_evolution_trail.gif" # oder .gif
fps = 1 # Frames per Second (Geschwindigkeit: Niedriger = langsamer)

# --- 1. BILDER FINDEN ---
# glob findet alle Dateien, die auf das Muster passen
files = glob.glob(image_pattern)

# WICHTIG: Sortieren, damit die Reihenfolge stimmt (00, 01, 02...)
# Da wir führende Nullen im Dateinamen haben, funktioniert die Standardsortierung.
files.sort()

if not files:
    print("FEHLER: Keine Bilder gefunden!")
    print(f"Stelle sicher, dass du das vorherige Skript ausgeführt hast und Bilder mit dem Muster '{image_pattern}' existieren.")
    exit()

print(f"{len(files)} Bilder gefunden. Starte Video-Erstellung...")

# --- 2. VIDEO SCHREIBEN ---
# Wir erstellen eine Liste der Bilder (als Arrays), die imageio verarbeiten kann
images_for_video = []
for filename in files:
    # Bild einlesen
    img_data = iio.imread(filename)
    images_for_video.append(img_data)

# Video schreiben
# Wenn der Dateiname auf .mp4 endet, wird ein Video erstellt.
# Wenn er auf .gif endet, wird ein GIF erstellt.
try:
    iio.imwrite(output_video_name, images_for_video, fps=fps)
    print(f"Fertig! Video gespeichert als: {output_video_name}")
except Exception as e:
    print(f"Fehler beim Speichern des Videos: {e}")
    print("Tipp: Falls MP4 nicht klappt, versuche den output_video_name auf '.gif' zu ändern.")

# --- OPTIONAL: AUFRÄUMEN ---
# Möchtest du die einzelnen PNG-Dateien danach löschen?
# Uncomment die nächsten Zeilen, wenn ja.
# print("Räume temporäre PNG-Dateien auf...")
# for f in files:
#     os.remove(f)
# print("Aufgeräumt.")