# process_players_csv.py

import csv
import os
from add_face import save_face_image
import traceback

CSV_FILE = "players.csv"
IMAGE_BASE_DIR = "images" # The main folder containing the subdirectories

def process_players_csv():
    """
    Reads the players.csv file and adds each player to the training dataset.
    Assumes a directory structure like: images/<player_name>/<image_file>.jpg
    """
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] The file '{CSV_FILE}' was not found. Please create it.")
        return

    try:
        with open(CSV_FILE, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header row
            print("[INFO] Starting to process players from players.csv...")

            for row in reader:
                # The format is assumed to be [id, filename, name]
                if len(row) >= 3:
                    name_from_csv = row[2].strip()
                    filename = row[1].strip()
                    
                    # Correctly construct the image path based on the new directory structure
                    # The name is converted to lowercase to match typical directory naming
                    image_path = os.path.join(IMAGE_BASE_DIR, name_from_csv.replace(" ", "_").lower(), filename)

                    if name_from_csv and os.path.exists(image_path):
                        print(f"  -> Processing: {name_from_csv} from {image_path}")
                        try:
                            save_face_image(name_from_csv, image_path)
                        except Exception as e:
                            print(f"     [ERROR] Failed to save {name_from_csv}: {e}")
                            traceback.print_exc()
                    else:
                        print(f"  -> [WARNING] Skipping invalid entry for '{name_from_csv}'. Image file '{image_path}' does not exist.")

        print("[INFO] Finished processing players from players.csv.")

    except Exception as e:
        print(f"[CRITICAL ERROR] An unexpected error occurred while processing the CSV: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    process_players_csv()