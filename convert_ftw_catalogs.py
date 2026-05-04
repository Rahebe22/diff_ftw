import csv
from pathlib import Path

input_files = [
    "croplands3-2.1.1_finetune_all_samples.csv",
    "croplands3-2.1.1_test.csv",
]

renames = {
    "image": "window_b",
    "image_dir": "window_b",
    "label": "mask",
    "usage": "split",
}

for filename in input_files:
    path = Path(filename)
    out_path = path.with_name(path.stem + "_ftw.csv")

    with path.open("r", newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        fieldnames = []
        for field in reader.fieldnames or []:
            normalized_field = field.strip()
            if normalized_field in renames:
                fieldnames.append(renames[normalized_field])
            elif normalized_field == "":
                continue
            else:
                fieldnames.append(normalized_field)

        with out_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                new_row = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    normalized_key = key.strip()
                    if normalized_key == "":
                        continue
                    if normalized_key in renames:
                        new_row[renames[normalized_key]] = value
                    else:
                        new_row[normalized_key] = value
                writer.writerow(new_row)

    print("Saved", out_path)
