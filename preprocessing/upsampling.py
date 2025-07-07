import pandas as pd
import os
import shutil
def upsample_images(df, image_dir, output_dir, min_samples=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for col in df.columns[1:]:
        samples = df[df[col] == 1]
        total = len(samples)
        if total == 0:
            continue
        factor = (min_samples // total) if total < min_samples else 1
        duplicated = pd.concat([samples] * factor, ignore_index=True)
        for i, row in duplicated.iterrows():
            src = os.path.join(image_dir, f"{row['ID']}.jpg")
            dst = os.path.join(output_dir, f"{row['ID']}_{i}.jpg")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
