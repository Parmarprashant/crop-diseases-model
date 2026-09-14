import os
import sys
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance

random.seed(42)
np.random.seed(42)

os.makedirs('validation/stress_samples', exist_ok=True)
df = pd.read_csv('Data/outputs/outputs/test_split.csv')

def resolve_path(row):
    p = os.path.join('Data', str(row['image_path'])) if not os.path.isabs(str(row['image_path'])) else str(row['image_path'])
    if os.path.exists(p):
        return p
    for ext in ['.jpg', '.jpeg', '.png']:
        fb = os.path.join('Data', 'master_images', 'master_images', 'images', str(row['image_id']) + ext)
        if os.path.exists(fb):
            return fb
    return None

# Collect 220 agricultural samples
agri_samples = []

# 1. Target strata
strata_diseased = {
    'Apple': 12, 'Peach': 12, 'Soybean': 12, 'Blackgram': 12,
    'Grape': 12, 'Tomato': 16, 'Wheat': 14, 'Corn': 12,
    'Potato': 14, 'Chilli': 12, 'Bell Pepper': 12,
    'Banana': 12, 'Groundnut': 12, 'Citrus': 10, 'Paddy': 12,
    'Sugarcane': 4
}

for crop, n in strata_diseased.items():
    sub = df[(df['crop'].str.lower() == crop.lower()) & (~df['canonical_class'].str.contains('healthy', case=False, na=False))].copy()
    # Prioritize real-world field sources: PlantDoc, plantseg, BPLD, Multicrop
    field_pref = sub.sort_values(by='source_dataset', key=lambda col: col.map(lambda s: 0 if any(k in str(s) for k in ['PlantDoc', 'plantseg', 'BPLD']) else 1))
    sampled = field_pref.head(n)
    for _, row in sampled.iterrows():
        p = resolve_path(row)
        if p:
            source_name = str(row.get('source_dataset', 'test_split'))
            agri_samples.append({
                'image_id': str(row['image_id']),
                'source': source_name,
                'crop': crop.lower(),
                'disease': str(row['canonical_class']),
                'healthy_or_diseased': 'diseased',
                'verification_status': 'verified_ground_truth',
                'image_type': 'foliar_disease',
                'field_or_dataset': 'test_split',
                'cohort_group': 'agricultural',
                'file_path': p,
                'notes': f"Stratum: {crop} ({source_name})"
            })

# Strawberry: 3 scorch + 5 anthracnose + 4 healthy = 12 total
sub_sb = df[df['crop'].str.lower() == 'strawberry'].copy()
for _, row in sub_sb.iterrows():
    p = resolve_path(row)
    if p and len([s for s in agri_samples if s['crop'] == 'strawberry']) < 12:
        is_h = 'healthy' in str(row['canonical_class']).lower()
        source_name = str(row.get('source_dataset', 'test_split'))
        agri_samples.append({
            'image_id': str(row['image_id']),
            'source': source_name,
            'crop': 'strawberry',
            'disease': str(row['canonical_class']),
            'healthy_or_diseased': 'healthy' if is_h else 'diseased',
            'verification_status': 'verified_ground_truth',
            'image_type': 'healthy_control' if is_h else 'foliar_disease',
            'field_or_dataset': 'test_split',
            'cohort_group': 'agricultural',
            'file_path': p,
            'notes': f"Stratum: Strawberry ({row['canonical_class']})"
        })

# Dedicated Healthy Controls: 18 samples
h_strata = {'Tomato': 3, 'Potato': 3, 'Bell Pepper': 3, 'Paddy': 3, 'Banana': 2, 'Peach': 2, 'Groundnut': 2}
for crop, n in h_strata.items():
    sub_h = df[(df['crop'].str.lower() == crop.lower()) & (df['canonical_class'].str.contains('healthy', case=False, na=False))].copy()
    sampled_h = sub_h.head(n)
    for _, row in sampled_h.iterrows():
        p = resolve_path(row)
        if p:
            source_name = str(row.get('source_dataset', 'test_split'))
            agri_samples.append({
                'image_id': str(row['image_id']),
                'source': source_name,
                'crop': crop.lower(),
                'disease': str(row['canonical_class']),
                'healthy_or_diseased': 'healthy',
                'verification_status': 'verified_ground_truth',
                'image_type': 'healthy_control',
                'field_or_dataset': 'test_split',
                'cohort_group': 'agricultural',
                'file_path': p,
                'notes': f"Healthy Control: {crop}"
            })

print(f"Total agricultural samples collected: {len(agri_samples)}")

# Generate 30 Defensive Samples
defensive_samples = []

# Base leaf for degradations
base_leaf_path = agri_samples[0]['file_path']
base_im = Image.open(base_leaf_path).convert('RGB')

# 1. Severe Blur (6 samples)
for i in range(1, 7):
    radius = 16 + i * 4
    blurred = base_im.filter(ImageFilter.GaussianBlur(radius))
    out_p = os.path.join('validation/stress_samples', f'defensive_blur_{i}.jpg')
    blurred.save(out_p, quality=85)
    defensive_samples.append({
        'image_id': f'defensive_blur_{i}',
        'source': 'synthetic_stress',
        'crop': 'unknown',
        'disease': 'Expected Quality Gate Rejection (Blur)',
        'healthy_or_diseased': 'degraded',
        'verification_status': 'verified_stress',
        'image_type': 'severe_blur',
        'field_or_dataset': 'synthetic_stress',
        'cohort_group': 'defensive',
        'file_path': out_p,
        'notes': f"Severe Gaussian blur sigma={radius}"
    })

# 2. Severe Darkness (6 samples)
for i in range(1, 7):
    factor = 0.02 * i
    dark = ImageEnhance.Brightness(base_im).enhance(factor)
    out_p = os.path.join('validation/stress_samples', f'defensive_darkness_{i}.jpg')
    dark.save(out_p, quality=85)
    defensive_samples.append({
        'image_id': f'defensive_darkness_{i}',
        'source': 'synthetic_stress',
        'crop': 'unknown',
        'disease': 'Expected Quality Gate Rejection (Darkness)',
        'healthy_or_diseased': 'degraded',
        'verification_status': 'verified_stress',
        'image_type': 'severe_darkness',
        'field_or_dataset': 'synthetic_stress',
        'cohort_group': 'defensive',
        'file_path': out_p,
        'notes': f"Severe underexposure factor={factor:.2f}"
    })

# 3. Random Noise (6 samples)
for i in range(1, 7):
    noise_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    noise_im = Image.fromarray(noise_arr)
    out_p = os.path.join('validation/stress_samples', f'defensive_noise_{i}.jpg')
    noise_im.save(out_p, quality=85)
    defensive_samples.append({
        'image_id': f'defensive_noise_{i}',
        'source': 'synthetic_stress',
        'crop': 'unknown',
        'disease': 'Expected Energy OOD Rejection (Noise)',
        'healthy_or_diseased': 'ood',
        'verification_status': 'verified_stress',
        'image_type': 'random_noise',
        'field_or_dataset': 'synthetic_stress',
        'cohort_group': 'defensive',
        'file_path': out_p,
        'notes': f"Uniform random RGB noise pattern seed={i}"
    })

# 4. Non-Plant Synthetic Texture OOD (6 samples)
for i in range(1, 7):
    x = np.linspace(0, 10 * i, 256)
    y = np.linspace(0, 10 * i, 256)
    X, Y = np.meshgrid(x, y)
    tex = (np.sin(X) * np.cos(Y) * 127 + 128).astype(np.uint8)
    tex_rgb = np.stack([tex, np.roll(tex, 30), np.roll(tex, 60)], axis=-1)
    tex_im = Image.fromarray(tex_rgb)
    out_p = os.path.join('validation/stress_samples', f'defensive_texture_{i}.jpg')
    tex_im.save(out_p, quality=85)
    defensive_samples.append({
        'image_id': f'defensive_texture_{i}',
        'source': 'synthetic_stress',
        'crop': 'unknown',
        'disease': 'Expected Energy OOD Rejection (Texture)',
        'healthy_or_diseased': 'ood',
        'verification_status': 'verified_stress',
        'image_type': 'synthetic_texture_ood',
        'field_or_dataset': 'synthetic_stress',
        'cohort_group': 'defensive',
        'file_path': out_p,
        'notes': f"Sinusoidal synthetic texture frequency={10*i}"
    })

# 5. Non-Leaf / Non-Plant Scenes (6 samples)
for i in range(1, 7):
    base_color = np.array([110 + i * 8, 80 + i * 5, 50 + i * 3], dtype=np.uint8)
    soil_arr = np.clip(base_color + np.random.normal(0, 15, (256, 256, 3)), 0, 255).astype(np.uint8)
    soil_im = Image.fromarray(soil_arr)
    out_p = os.path.join('validation/stress_samples', f'defensive_soil_{i}.jpg')
    soil_im.save(out_p, quality=85)
    defensive_samples.append({
        'image_id': f'defensive_soil_{i}',
        'source': 'synthetic_stress',
        'crop': 'unknown',
        'disease': 'Expected Quality/OOD Rejection (Non-Leaf Soil)',
        'healthy_or_diseased': 'non_plant',
        'verification_status': 'verified_stress',
        'image_type': 'non_leaf_scene',
        'field_or_dataset': 'synthetic_stress',
        'cohort_group': 'defensive',
        'file_path': out_p,
        'notes': f"Non-leaf soil background simulation index={i}"
    })

print(f"Total defensive samples generated: {len(defensive_samples)}")

all_manifest = agri_samples + defensive_samples
print(f"GRAND TOTAL COHORT: {len(all_manifest)}")

manifest_df = pd.DataFrame(all_manifest)
manifest_df.to_csv('validation/external_cohort_manifest.csv', index=False)
print("Cohort manifest successfully written to validation/external_cohort_manifest.csv")
