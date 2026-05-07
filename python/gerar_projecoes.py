import os
import numpy as np
import tifffile

# =======================================================
# 1. CONFIGURAÇÃO DAS PASTAS (O TEU DISCO RIGIDO)
# =======================================================
dir_imagens = '/Volumes/HDD 500GB/dados_microscopia'
dir_mascaras = '/Volumes/HDD 500GB/dados_mask'

out_dir = '/Volumes/HDD 500GB/dados_sum_proj'
os.makedirs(out_dir, exist_ok=True)

print(f"A ler imagens de: {dir_imagens}")
print(f"A ler máscaras de: {dir_mascaras}")
print(f"A guardar resultados em: {out_dir}")
print("-" * 50)

# =======================================================
# 2. PROCESSAMENTO (Com Escudo Anti-Lixo e Filtro DAPI)
# =======================================================
for i in range(1, 91):
    # Nomes exatos conforme os teus downloads
    img_path = os.path.join(dir_imagens, f'tile_{i}.tif')
    mask_path = os.path.join(dir_mascaras, f'{i}_masks_filtered.tif')

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        continue

    # Escudo Anti-Lixo (ignora ficheiros com menos de 1MB)
    if os.path.getsize(img_path) < 1000000:
        print(f"⚠️ A saltar Tile {i}: Ficheiro original corrompido.")
        continue

    print(f"A processar Tile {i}...")

    try:
        # LER OS TIFFs 3D Brutos
        imagem_bruta_3d = tifffile.imread(img_path)
        mask_3d = tifffile.imread(mask_path)

        # O SEGREDO MÁGICO: Isolar APENAS o canal 0 (O DAPI puro que vimos no Fiji)
        dapi_3d_puro = imagem_bruta_3d[:, 0, :, :]

        # Projeção por Soma apenas para o DNA
        dapi_sum = np.sum(dapi_3d_puro, axis=0)
        dapi_sum_norm = (dapi_sum - np.min(dapi_sum)) / (np.max(dapi_sum) - np.min(dapi_sum) + 1e-8)
        dapi_sum_16bit = (dapi_sum_norm * 65535).astype(np.uint16)

        # Projeção Máxima para a máscara
        mask_max = np.max(mask_3d, axis=0)

        # GUARDAR na nova pasta (prontos para o MATLAB!)
        tifffile.imwrite(os.path.join(out_dir, f'dapi_sum_tile_{i}.tif'), dapi_sum_16bit)
        tifffile.imwrite(os.path.join(out_dir, f'mask_max_tile_{i}.tif'), mask_max.astype(np.uint16))
    
    except Exception as e:
        print(f" ❌ Erro ao processar a tile {i}: {e}")

print("\n🚀 Dataset perfeito gerado com sucesso!")