import zarr
import numpy as np
import tifffile

# 1. CAMINHO PARA A PASTA PRINCIPAL (ajusta se a pasta estiver noutro sítio)
zarr_path = '/Users/ricardo/Downloads/tile_69.zarr'

# 2. ABRIR A RAIZ DO ZARR
print("A ler a base de dados Zarr...")
root = zarr.open(zarr_path, mode='r')

# 3. EXTRAIR AS IMAGENS (Nível 0 = Máxima Qualidade)
# A estrutura neste OME-Zarr é: (Canal, Z, Y, X)
# [0, :, :, :] significa: Canal 0, e todas as fatias Z, Y, X.
# Se o DAPI não for o Canal 0, experimenta [1, :, :, :] ou [2, :, :, :]
print("A extrair DAPI 3D...")
dapi_3d = root['0'][0, :, :, :] 

print("A extrair Máscara 3D...")
# A máscara neste caso é apenas (Z, Y, X)
mask_3d = root['labels']['segmentation_mask']['0'][:, :, :]

print(f"Dimensões 3D extraídas: DAPI={dapi_3d.shape}, Mask={mask_3d.shape}")

# =======================================================
# 4. FAZER AS PROJEÇÕES 2D
# =======================================================

print("A aplicar Projeções...")

# IMAGEM: Sum Projection (Esmaga o Z-stack somando as intensidades)
dapi_sum = np.sum(dapi_3d, axis=0)

# Normalizar a soma para 16-bit (0 a 65535) para o MATLAB conseguir ler
dapi_sum_norm = (dapi_sum - np.min(dapi_sum)) / (np.max(dapi_sum) - np.min(dapi_sum))
dapi_sum_16bit = (dapi_sum_norm * 65535).astype(np.uint16)

# MÁSCARA: Maximum Intensity Projection (Mantém os IDs intactos)
mask_max = np.max(mask_3d, axis=0)

# =======================================================
# 5. GUARDAR EM FICHEIROS TIFF
# =======================================================
tifffile.imwrite('dapi_sum_projection.tif', dapi_sum_16bit)
tifffile.imwrite('mask_max_projection.tif', mask_max.astype(np.uint16))

print("Sucesso! Os ficheiros 'dapi_sum_projection.tif' e 'mask_max_projection.tif' foram gerados.")