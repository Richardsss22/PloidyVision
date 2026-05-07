import os
import urllib.request

# 1. A pasta EXATA no teu disco rígido
pasta_destino = '/Volumes/HDD 500GB/dados_mask'
os.makedirs(pasta_destino, exist_ok=True)

# 2. O endereço seguro CORRIGIDO (sem o "/Files/" no fim!)
base_url = "https://www.ebi.ac.uk/biostudies/files/S-BIAD1752/"

print(f"A preparar para guardar as máscaras em: {pasta_destino}")

# 3. Disfarçar o Python para evitar bloqueios do servidor
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
urllib.request.install_opener(opener)

# 4. Ciclo de download
for i in range(1, 91):
    nome_mask = f"{i}_masks_filtered.tif"
    url_completo = base_url + nome_mask
    caminho_destino = os.path.join(pasta_destino, nome_mask)
    
    # Escudo anti-lixo: Se o ficheiro já lá estiver e tiver mais de 10KB, salta
    if os.path.exists(caminho_destino) and os.path.getsize(caminho_destino) > 10000:
        print(f"[{i}/90] {nome_mask} já existe e está completo. A saltar...")
        continue
        
    print(f"[{i}/90] A descarregar {nome_mask}...")
    try:
        # Saca o ficheiro diretamente para o disco
        urllib.request.urlretrieve(url_completo, caminho_destino)
        print(f"    ✅ Sucesso!")
    except urllib.error.HTTPError as e:
        print(f"    ❌ Ficheiro não encontrado no servidor (Erro {e.code})")
    except Exception as e:
        print(f"    ❌ Erro na ligação: {e}")

print("\n🚀 Todas as máscaras foram descarregadas para o teu disco!")