import os
import urllib.request

# 1. COLA AQUI O CAMINHO QUE COPIASTE DO MAC!
# Vai ser algo parecido com isto: '/Volumes/O_Meu_Disco_Rigido/dados_microscopia/'
pasta_destino = '/Volumes/HDD 500GB/dados_microscopia'

# O código garante que a pasta existe antes de começar
os.makedirs(pasta_destino, exist_ok=True)

# 2. O link base do BioImage Archive
base_url = "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/752/S-BIAD1752/Files/"

print(f"A preparar para guardar as imagens em: {pasta_destino}")
print("A iniciar o download das tiles...")

# 3. Ciclo para sacar as tiles da 1 à 90
for i in range(32, 35):
    nome_ficheiro = f"tile_{i}.tif"
    url_completo = base_url + nome_ficheiro
    caminho_destino = os.path.join(pasta_destino, nome_ficheiro)
    
    # Se a net cair e tiveres de recomeçar, ele não saca os que já estão no disco!
    if os.path.exists(caminho_destino):
        print(f"[{i}/90] {nome_ficheiro} já existe no disco. A saltar...")
        continue
        
    print(f"[{i}/90] A descarregar {nome_ficheiro}...")
    
    try:
        # Faz o download direto para o disco externo
        urllib.request.urlretrieve(url_completo, caminho_destino)
    except Exception as e:
        print(f" ❌ Erro ao descarregar {nome_ficheiro}: {e}")

print("\n🚀 Download de todas as imagens concluído com sucesso!")