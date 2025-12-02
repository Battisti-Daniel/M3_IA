# 🚗 Sistema de Detecção de Placas de Veículos

Sistema completo para detecção e reconhecimento de placas de veículos em tempo real utilizando YOLO11 e EasyOCR com aceleração por GPU (CUDA).

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Requisitos](#-requisitos)
- [Instalação](#-instalação)
- [Execução](#-execução)
- [Uso do Sistema](#-uso-do-sistema)
- [Configurações](#-configurações)
- [Modelos Disponíveis](#-modelos-disponíveis)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Solução de Problemas](#-solução-de-problemas)

---

## 🎯 Visão Geral

Este sistema utiliza inteligência artificial para:
1. **Detectar veículos** em vídeos, webcams ou capturas de tela
2. **Localizar placas** nos veículos detectados
3. **Reconhecer o texto** das placas usando OCR
4. **Rastrear veículos** entre frames usando ByteTrack
5. **Registrar detecções** em arquivo CSV

---

## ✨ Funcionalidades

- 📹 **Múltiplas fontes de vídeo**: Webcam, arquivos de vídeo, captura de tela
- 🎯 **Detecção em tempo real** com YOLO11
- 🔤 **OCR otimizado** com EasyOCR para placas brasileiras
- 🚀 **Aceleração GPU** com CUDA para máximo desempenho
- 📊 **Interface gráfica** intuitiva com visualização em tempo real
- 📁 **Log de detecções** em formato CSV
- ⚙️ **Altamente configurável** via arquivo de configuração

---

## 💻 Requisitos

### Hardware
- **GPU NVIDIA** com suporte a CUDA (recomendado)
  - Mínimo: GTX 1050 ou superior
  - Testado em: GTX 1650
- **RAM**: Mínimo 8GB (16GB recomendado)
- **Espaço em disco**: ~5GB para modelos e dependências

### Software
- **Sistema Operacional**: Windows 10/11
- **Python**: 3.11 (obrigatório - outras versões podem causar incompatibilidades)
- **CUDA Toolkit**: 12.4 (instalado automaticamente via PyTorch)
- **Drivers NVIDIA**: Versão atualizada compatível com CUDA 12.4

### Verificar instalação do Python 3.11
```powershell
py -3.11 --version
```

Se não tiver Python 3.11, baixe em: https://www.python.org/downloads/release/python-3119/

---

## 🔧 Instalação

### Passo 1: Clonar ou baixar o projeto

```powershell
# Se estiver usando Git
git clone https://github.com/Battisti-Daniel/M3_IA.git
cd Trabalho_M3
```

Ou baixe o ZIP e extraia na pasta desejada.

### Passo 2: Executar o script de instalação

```powershell
# Clique duas vezes no arquivo ou execute:
.\setup.bat
```

Este script irá:
1. ✅ Criar um ambiente virtual Python (`venv`)
2. ✅ Instalar PyTorch com suporte a CUDA 12.4
3. ✅ Instalar todas as dependências necessárias
4. ✅ Verificar se a GPU está sendo reconhecida

**⏱️ Tempo estimado**: 5-15 minutos (dependendo da conexão de internet)

### Passo 3: Baixar os modelos de IA

```powershell
# Clique duas vezes no arquivo ou execute:
.\download-models.bat
```

Este script irá baixar os modelos YOLO treinados:
- Modelos de detecção de veículos (~25MB)
- Modelos de detecção de placas (~140MB)

**⏱️ Tempo estimado**: 2-5 minutos

---

## ▶️ Execução

### Iniciar o sistema

```powershell
# Clique duas vezes no arquivo ou execute:
.\run.bat
```

Isso irá:
1. Ativar o ambiente virtual
2. Executar a interface gráfica do sistema

---

## 🖥️ Uso do Sistema

### Interface Principal

Após iniciar o sistema, você verá a interface gráfica com as seguintes opções:

#### 1. Seleção da Fonte de Vídeo
- **Webcam 0/1**: Usar webcam conectada
- **Arquivo de Vídeo**: Selecionar um arquivo MP4, AVI, etc.
- **Captura de Tela**: Selecionar uma região da tela para capturar

#### 2. Seleção de Modelos
- **Modelo de Veículos**: Escolher entre yolo11n (rápido) ou yolo11s (preciso)
- **Modelo de Placas**: Escolher entre license-plate-v1n (rápido) ou license-plate-v1s (preciso)

#### 3. Controles
- **▶️ Iniciar**: Começar a detecção
- **⏹️ Parar**: Parar a detecção
- **📊 Ver Logs**: Abrir arquivo de detecções

### Visualização

Durante a execução, você verá:
- 🟢 **Caixas verdes**: Veículos detectados
- 🔵 **Caixas azuis**: Placas detectadas
- 📝 **Texto**: Placa reconhecida com porcentagem de confiança
- 📈 **FPS**: Taxa de quadros por segundo no canto superior

### Logs de Detecção

As detecções são salvas em `logs/detections.csv` com as seguintes informações:
- Data/hora da detecção
- Texto da placa
- Tipo de placa (Mercosul/Antiga)
- Confiança da detecção
- Confiança do OCR
- Tipo de veículo

---

## ⚙️ Configurações

Edite o arquivo `src/config.py` para personalizar o sistema:

### Configurações de Detecção
```python
MIN_PLATE_CONFIDENCE = 0.50  # Confiança mínima YOLO (0.0 a 1.0)
MIN_OCR_CONFIDENCE = 0.30    # Confiança mínima OCR (0.0 a 1.0)
MIN_PLATE_LENGTH = 3         # Tamanho mínimo do texto da placa
MAX_PLATE_LENGTH = 8         # Tamanho máximo do texto da placa
```

### Configurações de Performance
```python
YOLO_IMGSZ = 640             # Tamanho da imagem (320=rápido, 640=preciso)
OCR_SKIP_FRAMES = 3          # Executar OCR a cada N frames
DEFAULT_TARGET_FPS = 20      # FPS alvo do sistema
```

### Configurações de Visualização
```python
SHOW_VIDEO_OVERLAY = True    # Mostrar caixas e texto no vídeo
SHOW_FPS_OVERLAY = True      # Mostrar contador de FPS
SHOW_VEHICLE_BBOX = True     # Mostrar caixa do veículo
SHOW_PLATE_TEXT = True       # Mostrar texto da placa
SHOW_CONFIDENCE = True       # Mostrar porcentagem de confiança
```

---

## 🤖 Modelos Disponíveis

### Detecção de Veículos (YOLO11)

| Modelo | Tamanho | Velocidade | Precisão | Recomendado |
|--------|---------|------------|----------|-------------|
| `yolo11n.pt` | 5.6MB | ⚡⚡⚡ Muito rápido | ⭐⭐ | GPU fraca |
| `yolo11s.pt` | 19MB | ⚡⚡ Rápido | ⭐⭐⭐ | ✅ Geral |

### Detecção de Placas (Fine-tuned)

| Modelo | Tamanho | Velocidade | Precisão | Recomendado |
|--------|---------|------------|----------|-------------|
| `license-plate-v1n.pt` | 5.4MB | ⚡⚡⚡ Muito rápido | ⭐⭐ | GPU fraca |
| `license-plate-v1s.pt` | 19MB | ⚡⚡ Rápido | ⭐⭐⭐ | ✅ Geral |
| `license-plate-v1x.pt` | 114MB | ⚡ Lento | ⭐⭐⭐⭐ | Alta precisão |
| `nosso_modelo_yolo11n.pt` | 5.6MB | ⚡⚡⚡ Muito rápido | ⭐⭐ | Treinamento local |

---

## 📂 Estrutura do Projeto

```
M4/
├── 📄 main.py              # Ponto de entrada da aplicação
├── 📄 setup.bat            # Script de instalação
├── 📄 run.bat              # Script de execução
├── 📄 download-models.bat  # Download dos modelos
├── 📄 requirements.txt     # Dependências Python
├── 📄 README.md            # Esta documentação
│
├── 📁 src/                 # Código fonte
│   ├── config.py           # Configurações do sistema
│   ├── data_structures.py  # Estruturas de dados
│   ├── device.py           # Detecção de GPU/CPU
│   ├── model_manager.py    # Gerenciamento de modelos YOLO
│   ├── ocr.py              # Reconhecimento de texto (OCR)
│   ├── pipeline.py         # Pipeline de processamento
│   ├── preprocessing.py    # Pré-processamento de imagens
│   ├── tracker.py          # Rastreamento de veículos
│   └── ui.py               # Interface gráfica
│
├── 📁 models/              # Modelos de IA
│   ├── yolo11n.pt          # YOLO Nano (veículos)
│   ├── yolo11s.pt          # YOLO Small (veículos)
│   ├── license-plate-v1n.pt # Placas Nano
│   ├── license-plate-v1s.pt # Placas Small
│   └── license-plate-v1x.pt # Placas Extra-Large
│
├── 📁 logs/                # Logs de detecção
│   └── detections.csv      # Registro de placas detectadas
│
└── 📁 venv/                # Ambiente virtual Python (gerado)
```

---

## 🔧 Solução de Problemas

### ❌ Erro: "Python 3.11 não encontrado"

**Problema**: O sistema requer Python 3.11 especificamente.

**Solução**:
1. Baixe Python 3.11 em: https://www.python.org/downloads/release/python-3119/
2. Durante a instalação, marque "Add Python to PATH"
3. Execute novamente `setup.bat`

### ❌ Erro: "CUDA não disponível"

**Problema**: PyTorch não está usando a GPU.

**Solução**:
1. Verifique se tem uma GPU NVIDIA:
   ```powershell
   nvidia-smi
   ```
2. Atualize os drivers da NVIDIA
3. Delete a pasta `venv` e execute `setup.bat` novamente

### ❌ Erro: "Modelo não encontrado"

**Problema**: Os modelos YOLO não foram baixados.

**Solução**:
```powershell
.\download-models.bat
```

### ❌ Sistema lento / baixo FPS

**Problema**: Performance abaixo do esperado.

**Soluções**:
1. Use modelos menores (nano em vez de small)
2. Reduza o tamanho da imagem em `config.py`:
   ```python
   YOLO_IMGSZ = 320  # Em vez de 640
   ```
3. Aumente o intervalo do OCR:
   ```python
   OCR_SKIP_FRAMES = 5  # Em vez de 3
   ```

### ❌ Webcam não detectada

**Problema**: Sistema não encontra a webcam.

**Solução**:
1. Verifique se a webcam está conectada
2. Teste em outro programa (ex: aplicativo Câmera do Windows)
3. Tente usar "Webcam 1" em vez de "Webcam 0"

### ❌ OCR não reconhece a placa

**Problema**: Placa detectada mas texto incorreto.

**Soluções**:
1. Melhore a iluminação
2. Aproxime a câmera da placa
3. Reduza a confiança mínima:
   ```python
   MIN_OCR_CONFIDENCE = 0.20
   ```

---

## 📊 Verificar se GPU está funcionando

Execute este comando para verificar:

```powershell
.\venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma')"
```

**Saída esperada**:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1650
```

---

## 📝 Licença

Este projeto foi desenvolvido para fins educacionais na disciplina de Inteligência Artificial II.

---

## 👥 Autores

Desenvolvido por estudantes da faculdade como projeto do Módulo 4.

---

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO11
- [JaidedAI](https://github.com/JaidedAI/EasyOCR) - EasyOCR
- [PyTorch](https://pytorch.org/) - Framework de Deep Learning
- [HuggingFace](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) - Modelos de placas
