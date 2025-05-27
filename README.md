#  Detecção e Recorte de Objetos com YOLOv8 + OpenCV

Este projeto utiliza o modelo **YOLOv8 treinado sob medida** para detectar objetos em tempo real via webcam, desenhar as bounding boxes no vídeo ao vivo e **salvar automaticamente imagens recortadas dos objetos detectados** com nomes únicos. Toda a lógica é construída em Python com suporte à GPU (quando disponível), integrando PyTorch, OpenCV e Ultralytics.

> Uma solução prática de visão computacional em tempo real, que mostra domínio sobre inferência de modelos, manipulação de imagem, desempenho e organização de dados visuais.

---

## 🚀 Funcionalidades

- ⚡ **Inferência em tempo real** com alta taxa de frames por segundo (FPS).
- 🖼️ **Recorte automático** de regiões de interesse detectadas.
- 🧠 **Modelo YOLOv8 customizado** carregado diretamente via Ultralytics + PyTorch.
- 🗂️ **Armazenamento inteligente** das imagens detectadas com UUIDs.
- 🛠️ Compatível com **GPU (CUDA)** ou **CPU**.
- ✅ Código modular, com função dedicada para leitura de labels e salvamento.

---

## 🧪 Tecnologias Utilizadas

- [Python 3.8+](https://www.python.org/)
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyYAML](https://pyyaml.org/) para leitura da `data.yaml`
- Threads e UUID para organização de arquivos

---

## 📁 Estrutura do Projeto

```
📂 detected_objects/      # Imagens recortadas dos objetos detectados
📄 best.pt                # Modelo YOLOv8 treinado
📄 data.yaml              # Arquivo com mapeamento das classes
📄 detect_and_crop.py     # Script principal
📄 README.md              # Este documento
```

---

## 🔧 Como Executar

1. Instale as dependências:

```bash
pip install torch opencv-python ultralytics pyyaml
```

2. Certifique-se de que os arquivos `best.pt` e `data.yaml` estejam no local correto e ajustados ao seu modelo.

3. No script, configure o caminho correto na função `getNames()`:

```python
default_path = "CAMINHO/DO/PROJETO"
```

4. Execute o script:

```bash
python detect_and_crop.py
```

---

## 💡 Exemplo de Funcionamento

- O vídeo da webcam será exibido com as **bounding boxes** desenhadas.
- Para cada objeto detectado, será salvo um arquivo no diretório `detected_objects/` com o nome:

```
<classe>_<uuid>.jpg
```

Exemplo:
```
cachorro_1f2a7d8e-5d3c-44c9-a7b2-9b0c6b9df172.jpg
```

- No terminal, a taxa de FPS é impressa a cada segundo:

```
FPS: 28.54
```

---

## 🧠 Aplicações Reais

- 📦 Sistemas de inspeção automática e recorte de produtos
- 🎓 Treinamento de modelos com dados gerados em tempo real
- 🛡️ Prototipagem de soluções de segurança e vigilância
- 🧪 Projetos acadêmicos e laboratoriais em visão computacional
- 📸 Coleta de datasets personalizados por classe

---

## 🤝 Vamos nos conectar?

Este projeto demonstra minha capacidade de integrar **modelos de machine learning**, **processamento de imagens em tempo real** e **organização eficiente de dados visuais**.

Se você é um recrutador ou profissional buscando alguém com experiência prática em **visão computacional, automação de dados ou soluções baseadas em IA**, ficarei feliz em conversar.

---
