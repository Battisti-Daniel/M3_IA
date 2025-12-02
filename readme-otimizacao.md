# Otimizações aplicadas

## Resumo
- Reduzimos cópias e operações redundantes durante o ciclo de processamento para aliviar o consumo por frame.
- Evitamos execuções de OCR/YOLO desnecessárias filtrando detecções pouco confiáveis ou com recortes mínimos.
- Alinhamos o uso de precisão mista para ser calculado uma vez e reaproveitado ao longo de toda a pipeline.

## Detalhes por componente

### Pipeline
- **Cache de precisão do dispositivo**: armazenamos `self.half_precision` na inicialização e reutilizamos nas chamadas do YOLO de veículos e placas, evitando cálculos repetidos a cada frame. Isso reduz overhead em loops apertados. 
- **Remoção de trilhas não utilizadas**: eliminamos `update_trail()` em cada track, já que a renderização do rastro está desativada; a mudança corta operações de lista por objeto rastreado.
- **Filtros antecipados de placa**: adicionamos verificação de confiança mínima (`MIN_PLATE_CONFIDENCE`) e tamanho mínimo de bounding box antes de recortar e enviar para OCR, economizando processamento em detecções ruidosas ou muito pequenas.
- **Reuso de frame**: o laço principal agora envia o frame original para o pipeline, evitando a cópia extra por iteração.

### UI / Loop principal
- **Processamento sem cópia**: o `_loop` passa o frame diretamente para `process_frame`, eliminando duplicação de memória por frame e acelerando a etapa de leitura + inferência.

## Próximos passos sugeridos
- Ajustar `YOLO_IMGSZ` e `OCR_SKIP_FRAMES` conforme o hardware disponível para equilibrar velocidade e precisão.
- Habilitar modelos "nano" quando a prioridade for FPS.

## Dependência do ByteTrack
- A implementação do ByteTrack usada no `src/tracker.py` é **nativa** (não depende do pacote oficial),
  portanto não exige nenhuma instalação extra além das bibliotecas já listadas em `requirements.txt`.
