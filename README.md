# Sistema de Detecção de Drones (Acústico + RF em projeto)

Este repositório reúne o código do protótipo de detecção de drones desenvolvido para o TCC, com foco em:

- detecção **acústica** em tempo quase real (microfone + MFCC + RandomForest);  
- interface gráfica simples para o operador (Tkinter);  
- motor de **decisão geral** preparado para combinar múltiplos sensores;  
- módulo de **radiofrequência (RF)** descrito e integrado como *simulação* (sem implementação física nesta versão).

A versão atual entrega, na prática, a parte acústica + interface gráfica + decisão geral.  
O canal de RF aparece na interface e no código, mas ainda não trabalha com sinal real.

1º:
    Lembre de baixar todas as Libs do requirements, até porque, ele existe pra isso 
    Obs: A versão que eles estão, e bom deixar como esta, porque ja tive problemas tentando atualizar. 

2º:
    Quando for instala o python voce precisa baixar o 3.11 lembre-se disso.

3º: 
    Para Clonar o projeto lembre de por a pasta em um local de facil acesso, por exemplo: C:\Users\SeuUsuario\Documents\TCC_Drone

4º:
    Lembre de criar a sua .venv(ambiente virtual) que e onde vai ficar guardado, basicamnete, tudo do projeto.

    Por exemplo:
cd C:\Users\SeuUsuario\Documents\TCC_Drone 
python -m venv .venv

    Eu ja passei pela situação de ser bloqueado então caso ocorra, faça isso, talvez ajude voce como me ajudou
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\activate

5º:
    Caso precise, normalmente e bom so caso atualizar o pip, mas pra instala tudo de uma vez voce pode tentar:
pip install --upgrade pip
pip install -r requirements.txt

6º: O MAIS IMPORTANTE, AJUSTAR O MICROFONE CORRETO
    Sim, voce leu corretamente, ajustar o microfone, então voce vai mandar:

python -m sounddevice

    E isso deve listar os microfos que voce tem junto dos seus respectivos drives, por isso pode parecer duplicado e etc, e meio chato de achar o melhor, demora um pouca mas voce pega o jeito.
    Se voce precisar trocar o mic, va para: UI_UX/acoustic_detector_core.py, e la vai aparecer:

import sounddevice as sd

sd.default.device = (1, None)  # usar o device 1 como entrada, saída padrão

    E so trocar e deve funcionar(pra mim funcionou pelo menos).

6º: LUCRAR
    agora e so mandar o comando basico, inicia a venv(dentro da sua pasta claro) e inicia o APP:

cd C:\Users\SeuUsuario\Documents\TCC_Drone
.\.venv\Scripts\activate

python UI_UX\detector_app_gui.py

FIM (por enquanto).
