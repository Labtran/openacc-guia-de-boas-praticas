Avaliar o Desempenho da aplicação
==============================
Uma variedade de ferramentas pode ser utilizada para avaliar o desempenho da aplicação e
a disponibilidade, dependerá do seu ambiente de desenvolvimento. De simples
temporizadores de aplicação a analisadores gráficos de desempenho, a escolha de
ferramenta de análise do desempenho está fora do âmbito deste documento. O objectivo
desta secção é fornecer orientação na escolha de secções importantes do código
para a aceleração, que é independente das ferramentas de perfilagem disponíveis.

Ao longo deste guia, a ferramenta de análise do desempenho da NVIDIA Nsight Systems, que é fornecida com o conjunto de ferramentas CUDA, será utilizada para a definição do perfil da CPU. Quando for necessário o perfil do acelerador, a aplicação será executada numa GPU NVIDIA e o perfilador NVIDIA Nsight Systems será novamente utilizado.

Perfil de linha de base
------------------
Antes de paralelizar uma aplicação com OpenACC o programador deve primeiro
compreender onde o tempo está actualmente a ser gasto no código. Rotinas e laços
que ocupam uma percentagem significativa do tempo de execução são frequentemente referidas
como *hot spots* e será o ponto de partida para acelerar a aplicação. 
Existe uma variedade de ferramentas para gerar perfis de aplicação, como o gprof,
Vampir, Nsight Systems, e TAU. A seleção da ferramenta específica que funciona 
melhor para uma determinada aplicação está fora do âmbito deste documento, mas independentemente 
de que ferramenta ou ferramentas são utilizadas abaixo estão algumas peças importantes de informação
que ajudará a orientar os próximos passos na paralelização da aplicação.

* Desempenho da aplicação - Quanto tempo leva a aplicação a ser executada? Como
  o programa utiliza eficazmente os recursos computacionais? 
* Hotspots de programa - Quais as rotinas do programa são as que gastam a maior parte dos seus
  tempo? O que está a ser feito dentro destas rotinas importantes? Concentrando a atenção na
  a maioria das partes demoradas da aplicação produzirá os maiores resultados.
* Limitadores de desempenho - Dentro dos hotspots identificados, o que está atualmente
  limitando o desempenho da aplicação? Alguns limitadores comuns podem ser I/O,
  largura de banda de memória , reutilização de cache, desempenho de ponto flutuante, comunicação, etc.
  Uma forma de avaliar os limitadores de desempenho de um determinado ninho de laço é
  avaliar a sua *intensidade computacional*, que é uma medida de quantas
  operações são realizadas num elemento de dados por carga ou armazenagem a partir da memória. 
* Paralelismo disponível - Examine os laços dentro dos hotspots para compreender
  quanto trabalho realiza cada ninho de laço. Faça os laços iterarem 10's, 100's,
  1000's de vezes (ou mais)? As iterações do laço funcionam independentemente de
  um ao outro? Olhe não só para os laços individuais, mas olhe para um ninho de laços
  para compreender o panorama geral de todo o ninho. 

A obtenção de dados de base como os acima referidos ajuda a informar o programador onde
concentrar esforços para obter os melhores resultados e fornecer uma base de comparação
desempenho ao longo de todo o resto do processo. É importante escolher o input
que reflectirá de forma realista como a aplicação será utilizada uma vez que
foi acelerada. É tentador utilizar um problema de referência conhecido para a definição de perfis,
mas frequentemente estes problemas de referência utilizam uma dimensão reduzida do problema ou uma redução
I/O, o que pode levar a pressupostos incorrectos sobre o desempenho do programa. Muitos
dos programadores também utilizam o perfil de base para reunir os resultados esperados da
aplicação e utilizam para verificar a correcção da aplicação tal como está
acelerado.

Perfis adicionais
--------------------
Através do processo de portabilidade e optimização de uma aplicação com OpenACC, é
necessário para recolher dados de perfil adicionais seguir os próximos passos no
processo. Algumas ferramentas de perfilagem, tais como Nsight Systems e Vampir, apoiam a perfilagem em
CPUs e GPUs, enquanto outras ferramentas, tais como o gprof,
só suportam a definição de perfis numa plataforma específica. Além disso, alguns compiladores
constroem o seu próprio perfil na aplicação, como é o caso do NVHPC
que suporta a definição da variável de ambiente NVCOMPILER\_ACC\_TIME para 
obter informações sobre o tempo de execução da aplicação. Ao desenvolver-se em
plataformas offload, tais como plataformas CPU + GPU, é geralmente importante utilizar uma ferramenta de elaboração de perfis ao longo do processo de desenvolvimento que possa avaliar ambos
tempo gasto em computação e tempo gasto na execução de transferências de dados PCIe. Isto
utilizará o NVIDIA Nsight Systems Profiler para realizar esta análise, embora
só está disponível nas plataformas NVIDIA.

Estudo de caso - Análise
---------------------
Para melhor compreensão do programa de estudo de casos, utilizaremos o
NVIDIA NSight Systems interface de linha de comando que faz parte do CUDA Toolkit e NVIDIA HPC SDK. Primeiro,
é necessário construir o executável. Lembre-se de utilizar as flags incluídas no exemplo abaixo para assegurar que a informação adicional sobre como o
compilador optimizou o programa seja apresentado. O executável é construído com o
seguindo o comando:

~~~~
    $ nvc -fast -Minfo=all laplace2d.c
    GetTimer:
         21, include "timer.h"
              61, FMA (fused multiply-add) instruction(s) generated
    main:
         41, Loop not fused: function call before adjacent loop
             Loop unrolled 8 times
         49, StartTimer inlined, size=2 (inline) file laplace2d.c (37)
         52, FMA (fused multiply-add) instruction(s) generated
         58, Generated vector simd code for the loop containing reductions
         68, Memory copy idiom, loop replaced by call to __c_mcopy8
         79, GetTimer inlined, size=10 (inline) file laplace2d.c (54)
~~~~

Uma vez construído o executável, o comando 'nsys' irá executar o
executável e gerar um relatório de perfil que pode ser visto offline na NVIDIA Nsight Systems GUI

~~~~
    $ nsys profile ./a.out
    
    Jacobi relaxation Calculation: 4096 x 4096 mesh
        0, 0.250000
      100, 0.002397
      200, 0.001204
      300, 0.000804
      400, 0.000603
      500, 0.000483
      600, 0.000403
      700, 0.000345
      800, 0.000302
      900, 0.000269
     total: 36.480533 s
     Processing events...
Capturing symbol files...
Saving temporary "/tmp/nsys-report-2f5b-f32e-7dec-9af0.qdstrm" file to disk...
Creating final output files...

Processing [==============================================================100%]
Saved report file to "/tmp/nsys-report-2f5b-f32e-7dec-9af0.qdrep"
Report file moved to "/home/ubuntu/openacc-programming-guide/examples/laplace/ch2/report1.qdrep"
~~~~

Uma vez recolhidos os dados, e uma vez gerado o relatório .qdrep,
pode ser visualizado utilizando a GUI da Nsight Systems. É necessário copiar primeiro o
(report1.qdrep no exemplo acima) para uma máquina que tem capacidades gráficas
e descarregar a interface do Nsight Systems. A seguir, deve abrir
a aplicação e seleccionar o seu ficheiro através do gestor de ficheiros.

![Nsight Systems initial window in the GUI. You must use the toolbar at the top to find your target report file](images/ch2-nsight-open.png)

Quando abrimos o relatório em Nsight Systems, vemos que a grande maioria dos
o tempo é despendido em duas rotinas:main e \_\_c\_mcopy8.. Uma captura de tela da tela inicial do sistema Nsight é mostrado na figura 2.1. Uma vez que o código
para este estudo de caso está completamente dentro da função principal do programa,
não é surpreendente que quase todo o tempo seja gasto em geral, mas em
aplicações maiores, é provável que o tempo seja gasto em várias outras
rotinas.

![Nsight initial profile window showing 81% of runtime in main and 17% in a
memory copy routine.](images/ch2-nsight-initial.png)

Clicando na função principal, podemos ver que quase todo o tempo de execução
dentro do main vem do laço que calcula o próximo valor para A. Isto é
mostrado na figura 2.2. O que não é óbvio na saída do perfilador,
no entanto, é que o tempo gasto na rotina de cópia de memória mostrada na tela inicial
é na realidade o segundo ninho de laço, que executa a troca de matriz no
fim de cada iteração. A saída do compilador mostra acima que o laço na linha
68 foi substituída por uma cópia de memória, porque fazê-lo é mais eficiente do que
copiando cada elemento individualmente. Assim, o que o profiler nos está realmente a mostrar
é que os principais hotsspots para a nossa aplicação são o ninho de laço que
calcula `Anew` de `A` e o ninho de laço que copia de `Anew` a `A'.
para a próxima iteração, por isso vamos concentrar os nossos esforços nestas duas etapas
ninhos.

Nos capítulos que se seguem, optimizaremos os laços identificados neste
capítulo como os hotspots dentro da nossa aplicação de exemplo.
