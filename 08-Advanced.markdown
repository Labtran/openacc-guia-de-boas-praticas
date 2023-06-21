Características avançadas do OpenACC
=========================
Este capítulo discutirá os recursos e técnicas do OpenACC que não se encaixam perfeitamente em outras secções do guia. Estas técnicas são consideradas
avançadas, portanto os leitores devem se sentir confortáveis com os recursos discutidos nos
capítulos anteriores antes de prosseguir para este capítulo.

Funcionamento assíncrono
----------------------
No capítulo anterior, discutimos a necessidade de otimizar a localidade dos dados
para reduzir o custo das transferências de dados em sistemas onde o hospedeiro e o acelerador
possuem memórias fisicamente distintas. Haverá sempre alguma quantidade de transferências de dados
que simplesmente não podem ser otimizadas e ainda produzir
resultados corretos. Após minimizar as transferências de dados, pode ser possível reduzir ainda mais
a penalidade de desempenho associada a essas transferências, sobrepondo as
cópias com outras operações no host, no dispositivo ou em ambos. Isto pode ser alcançado
com OpenACC usando a cláusula `async`. A cláusula `async` pode ser adicionada a
`parallel`, `kernels`, e `update` para especificar que uma vez que a operação associada tenha sido enviada para o acelerador ou para execução de CPU pode continuar fazendo outras coisas, ao invés de esperar que a operação do
acelerador seja concluída. Isso pode incluir o enfileiramento de operações adicionais do acelerador ou computar outro trabalho que não esteja relacionado com o trabalho
que está sendo executado pelo acelerador. O código abaixo demonstra a adição da cláusula
`async` a um `parallel loop` e uma diretiva `update` que segue:

~~~~ {.c .numberLines}
    #pragma acc parallel loop async
    for (int i=0; i<N; i++)
    {
      c[i] = a[i] + b[i]
    }
    #pragma acc update self(c[0:N]) async
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop async
    do i=1,N
      c(i) = a(i) + b(i)
    end do
    !$acc update self(c) async
~~~~

No caso acima, o thread do host enfileirará a região paralela na
*fila assíncrona padrão*, então a execução retornará à thread do host para que ela também possa enfileirar o `update`, e finalmente a thread da CPU continuará a
execução. Eventualmente, no entanto, a thread hospedeira precisará dos resultados computados
no acelerador e copiados de volta para o host usando o `update`, então ela deve
sincronizar com o acelerador para garantir que essas operações tenham terminado
antes de tentar usar os dados. A diretiva `wait` instrui o tempo de execução
a aguardar que as operações assíncronas passadas sejam concluídas antes de prosseguir. Assim, os exemplos
acima podem ser estendidos para incluir uma sincronização antes que os dados
que estão sendo copiados pela diretiva `update`.

~~~~ {.c .numberLines}
    #pragma acc parallel loop async
    for (int i=0; i<N; i++)
    {
      c[i] = a[i] + b[i]
    }
    #pragma acc update self(c[0:N]) async
    #pragma acc wait
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop async
    do i=1,N
      c(i) = a(i) + b(i)
    end do
    !$acc update self(c) async
    !$acc wait
~~~~

Embora isso seja útil, seria ainda mais útil expor as dependências
dessas operações assíncronas e as esperas associadas, de modo que operações independentes
pudessem ser potencialmente executadas concorrentemente. Tanto `async` quanto `wait`
têm um argumento opcional para um número inteiro não-negativo que especifica um número de fila para aquela operação. Todas as operações colocadas na mesma fila irão
operam em ordem, mas operações colocadas em filas diferentes podem operar em qualquer
ordem entre si. As operações em filas diferentes podem, mas
não é garantido que funcionem em paralelo. Estas filas de trabalho são únicas
por dispositivo, pelo que dois dispositivos terão filas distintas com o mesmo número. Se
um `wait` for encontrado sem um argumento, ele irá esperar todo o
trabalho previamente enfileirado naquele dispositivo. O estudo de caso abaixo demonstrará como
utilizar diferentes filas de trabalho para obter sobreposição de computação e
transferências de dados.

Para além de ser possível colocar operações em filas separadas, seria
útil ser capaz de juntar essas filas num ponto em que os resultados de
ambos são necessários antes de prosseguir. Isto pode ser conseguido adicionando uma cláusula `async`
a um `wait`. Isso pode parecer pouco intuitivo, então o código abaixo demonstra
como isso é feito.

~~~~ {.c .numberLines}
    #pragma acc parallel loop async(1)
    for (int i=0; i<N; i++)
    {
      a[i] = i;
    }
    #pragma acc parallel loop async(2)
    for (int i=0; i<N; i++)
    {
      b[i] = 2*i;
    }
    #pragma acc wait(1) async(2)
    #pragma acc parallel loop async(2)
    for (int i=0; i<N; i++)
    {
      c[i] = a[i] + b[i]
    }
    #pragma acc update self(c[0:N]) async(2)
    #pragma acc wait
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop async(1)
    do i=1,N
      a(i) = i
    end do
    !$acc parallel loop async(2)
    do i=1,N
      b(i) = 2.0 * i
    end do
    !$acc wait(1) async(2)
    !$acc parallel loop async(2)
    do i=1,N
      c(i) = a(i) + b(i)
    end do
    !$acc update self(c) async(2)
    !$acc wait
~~~~

O código acima inicializa os valores contidos em `a` e `b` usando filas de trabalho
separadas para que eles possam ser potencialmente feitos de forma independente. A função `wait(1)
async(2)` garante que a fila de trabalho 2 não prossiga até que a fila 1 tenha
concluída. A adição do vetor é então capaz de ser enfileirada no dispositivo
porque os kernels anteriores terão sido concluídos antes deste ponto. Por último
o código espera que todas as operações anteriores sejam concluídas. Usando essa técnica
expressamos as dependências de nossos loops para maximizar a simultaneidade entre as
regiões, mas ainda assim dando resultados corretos.

***Boas Práticas:*** O custo de enviar uma operação para o acelerador para
execução é frequentemente bastante alto em aceleradores de descarregamento, como GPUs
conectadas através de um barramento PCIe a uma CPU hospedeira. Uma vez que os loops e as transferências de dados
dentro de uma rotina tiverem sido testados, é freqüentemente benéfico fazer cada
região paralela e atualizar de forma assíncrona e então colocar uma diretiva `wait
após a última diretiva do acelerador. Isso permite que o tempo de execução enfileire todo o
o trabalho imediatamente, o que reduzirá a frequência com que o acelerador e o host devem
sincronizar e reduzir o custo de lançar o trabalho no acelerador. É
crítico, ao implementar esta otimização, que o desenvolvedor não deixe
o `wait` após a última diretiva do acelerador, caso contrário o código
provavelmente produzirá resultados incorretos. Esta é uma otimização tão benéfica
que alguns compiladores fornecem uma opção de tempo de compilação para habilitar isso para todas as
diretivas de aceleração automaticamente.

### Estudo de caso: Pipelining assíncrono de um conjunto Mandelbrot ###

![Mandelbrot Set Output](images/mandelbrot.png)

Para este exemplo, vamos modificar uma aplicação simples que gera um conjunto
mandelbrot, como a imagem mostrada acima. Uma vez que cada pixel da imagem
pode ser calculado de forma independente, o código é trivial para paralelizar, mas
devido ao grande tamanho da imagem em si, a transferência de dados para copiar os
resultados de volta para o hospedeiro antes de escrever num ficheiro de imagem é dispendiosa. Como essa
transferência de dados deve ocorrer, seria bom sobrepô-la à computação,
mas como o código está escrito abaixo, todo o cálculo tem de ocorrer antes de a cópia
possa ocorrer, portanto não há nada para sobrepor. *(Nota: A função `mandelbrot`
é uma função sequencial usada para calcular o valor de cada pixel. Ela
é deixada de fora deste capítulo para poupar espaço, mas está incluída nos
exemplos.)*

~~~~ {.c .numberLines}
    #pragma acc parallel loop
    for(int y=0;y<HEIGHT;y++) {
      for(int x=0;x<WIDTH;x++) {
        image[y*WIDTH+x]=mandelbrot(x,y);
      }
    }
    
    #pragma acc update self(image[:WIDTH*HEIGHT])
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop
    do iy=1,width
      do ix=1,HEIGHT
        image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
      enddo
    enddo
    
    !$acc update self(image)
~~~~

Uma vez que cada pixel é independente um do outro, é possível utilizar uma técnica
conhecida como pipelining para dividir a geração da imagem em partes mais pequenas,
que permite que o resultado de cada parte seja copiado enquanto a parte seguinte está a ser
sendo computada. A figura abaixo demonstra um pipeline idealizado onde a
computação e as cópias são do mesmo tamanho, mas isso raramente ocorre em
aplicações reais. Ao dividir a operação em duas partes, a mesma quantidade de dados
de dados é transferida, mas todas as transferências, exceto a primeira e a última, podem ser sobrepostas com
computação. O número e a dimensão destes pedaços de trabalho mais pequenos podem ser ajustados para encontrar o valor que proporciona o melhor desempenho.

![Idealized Pipeline Showing Overlapping of 2 Independent Operations](images/idealized_pipeline.png)

O código Mandelbrot pode usar esta mesma técnica, dividindo a geração de imagens
e as transferências de dados em partes mais pequenas e independentes.
Isto será feito em várias etapas para reduzir a probabilidade de introdução de um erro. O
O primeiro passo é introduzir um ciclo de bloqueio no cálculo, mas manter as
transferências de dados iguais. Isto irá garantir que o trabalho em si é corretamente
dividido de forma a obter resultados correctos. Após cada passo, o programador deve construir e
executar o código para garantir que a imagem resultante continua correta.

#### Passo 1: Computação de bloqueio ####

O primeiro passo para a geração de imagens em pipelining é introduzir um ciclo
que dividirá a computação em partes de trabalho que podem ser geradas
independentemente. Para o fazer, teremos de decidir quantos blocos de trabalho são desejados e usar isso para determinar os limites de início e fim de cada
bloco. Em seguida, introduzimos um loop adicional em torno dos dois existentes e
modificamos o loop `y` para operar apenas dentro do bloco de trabalho atual
atualizando os limites do loop com o que calculamos como valores inicial e final
para o bloco atual. Os ninhos de laços modificados são mostrados abaixo.

~~~~ {.c .numberLines}
    int num_blocks = 8;
    for(int block = 0; block < num_blocks; block++ ) {
      int ystart = block * (HEIGHT/num_blocks),
          yend   = ystart + (HEIGHT/num_blocks);
    #pragma acc parallel loop
      for(int y=ystart;y<yend;y++) {
        for(int x=0;x<WIDTH;x++) {
          image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }
    }

    #pragma acc update self(image[:WIDTH*HEIGHT])
~~~~

---

~~~~ {.fortran .numberLines}
    num_batches=8
    batch_size=WIDTH/num_batches
    do yp=0,num_batches-1
      ystart = yp * batch_size + 1
      yend   = ystart + batch_size - 1
      !$acc parallel loop
      do iy=ystart,yend
        do ix=1,HEIGHT
          image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
        enddo
      enddo
    enddo

    !$acc update self(image)
~~~~

Nesta altura, apenas confirmámos que podemos gerar com sucesso cada
bloco de trabalho de forma independente. O desempenho deste passo não deve
ser visivelmente melhor do que o código original e pode ser pior.

#### Passo 2: Bloqueio de transferências de dados ####

A próxima etapa do processo consiste em dividir as transferências de dados de e para o
dispositivo da mesma forma que a computação já foi dividida. Para o fazer
precisamos primeiro de introduzir uma região de dados à volta do ciclo de bloqueio. Isso garantirá que a memória do dispositivo usada para armazenar a imagem permanecerá no
dispositivo para todos os blocos de trabalho. Como o valor inicial do array de imagens não é
importante, nós usamos uma cláusula de dados `create` para alocar um array não inicializado
no dispositivo. Em seguida, usamos a diretiva `update` para copiar cada bloco da imagem
do dispositivo para o host depois de ter sido calculado. Para fazer isso,
precisamos determinar o tamanho de cada bloco para garantir que atualizaremos apenas a
parte da imagem que coincide com o bloco de trabalho atual. O código
resultante no final desta etapa está abaixo.

~~~~ {.c .numberLines}
    int num_blocks = 8, block_size = (HEIGHT/num_blocks)*WIDTH;
    #pragma acc data create(image[WIDTH*HEIGHT])
    for(int block = 0; block < num_blocks; block++ ) {
      int ystart = block * (HEIGHT/num_blocks),
          yend   = ystart + (HEIGHT/num_blocks);
    #pragma acc parallel loop
      for(int y=ystart;y<yend;y++) {
        for(int x=0;x<WIDTH;x++) {
          image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }
    #pragma acc update self(image[block*block_size:block_size])
    }
~~~~

---

~~~~ {.fortran .numberLines}
    num_batches=8
    batch_size=WIDTH/num_batches
    call cpu_time(startt)
    !$acc data create(image)
    do yp=0,NUM_BATCHES-1
      ystart = yp * batch_size + 1
      yend   = ystart + batch_size - 1
      !$acc parallel loop
      do iy=ystart,yend
        do ix=1,HEIGHT
          image(ix,iy) = mandelbrot(ix-1,iy-1)
        enddo
      enddo
      !$acc update self(image(:,ystart:yend))
    enddo
    !$acc end data
~~~~

No final deste passo, estamos a calcular e a copiar cada bloco da imagem
independentemente, mas isso ainda está a ser feito sequencialmente, cada bloco após o
anterior. O desempenho no final deste passo é geralmente comparável ao da
à versão original.

#### Passo 3: Cálculo de sobreposição e transferências ####

O último passo deste estudo de caso consiste em tornar assíncronas as operações do dispositivo
para que as cópias independentes e a computação possam ocorrer em simultâneo.
Para tal, utilizaremos filas de trabalho assíncronas para garantir que a computação
e a transferência de dados dentro de um único bloco estejam na mesma fila, mas
blocos separados estejam em filas diferentes. O número do bloco é um identificador
assíncrono para usar nessa mudança. É claro que, como agora estamos operando
completamente assíncrono, é fundamental que adicionemos uma diretiva `wait` após o loop de blocos
para garantir que todo o trabalho seja concluído antes de tentarmos usar os dados da imagem do
do host. O código modificado é encontrado abaixo.

~~~~ {.c .numberLines}
    int num_blocks = 8, block_size = (HEIGHT/num_blocks)*WIDTH;
    #pragma acc data create(image[WIDTH*HEIGHT])
    for(int block = 0; block < num_blocks; block++ ) {
      int ystart = block * (HEIGHT/num_blocks),
          yend   = ystart + (HEIGHT/num_blocks);
    #pragma acc parallel loop async(block)
      for(int y=ystart;y<yend;y++) {
        for(int x=0;x<WIDTH;x++) {
          image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }
    #pragma acc update self(image[block*block_size:block_size]) async(block)
    }
    #pragma acc wait
~~~~

---

~~~~ {.fortran .numberLines}
    num_batches=8
    batch_size=WIDTH/num_batches
    call cpu_time(startt)
    !$acc data create(image)
    do yp=0,NUM_BATCHES-1
      ystart = yp * batch_size + 1
      yend   = ystart + batch_size - 1
      !$acc parallel loop async(yp)
      do iy=ystart,yend
        do ix=1,HEIGHT
          image(ix,iy) = mandelbrot(ix-1,iy-1)
        enddo
      enddo
      !$acc update self(image(:,ystart:yend)) async(yp)
    enddo
    !$acc wait
    !$acc end data
~~~~

Com esta modificação, é agora possível que a parte computacional de um
bloco funcione simultaneamente com a transferência de dados de outro. O programador
deve agora experimentar com diferentes tamanhos de blocos para determinar qual é o valor ótimo
na arquitetura de interesse. É importante notar, no entanto, que
em algumas arquiteturas, o custo de criar uma fila assíncrona na primeira vez que ela é usada pode ser bastante caro. Em aplicações de longa duração, onde as filas
podem ser criadas uma vez no início de uma execução de muitas horas e reutilizadas durante todo o processo,
esse custo é amortizado. Em códigos de execução curta, como o código de demonstração
usado neste capítulo, esse custo pode superar o benefício do pipelining.
Duas soluções para isso são introduzir um loop de bloco simples no início do
código que pré-crie as filas assíncronas antes da secção temporizada, ou
usar uma operação de módulo para reutilizar o mesmo número menor de filas entre todos os
os blocos. Por exemplo, ao utilizar o número de bloco módulo 2 como o
 identificador assíncrono, apenas duas filas serão usadas e o custo de criação dessas filas será amortizado pela sua reutilização. Duas filas são geralmente
suficiente para ver um ganho de desempenho, uma vez que ainda permite que a computação e as
 atualizações se sobreponham, mas o desenvolvedor deve experimentar para encontrar o melhor valor
em uma determinada máquina.

Abaixo, vemos uma captura de ecrã que mostra os perfis antes e depois da aplicação destas
alterações no código em uma plataforma de GPU NVIDIA. Resultados semelhantes devem ser
possíveis em qualquer plataforma acelerada. Usando 16 blocos e duas filas 
assíncronas, como mostrado abaixo, foi observada uma melhoria de desempenho de aproximadamente 2X
na máquina de teste em relação ao desempenho sem pipelining.

![NVIDIA NSight Systems profiler timelines for the original mandelbrot code (Top) and the
pipelined code using 16 blocks over 2 asynchronous
queues (Bottom).](images/mandelbrot_async_nsight.png)

Programação multi-dispositivo
------------------------

Para sistemas que contêm mais de um acelerador, o OpenACC fornece uma API para fazer com que as operações aconteçam num dispositivo em particular. No caso de um sistema conter
aceleradores de diferentes tipos, a especificação também permite a consulta e
selecionar dispositivos de uma arquitetura específica.

### acc\_get\_num\_devices() ###
A rotina `acc_get_num_devices()` pode ser utilizada para consultar quantos dispositivos de
uma determinada arquitetura estão disponíveis no sistema. Ela aceita um parâmetro do tipo
tipo `acc_device_t` e retorna um número inteiro de dispositivos.

### acc\_get\_device\_num() and acc\_set\_device\_num() ###
As rotinas `acc_get_device_num()` consultam o
dispositivo atual que será utilizado de um determinado tipo e devolve o identificador
desse dispositivo.  A rotina `acc_set_device_num()` aceita dois parâmetros,
o número do dispositivo desejado e o tipo de dispositivo. Uma vez que o número do dispositivo tenha sido definido,
todas as operações serão enviadas para o dispositivo especificado até que um dispositivo
diferente seja especificado por uma chamada posterior a `acc_set_device_num()`.


### acc\_get\_device\_type() and acc\_set\_device\_type() ###
A rotina `acc_get_device_type()` não recebe parâmetros e retorna o tipo de dispositivo
do dispositivo padrão atual. A rotina `acc_set_device_type()` especifica
o tempo de execução o tipo de dispositivo que o tempo de execução deve utilizar para acelerar as operações,
mas permite que o tempo de execução escolha qual dispositivo desse tipo utilizar.

O OpenACC introduziu recentemente a diretiva `set`, que permite a
programação multi-dispositivo com menos dependência do uso da API OpenACC.
A diretiva `set` pode ser usada para definir o número do dispositivo e o tipo de dispositivo
que deve ser usado e é funcionalmente equivalente à diretiva
`acc_set_device_num()` da API. Para definir o número do dispositivo, use
`device_num`, e para definir o tipo utilize a cláusula `device_type`.

---

### Exemplo de programação multi-dispositivo ###
Como exemplo de programação multi-dispositivo, é possível alargar o exemplo de mandelbrot usado anteriormente para enviar diferentes blocos de trabalho para
diferentes aceleradores. Para que isso funcione, é necessário garantir
que sejam criadas cópias dos dados em cada dispositivo. Nós faremos isso
substituindo a região estruturada `data` no código por uma diretiva não estruturada `enter data` para cada dispositivo, utilizando a função `acc_set_device_num()` para
especificar o dispositivo para cada `enter data`. Para simplificar, nós alocaremos o array de imagens completo em cada dispositivo, embora apenas uma parte do array seja
necessária. Quando os requisitos de memória da aplicação forem grandes, será
necessário alocar apenas as partes pertinentes dos dados em cada acelerador.

Uma vez que os dados tenham sido criados em cada dispositivo, uma chamada a `acc_set_device_num()`
no loop de bloqueio, usando uma operação simples de módulo para selecionar qual dispositivo
deve receber cada bloco, enviará blocos para diferentes dispositivos.  

Por fim, é necessário introduzir um loop sobre os dispositivos para esperar que cada dispositivo
para completar. Como a diretiva `wait` é por dispositivo, o loop irá mais uma vez
utilizar `acc_set_device_num()` para selecionar um dispositivo para esperar, e então utilizar uma diretiva
`exit data` para desalocar a memória do dispositivo. O código final está abaixo.

~~~~ {.c .numberLines}
    // Allocate arrays on both devices
    for (int gpu=0; gpu < 2 ; gpu ++)
    {
      acc_set_device_num(gpu,acc_device_nvidia);
    #pragma acc enter data create(image[:bytes])
    }
   
    // Distribute blocks between devices
    for(int block=0; block < numblocks; block++)
    {
      int ystart = block * blocksize;
      int yend   = ystart + blocksize;
      acc_set_device_num(block%2,acc_device_nvidia);
    #pragma acc parallel loop async(block)
      for(int y=ystart;y<yend;y++) {
        for(int x=0;x<WIDTH;x++) {
          image[y*WIDTH+x]=mandelbrot(x,y);
        }
      }
    #pragma acc update self(image[ystart*WIDTH:WIDTH*blocksize]) async(block)
    }

    // Wait on each device to complete and then deallocate arrays
    for (int gpu=0; gpu < 2 ; gpu ++)
    {
      acc_set_device_num(gpu,acc_device_nvidia);
    #pragma acc wait
    #pragma acc exit data delete(image)
    }
~~~~

---

~~~~ {.fortran .numberLines}
    batch_size=WIDTH/num_batches
    do gpu=0,1
      call acc_set_device_num(gpu,acc_device_nvidia)
      !$acc enter data create(image)
    enddo
    do yp=0,NUM_BATCHES-1
      call acc_set_device_num(mod(yp,2),acc_device_nvidia)
      ystart = yp * batch_size + 1
      yend   = ystart + batch_size - 1
      !$acc parallel loop async(yp)
      do iy=ystart,yend
        do ix=1,HEIGHT
          image(ix,iy) = mandelbrot(ix-1,iy-1)
        enddo
      enddo
      !$acc update self(image(:,ystart:yend)) async(yp)
    enddo
    do gpu=0,1
      call acc_set_device_num(gpu,acc_device_nvidia)
      !$acc wait
      !$acc exit data delete(image)
    enddo
~~~~

Embora este exemplo aloque demais a memória do dispositivo ao colocar todo o array de imagens
no dispositivo, ele serve como um exemplo simples de como a rotina `acc_set_device_num()`
pode ser utilizada para operar em uma máquina com múltiplos dispositivos. Em
códigos de produção o desenvolvedor provavelmente vai querer particionar o trabalho de forma que
apenas as partes do array necessárias para um dispositivo específico estejam disponíveis lá.
Além disso, ao usar threads da CPU, pode ser possível enviar trabalho para os
dispositivos mais rapidamente e melhorar o desempenho geral. A Figura 7.3
apresenta uma captura de ecrã dos sistemas NVIDIA NSight que mostra o cálculo de Mandelbrot
dividido em duas GPUs NVIDIA.

![NVIDIA NSight Systems timeline for multi-device mandelbrot](images/multigpu_mandelbrot_timeline_nsight.png)
