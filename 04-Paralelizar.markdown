Paralelizar laços
=================
Agora que foram identificados os hotspots importantes na aplicação, o
programador deve acelerar progressivamente estes hotspots adicionando as directivas OpenACC
 para os laços importantes dentro dessas rotinas. Não há razão para
pensar no movimento de dados neste ponto do processo, o compilador OpenACC
 irá analisar os dados necessários na região identificada e
assegurar automaticamente que os dados estão disponíveis no acelerador. Ao focar
apenas sobre o paralelismo durante esta etapa, o programador pode mover mais computação para o dispositivo e assegurar que o programa ainda está
dando resultados corretos antes de optimizar a movimentação de dados no passo seguinte.
Durante esta etapa do processo é comum que o tempo total de execução da
aplicação aumente, mesmo que a execução dos laços individuais seja
mais rápido, utilizando o acelerador. Isto porque o compilador deve tomar uma atitude cautelosa
abordando a movimentação de dados, copiando frequentemente mais dados de e para o
acelerador do que é realmente necessário. Mesmo que o tempo total de execução
aumenta durante esta etapa, o desenvolvedor deve concentrar-se em expressar uma
quantidade significativa de paralelismo no código antes de passar à etapa seguinte
e a realização de um benefício das diretivas.

----

O OpenACC fornece duas abordagens diferentes para expor o paralelismo no código:
regiões `parallel` e `kernels`. Cada uma destas diretivas será detalhada em
as secções que se seguem.

A Construção da Kernels
---------------------
A construção `kernels` identifica uma região de código que pode conter
paralelismo, mas conta com as capacidades de paralelização automática do
compilador para analisar a região, identificar quais os laços que são seguros para paralelizar,
e depois acelerar esses laços. Desenvolvedores com pouco ou nenhuma
experiência em programação paralela, ou aqueles que trabalham em funções que contêm muitos laços
ninhos que possam ser paralelizados, achará a diretiva kernels um bom
ponto de partida para a aceleração do OpenACC. O código abaixo demonstra a utilização de
`kernels` tanto em C/C++ como em Fortran.

~~~~ {.c .numberLines}
    #pragma acc kernels
    {
      for (i=0; i<N; i++)
      {
        y[i] = 0.0f;
        x[i] = (float)(i+1);
      }
    
      for (i=0; i<N; i++)
      {
        y[i] = 2.0f * x[i] + y[i];
      }
    }
~~~~    

----

~~~~ {.fortran .numberLines}
    !$acc kernels
    do i=1,N
      y(i) = 0
      x(i) = i
    enddo
  
    do i=1,N
      y(i) = 2.0 * x(i) + y(i)
    enddo
    !$acc end kernels
~~~~    

Neste exemplo, o código está a inicializar duas matrizes e depois a efectuar um cálculo simples sobre elas. Note-se que identificamos um bloco de código, utilizando chaves encaracoladas em C e diretivas de início e fim em Fortran, que contém dois loops candidatos para aceleração. O compilador irá analisar estes loops para independência de dados e paralelizar ambos os loops gerando um kernel acelerador para cada um. Ao compilador é dada total liberdade para determinar a melhor forma de mapear o paralelismo disponível nestes loops para o hardware, o que significa que seremos capazes de utilizar este mesmo código independentemente do acelerador para o qual estamos a construir. O compilador utilizará o seu próprio conhecimento do acelerador alvo para escolher o melhor caminho para a aceleração. Uma advertência sobre a directiva kernels, no entanto, é que se o compilador não puder ter a certeza de que um laço é independente dos dados, não irá paralelizar o laço. Razões comuns para que um compilador possa identificar mal um laço como não-paralelo serão discutidas numa secção posterior.

A construção da Parallel
----------------------
A construção `parallel` identifica uma região de código que será paralela
através do OpenACC *gangs*. Por si só, uma região `parallel` é de utilidade limitada,
mas quando emparelhado com a directiva `loop` (discutida mais tarde com mais detalhe)
irá gerar uma versão paralela do laço para o acelerador.
Estas duas diretivas podem, e na maioria das vezes são, combinadas numa única directiva  `parallel
loop`. Ao colocar esta diretiva num laço, o programador afirma
que o laço afetado é seguro para paralelizar e permite que o compilador selecione
como agendar as iterações do laço no acelerador alvo. O código abaixo
demonstra a utilização da directiva combinada  `parallel loop` em ambos C/C++ e Fortran.

~~~~ {.c .numberLines}
    #pragma acc parallel loop
      for (i=0; i<N; i++)
      {
        y[i] = 0.0f;
        x[i] = (float)(i+1);
      }
    
    #pragma acc parallel loop
      for (i=0; i<N; i++)
      {
        y[i] = 2.0f * x[i] + y[i];
      }
~~~~

----

~~~~ {.fortran .numberLines}
    !$acc parallel loop
    do i=1,N
      y(i) = 0
      x(i) = i
    enddo
  
    !$acc parallel loop
    do i=1,N
      y(i) = 2.0 * x(i) + y(i)
    enddo
~~~~    

Note que, ao contrário da directiva `kernels`, cada laço tem de ser explicitamente
decorado com a diretiva `parallel loop`. Isto porque o construtor `parallel`
depende do programador para identificar o paralelismo no código
em vez de realizar a sua própria análise do compilador dos laços. Neste caso,
o programador está apenas a identificar a disponibilidade do paralelismo, mas ainda
deixando a decisão de como mapear esse paralelismo para o acelerador para
conhecimentos do compilador sobre o dispositivo. Esta é uma característica chave que
diferencia o OpenACC de outros modelos de programação semelhantes. O programador
identifica o paralelismo sem ditar ao compilador como explorar
esse paralelismo. Isto significa que o código OpenACC será portátil para dispositivos
para além do dispositivo sobre o qual o código está a ser desenvolvido, porque os detalhes
sobre como paralelizar o código são deixados para compilador em vez de
ser codificado na fonte. 

Diferenças entre o Parallel e Kernels
----------------------------------------
Um dos maiores pontos de confusão para os novos programadores do OpenACC é a razão pela qual
tem ambas as especificações diretivas `parallel` e `kernels`, que parecem
fazer a mesma coisa. Embora estejam intimamente relacionadas, há sutilezas
diferenças entre eles. A construção `kernels` dá ao compilador o máximo de
margem de manobra para paralelizar e optimizar o código como ele considera adequado para o alvo
acelerador, mas também depende muito da capacidade do compilador de
paralelizar automaticamente o código. Como resultado, o programador pode ver
diferenças no que os diferentes compiladores são capazes de fazer em paralelo e como o fazem. A diretiva `parallel loop` é uma asserção do programador
que é ao mesmo tempo seguro e desejável paralelizar o laço afetado. Isto
depende do programador para ter identificado corretamente o paralelismo no código
e remover qualquer coisa no código que possa não ser segura para paralelizar. Se o
programador afirma incorretamente que o laço pode ser paralelizado então
a aplicação resultante pode produzir resultados incorretos. 

Para colocar as coisas de outra forma: a construção `kernels` pode ser pensada como uma dica
para o compilador de onde deve procurar o paralelismo enquanto o `parallel`
é uma asserção para o compilador de onde existe paralelismo.

Uma coisa importante a notar sobre a construção do `kernels` é que o compilador
analisará o código e só fará o paralelismo quando tiver a certeza de que é seguro
para fazer.  Em alguns casos, o compilador pode não ter informação suficiente em
tempo de compilação para determinar se um laço é seguro para paralelizar, caso em que
não irá paralelizar o laço, mesmo que o programador consiga ver claramente que
o laço é seguramente paralelo. Por exemplo, no caso do código C/C++, em que
são representados como ponteiros, o compilador pode nem sempre ser
capaz de determinar que dois conjuntos não fazem referência à mesma memória, conhecida
como *pointer aliasing*. Se o compilador não puder saber que dois ponteiros não são
alias, não será capaz de paralisar um laço que acede a essas matrizes. 

***Boas Práticas:*** Os programadores de C devem utilizar a palavra-chave `restrict` (ou a palavra
`__restrict` decorador em C++) sempre que possível para informar o compilador que
os ponteiros não são parecidos, o que frequentemente dará ao compilador informação o suficiente
para depois paralelizar laços que de outra forma não teria. Além da palavra-chave `restrict`, declarando variáveis constantes utilizando a
palavra-chave `const` pode permitir que o compilador utilize uma read-only memory para essa
variável, se tal memória existir no acelerador. A utilização de `const` e
`restrict`  é uma boa prática de programação em geral, uma vez que dá ao compilador
informação adicional que pode ser utilizada quando se optimiza o código.

Os programadores de Fortran devem também notar que um compilador OpenACC irá paralelizar
sintaxe de matriz Fortran que está contida numa construção de `kernels`. Quando se utiliza `parallel`
em vez disso, será necessário introduzir explicitamente laços sobre os
elementos das matrizes.

Mais um benefício notável que a construção `kernels` proporciona é que se os dados
são movidos para o dispositivo para utilização em loops contidos na região, os dados irão
permanecer no dispositivo durante toda a extensão da região, ou até que seja necessário
novamente sobre o host dentro daquela região. Isto significa que se existirem  laços múltiplos acessando
os mesmos dados só serão copiados para o acelerador uma vez. Quando `parallel
loop` é utilizado em dois laços subsequentes que acedem aos mesmos dados que um compilador pode
ou não copiar os dados para a frente e para trás entre o host e o dispositivo entre
os dois laços. Nos exemplos mostrados na secção anterior, o compilador
gera movimento de dados implícito para ambos os laços paralelos, mas apenas
gera movimento de dados uma vez para a abordagem de `kernels`, o que pode resultar em
menos movimento de dados por padrão. Esta diferença será reexaminada no caso
estudado mais adiante neste capítulo.

For more information on the differences between the `kernels` and `parallel`
directives, please see [http://www.pgroup.com/lit/articles/insider/v4n2a1.htm].

---

Nesta altura, muitos programadores ficarão pensando qual diretiva
devem utilizar no seu código. Os programadores paralelos mais experientes, que podem ter
já identificado laços paralelos dentro do seu código, encontrarão provavelmente a
abordagem `parallel loop` mais desejável. Programadores com menos experiência em programação paralela ou cujo código contém um grande número de laços que necessitam
de análise pode achar a abordagem `kernels` muito mais simples, uma vez que coloca mais de
o encargo para o compilador. Ambas as abordagens têm vantagens,  os novos programadores OpenACC
devem determinar por si próprios qual a abordagem que melhor se adapta. Um programador pode até optar por utilizar `kernels` numa parte do código,
mas `paralell` na outra, se fizer sentido fazê-lo.

**Nota:** Para o restante do documento, a frase *parallel region* será
utilizado para descrever ou uma região `parallel` ou `kernels`. Ao referir-se à região de
construção `parallel`, será utilizada uma fonte terminal, como se mostra no presente
frase.

O Construtor Loop 
------------------
A construção "loop" dá ao compilador informações adicionais sobre todo
próximo laço no código fonte. A diretiva `loop` foi mostrada acima em
ligação com a diretiva `parallel`, embora também seja válida com
`kernels`. As cláusulas de laço vêm em duas formas: cláusulas de correção e cláusulas
para optimização. Este capítulo irá discutir apenas as duas cláusulas de correção
e um capítulo posterior irá discutir cláusulas de optimização.

### private ###
A cláusula private especifica que cada iteração de laço requer a sua própria cópia das variáveis listadas. Por exemplo, se cada laço contiver um pequeno
array chamado `tmp` que utiliza durante o seu cálculo, então esta variável deve
ser privado para cada iteração de laço, a fim de garantir resultados correctos. Se
o `tmp` não é declarado privado, depois que as threads executam diferentes iterações de laço
pode aceder a esta variável partilhada `tmp` de forma imprevisível, resultando numa
condição de `race condition` e resultados potencialmente incorrectos. Abaixo está a sintax para a
cláusula `private`.

    private(var1, var2, var3, ...)

Há alguns casos especiais que devem ser entendidos sobre variáveis escalares
dentro de laços. Em primeiro lugar, os iteradores de laço serão privados por padrão, pelo que
não precisam de ser listados como privados. Em segundo lugar, salvo especificação em contrário,
qualquer escalar acessada dentro de um laço paralelo será feito *primeiro privado*
por padrão, o que significa que será feita uma cópia privada da variável para cada laço de
iteração e será inicializada com o valor dessa escalar
ao entrar na região. Finalmente, quaisquer variáveis (escalares ou não) que sejam declaradas
dentro de um laço em C ou C++ será tornada privada para as iterações desse laço
por padrão.

Nota: A construção `parallel` também tem uma cláusula "privada" que irá tornar privadas
as variáveis listadas para cada gangue na região paralela. 

### reduction ###
A cláusula `reduction` funciona de forma semelhante à cláusula `private`, na medida em que
é gerada uma cópia privada da variável afetada para cada iteração de laço, mas
`reduction` vai um passo além para reduzir todas essas cópias privadas a um só
resultado final, que é devolvido da região. Por exemplo, o máximo de
todas as cópias privadas da variável podem ser necessárias. A
redução só pode ser especificada numa variável escalar e só comum, operações  especificadas
podem ser realizadas, tais como `+`, `*`, `min`, `max`, e várias
operações bitwise (ver a especificação OpenACC para uma lista completa). O formato da cláusula de redução é o seguinte, onde *operador* deve ser
substituído com a operação de interesse e *variável* deve ser substituído por
sendo a variável reduzida:

    reduction(operator:variable)

Um exemplo de utilização da cláusula `reduction` será apresentado no estudo de caso abaixo.

Diretiva Routine
-----------------
As chamadas de funções ou sub-rotinas dentro de laços paralelos podem ser problemáticas para
compiladores, uma vez que nem sempre é possível para o compilador ver todos os
laços de uma só vez. Os compiladores do OpenACC 1.0 foram forçados a colocar em linha todas as
rotinas chamadas dentro de regiões paralelas ou não paralelas, contendo
chamadas de rotina. OpenACC 2.0 introduziu a directiva `routine` para abordar
esta lacuna. A diretiva `routine` dá ao compilador a necessária
informação sobre a função ou sub-rotina e os laços que esta contém em ordem
para paralelizar a chamada região paralela. A diretiva de rotina deve ser acrescentada
a uma definição da função que informa o compilador do nível de paralelismo
utilizado no âmbito da rotina. Os *níveis de paralelismo* do OpenACC serão discutidos numa
seção posterior.

### C++ Class Functions ###
Quando se opera em classes C++, é frequentemente necessário chamar a classe
funções a partir de regiões paralelas. O exemplo abaixo mostra uma classe C++
`float3` que contém 3 valores de ponto flutuante e tem uma função `set` que é
utilizado para definir os valores dos seus membros `x`, `y`, e `z` para os de outro
exemplo de `float3`. Para que isto funcione a partir de uma região paralela,
a função `set` é declarada como uma rotina OpenACC utilizando a directiva `routine`. 
Uma vez que sabemos que será chamada por cada iteração de um laço paralelo
, é declarada uma rotina `seq` (ou *sequencial*).

~~~~ {.cpp .numberLines}
    class float3 {
       public:
     	float x,y,z;
    
       #pragma acc routine seq
       void set(const float3 *f) {
    	x=f->x;
    	y=f->y;
    	z=f->z;
       }
    };
~~~~

Atomic Operations
-----------------
Quando uma ou mais iterações de loop precisam de aceder a um elemento da memória
ao mesmo tempo podem ocorrer corridas de dados. Por exemplo, se uma iteração de laço for
modificando o valor contido numa variável e outra está a tentar ler a partir da mesma variável em paralelo, podem ocorrer resultados diferentes, dependendo de qual
a iteração ocorre primeiro. Nos programas seriais, os loops sequenciais asseguram que
a variável será modificada e lida numa ordem previsível, mas programas paralelizados não dão garantias de que uma determinada iteração de laço irá acontecer
antes de outro. Em casos simples, tais como encontrar uma soma, máxima, ou mínima
uma operação de redução garantirá a correção do valor. Para mais complexos
a diretiva `atomic` assegurará que nenhum das duas threads possa tentar executar a operação contida simultaneamente. O uso de atómica é por vezes uma
parte necessária da paralelização para assegurar a correção.

A diretiva `atomic` aceita uma de quatro cláusulas para declarar o tipo de
operação contida dentro da região. A operação de `read` assegura que não haja dois
iterações de laço serão lidas a partir da região ao mesmo tempo. A operação `write`
assegurará que não haja duas iterações de escrita na região na mesma altura. Uma operação de `update` é uma operação combinada de leitura e escrita. Finalmente, uma
A operação `capture` efetua uma atualização, mas guarda o valor calculado nessa
região a utilizar no código que se segue. Se nenhuma cláusula for dada, então uma atualização da
operação ocorrerá.

### Atomic Example ###

<!-- ![A histogram of number distribution.](images/histogram.png) -->

Um histograma é uma técnica comum para contar o número de vezes que os valores ocorrem
a partir de um conjunto de entradas de acordo com o seu valor. O exemplo de
código abaixo realiza iterações através de uma série de números inteiros de uma gama conhecida e
conta as ocorrências de cada número nesse intervalo. Uma vez que cada número na
gama pode ocorrer várias vezes, precisamos assegurar que cada elemento do
conjunto de histogramas é atualizado atomicamente. O código abaixo demonstra a utilização da
diretiva `atomic` para gerar um histograma.

~~~~ {.c .numberLines}
    #pragma acc parallel loop
    for(int i=0;i<HN;i++)
      h[i]=0;

    #pragma acc parallel loop
    for(int i=0;i<N;i++) {
      #pragma acc atomic update
      h[a[i]]+=1;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc kernels
    h(:) = 0
    !$acc end kernels
    !$acc parallel loop
    do i=1,N
      !$acc atomic
      h(a(i)) = h(a(i)) + 1
    enddo
    !$acc end parallel loop
~~~~

Note-se que as actualizações da matriz de histogramas `h` são realizadas atomicamente.
Porque estamos a aumentar o valor do elemento da matriz, uma operação de actualização
é usado para ler o valor, modificá-lo, e depois escrevê-lo de volta.

Case Study - Parallelize
------------------------
No último capítulo identificámos os dois ninhos de laço no âmbito da convergência
loop como as partes mais demoradas da nossa aplicação.  Além disso, nós
olhamos para os laços e fomos capazes de determinar que o laço de convergência exterior
não é paralelo, mas os dois loops aninhados no seu interior são seguros para serem paralelos. Neste capítulo vamos acelerar esses ninhos de laço com o OpenACC usando as
diretivas discutidas anteriormente neste capítulo. Para enfatizar ainda mais as semelhanças e diferenças entre as diretivas `parallel` e `kernels`, nós iremos acelerar os laços usando ambos e discutir as diferenças.

### Parallel Loop ###
Identificamos anteriormente o paralelismo disponível no nosso código, agora vamos utilizar
a diretiva `parallel loop` para acelerar os laços que identificámos. Sabendo que os dois conjuntos de laços duplamente aninhados são paralelos, basta acrescentar uma diretiva `parallel loop` acima de cada um deles. Isto irá informar o compilador
que o exterior dos dois laços é seguramente paralelo. Alguns compiladores irão
além disso, analisaram o laço interno e determinaram que ele também é paralelo, mas
para termos a certeza de que também acrescentaremos uma diretiva `loop` em torno dos loops internos. 

Há mais uma subtileza para acelerar os loops neste exemplo: nós somos
tentando calcular o valor máximo para a variável "erro". Como
discutido acima, isto é considerado uma *redução* uma vez que estamos a reduzir de
todos os valores possíveis para o "erro" até ao máximo único. Isto significa
que é necessário indicar uma redução no primeiro ninho de laço (o
que calcula o "erro"). 

***Boas Práticas:*** Alguns compiladores detectarão a redução no "erro" e irão
inserir implicitamente a cláusula de "redução", mas para a máxima portabilidade o programador deve sempre indicar reduções no código.

Neste ponto, o código parece-se com os exemplos abaixo.

~~~~ {.c .numberLines startFrom="52"}
    while ( error > tol && iter < iter_max )
    {
      error = 0.0;
      
      #pragma acc parallel loop reduction(max:error) 
      for( int j = 1; j < n-1; j++)
      {
        #pragma acc loop reduction(max:error)
        for( int i = 1; i < m-1; i++ )
        {
          A[j][i] = 0.25 * ( Anew[j][i+1] + Anew[j][i-1]
                           + Anew[j-1][i] + Anew[j+1][i]);
          error = fmax( error, fabs(A[j][i] - Anew[j][i]));
        }
      }

      #pragma acc parallel loop
      for( int j = 1; j < n-1; j++)
      {
        #pragma acc loop
        for( int i = 1; i < m-1; i++ )
        {
          A[j][i] = Anew[j][i];
        }
      }
      
      if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
      
      iter++;
    }
~~~~    
      
----

~~~~ {.fortran .numberLines startFrom="52"}
    do while ( error .gt. tol .and. iter .lt. iter_max )
      error=0.0_fp_kind
        
      !$acc parallel loop reduction(max:error)
      do j=1,m-2
        !$acc loop reduction(max:error)
        do i=1,n-2
          A(i,j) = 0.25_fp_kind * ( Anew(i+1,j  ) + Anew(i-1,j  ) + &
                                    Anew(i  ,j-1) + Anew(i  ,j+1) )
          error = max( error, abs(A(i,j) - Anew(i,j)) )
        end do
      end do

      !$acc parallel loop
      do j=1,m-2
        !$acc loop
        do i=1,n-2
          A(i,j) = Anew(i,j)
        end do
      end do

      if(mod(iter,100).eq.0 ) write(*,'(i5,f10.6)'), iter, error
      iter = iter + 1
    end do
~~~~    

***Boas Práticas:*** A maioria dos compiladores do OpenACC aceitará apenas a diretiva `parallel loop` sobre os laços `j` e detectam por si próprios que o laço `i`
também pode ser paralelizado sem necessidade das diretivas "loop" sobre os laços "i".
Colocando uma diretiva `loop` em cada laço que pode ser
em paralelo, o programador assegura que o compilador compreenderá que o
o laço é seguro para o paralelismo. Quando utilizado dentro de uma região `parallel`, a diretiva `loop` assegura que as iterações do laço são independentes umas das outras e  é seguro o paralelismo, e devem ser utilizados para fornecer ao compilador tanto quanto possível
informações sobre os laços.

A construção do código acima usando o compilador NVHPC produz o seguinte feedback do compilador (mostrado para C, mas a saída do Fortran é semelhante).

    $ nvc -acc -Minfo=accel laplace2d-parallel.c
    main:
         56, Generating Tesla code
             57, #pragma acc loop gang /* blockIdx.x */
                 Generating reduction(max:error)
             59, #pragma acc loop vector(128) /* threadIdx.x */
         56, Generating implicit copyin(A[:][:]) [if not already present]
             Generating implicit copy(error) [if not already present]
             Generating implicit copyout(Anew[1:4094][1:4094]) [if not already present]
         59, Loop is parallelizable
         67, Generating Tesla code
             68, #pragma acc loop gang /* blockIdx.x */
             70, #pragma acc loop vector(128) /* threadIdx.x */
         67, Generating implicit copyin(Anew[1:4094][1:4094]) [if not already present]
             Generating implicit copyout(A[1:4094][1:4094]) [if not already present]
         70, Loop is parallelizable


A análise do feedback do compilador dá ao programador a capacidade de assegurar que
o compilador está a produzir os resultados esperados ou a corrigir quaisquer problemas.
Na saída acima vemos que foram gerados kernels de acelerador para os dois
laços que foram identificados (nas linhas 58 e 71, no código fonte compilado)
e que o compilador gerou automaticamente o movimento de dados, que será
discutido com mais detalhe no próximo capítulo.

Outras cláusulas da directiva `loop` que podem beneficiar ainda mais o desempenho
do código resultante será discutido num capítulo posterior.   

<!---(***TODO: Link to later chapter when done.***)--->

### Kernels ###
A utilização da diretiva `kernels` para acelerar os laços que identificamos requer a
inserção de apenas uma diretiva no código e permitir que o compilador execute
a análise paralela. Acrescentar uma construção de `kernels` em torno dos dois ninhos de laços computacionais resulta no seguinte código.

~~~~ {.c .numberLines startFrom="51"}
    while ( error > tol && iter < iter_max )
    {
      error = 0.0;
      
      #pragma acc kernels 
      {
        for( int j = 1; j < n-1; j++)
        {
          for( int i = 1; i < m-1; i++ )
          {
            A[j][i] = 0.25 * ( Anew[j][i+1] + Anew[j][i-1]
                             + Anew[j-1][i] + Anew[j+1][i]);
            error = fmax( error, fabs(A[j][i] - Anew[j][i]));
          }
        }
      
        for( int j = 1; j < n-1; j++)
        {
          for( int i = 1; i < m-1; i++ )
          {
            A[j][i] = Anew[j][i];
          }
        }
      }        
      
      if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
      
      iter++;
    }
~~~~    

----

~~~~ {.fortran .numberLines startFrom="51"}
    do while ( error .gt. tol .and. iter .lt. iter_max )
      error=0.0_fp_kind
        
      !$acc kernels 
      do j=1,m-2
        do i=1,n-2
          A(i,j) = 0.25_fp_kind * ( Anew(i+1,j  ) + Anew(i-1,j  ) + &
                                    Anew(i  ,j-1) + Anew(i  ,j+1) )
          error = max( error, abs(A(i,j) - Anew(i,j)) )
        end do
      end do

      do j=1,m-2
        do i=1,n-2
          A(i,j) = Anew(i,j)
        end do
      end do
      !$acc end kernels
        
      if(mod(iter,100).eq.0 ) write(*,'(i5,f10.6)'), iter, error
      iter = iter + 1
    end do
~~~~    
    
O código acima demonstra o poder que a construção `kernels`
fornece, uma vez que o compilador irá analisar o código e identificar ambos os ninhos de laço
como paralelo e descobrirá automaticamente a redução sobre a variável `error` sem intervenção do programador. Um compilador OpenACC irá provavelmente
descobrir não só que os laços exteriores são paralelos, mas também os laços interiores,
resultando em mais paralelismo disponível com menos diretivas do que a
abordagem `parallel loop`. Se o programador tivesse colocado a construção de `kernels` em torno do
o ciclo de convergência, que já determinamos não é paralelo, o
compilador provavelmente não teria encontrado qualquer paralelismo disponível. Mesmo com a
diretiva `kernels`, é necessário que o programador faça alguma quantidade de
análise para determinar onde o paralelismo pode ser encontrado.

Olhar para a saída do compilador aponta para algumas diferenças mais sutis
entre as duas abordagens.

    $ nvc -acc -Minfo=accel laplace2d-kernels.c
    main:
         56, Generating implicit copyin(A[:][:]) [if not already present]
             Generating implicit copyout(Anew[1:4094][1:4094],A[1:4094][1:4094]) [if not already present]
         58, Loop is parallelizable
         60, Loop is parallelizable
             Generating Tesla code
             58, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
             60, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
             64, Generating implicit reduction(max:error)
         68, Loop is parallelizable
         70, Loop is parallelizable
             Generating Tesla code
             68, #pragma acc loop gang, vector(4) /* blockIdx.y threadIdx.y */
             70, #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */

A primeira coisa a notar na saída acima é que o compilador identificou corretamente
os quatro laços como sendo paralelizáveis e gerou kernels a partir
desses laços. Nota-se também que o compilador apenas gerou diretivas de movimento de dados implícitos
 na linha 54 (o início da região `kernels`), em vez de
do que no início de cada `parallel loop`. Isto significa que o resultado
o código deve efectuar menos cópias entre o host e a memória do dispositivo nesta versão
do que a versão da secção anterior. Uma diferença mais subtil entre
o resultado é que o compilador escolheu um esquema de decomposição de laço diferente (como
é evidente pelas directivas implícitas `acc loop` na saída do compilador) do que
o laço paralelo porque o `kernels` lhe permitiu fazê-lo. Mais pormenores sobre como
interpretar este feedback de decomposição e como mudar o comportamento será
discutido num capítulo posterior.

---

Neste momento, exprimimos todo o paralelismo no código de exemplo e
o compilador paralelizou-o para um dispositivo acelerador. Analisando
o desempenho deste código podemos chegar a resultados surpreendentes em alguns aceleradores. 
Os resultados abaixo demonstram o desempenho deste código em 1 - 16
threads de CPU num CPU Threadripper AMD e num NVIDIA Volta V100
GPU usando ambas as implementações acima. O eixo *y* da figura 3.1 é a execução em
tempo em segundos, por isso menor é melhor. Para as duas versões OpenACC, a barra é
dividida pelo tempo de transferência de dados entre o host e o dispositivo e o tempo de execução
no dispositivo.

![Jacobi Iteration Performance - Step 1](images/jacobi_step1_graph.png)

O desempenho desta melhora à medida que mais threads de CPU são adicionadas ao cálculo,
no entanto, uma vez que o código está ligado à memória, o benefício do desempenho da adição
das threads adicionais diminuem rapidamente. Além disso, as versões OpenACC têm um mau desempenho
em comparação com a CPU
linha de base. Tanto as versões OpenACC `kernels' e `parallel loop' atuam
pior do que a linha de base da CPU em série. É também claro que a versão `parallel loop`
gasta significativamente mais tempo na transferência de dados do que a versão `kernels`.
É necessária uma análise mais aprofundada do desempenho para
identificar a fonte desta desaceleração. Esta análise já foi aplicada ao gráfico acima, que decompõe o tempo gasto
computando a solução e copiando os dados de e para o acelerador.

Há uma variedade de ferramentas disponíveis para a realização desta análise, mas desde que este
estudo de caso foi compilado para uma GPU NVIDIA, NVIDIA Nsight Systems será
utilizado para compreender a peformance da aplicação. A captura de ecrã na figura 3.2
mostra o perfil do Nsight Systems para ***2*** iterações do laço de convergência na
versão `parallel loop` do código.

![Screenshot of NVIDIA Nsight Systems Profile on 2 steps of the Jacobi Iteration
showing a high amount of data transfer compared to
computation.](images/ch3_profile.png) 

Uma vez que a máquina de teste tem dois espaços de memória distintos, um para a CPU e outro para a
para a GPU, é necessário copiar os dados entre as duas memórias. Nesta captura de ecrã, a ferramenta representa transferências de dados utilizando as caixas coloridas bronzeadas nas
duas linhas *MemCpy* e o tempo de cálculo nas caixas verdes e roxas nas
linhas abaixo *Compute*. Deve ser óbvio, a partir da linha temporal apresentada, que
significativamente mais tempo está a ser gasto a copiar dados de e para o
acelerador antes e depois de cada kernel computado do que realmente computar sobre o
dispositivo. Na realidade, a maior parte do tempo é gasto em cópias de memória ou
os overhead incorrido pelo agendamento de memória de tempo de execução. No próximo
capítulo vamos corrigir esta ineficiência, mas primeiro, porque é que a versão `kernels` supera a versão `parallel loop`? 

Quando um compilador OpenACC paraleliza uma região de código, deve analisar os dados
que são necessários dentro dessa região e copiá-los de e para o acelerador, se
necessário. Esta análise é feita a um nível por região e, tipicamente
padrão para copiar matrizes utilizadas no acelerador tanto de e para o dispositivo
no início e no fim da região, respectivamente. Desde o `parallel loop`
tem duas regiões de computação, por oposição a apenas uma na versão `kernels`, os dados são copiados para a frente e para trás entre as duas regiões. Como resultado,
os tempos de cópia e de sobrecarga são cerca do dobro dos tempos da região `kernels`,
embora os tempos computacionais do kernel sejam aproximadamente os mesmos.
