Optimizar loops
==============
Uma vez expressa a localidade dos dados, os programadores podem querer afinar mais o
código para o hardware de interesse. É importante entender que quanto mais os
loops são ajustados para um tipo particular de hardware, menos portátil
o código se torna para outras arquiteturas. Se você está geralmente executando em um
acelerador em particular, no entanto, pode haver alguns ganhos a serem obtidos ajustando
como os loops são mapeados para o hardware subjacente. 

É tentador começar a ajustar os loops antes que toda a localidade de dados tenha sido
expressa no código. No entanto, como as cópias de dados são frequentemente o limitador do
desempenho da aplicação na atual geração de aceleradores, o impacto no desempenho do ajuste de um loop específico pode ser muito difícil de medir
até que a localidade dos dados tenha sido optimizada. Por esse motivo, a melhor prática é
esperar para otimizar determinados loops até que toda a localidade de dados tenha sido
expressa no código, reduzindo o tempo de transferência de dados ao mínimo.

Ordenação eficiente de loops
-----------------------
Antes de alterar a forma como o OpenACC mapeia os loops no hardware de interesse, o
desenvolvedor deve examinar os loops importantes para garantir que os arrays de dados
estão sendo acessados de maneira eficiente. A maioria dos hardwares modernos, seja uma CPU com
grandes caches e operações SIMD ou uma GPU com acessos de memória coalescidos e
operações SIMT, favorece o acesso a arrays de uma maneira *stride 1*.  Ou seja
que cada iteração de loop acessa endereços de memória consecutivos.
Isto é conseguido assegurando que o laço mais interno de um ninho de laços itera na
dimensão da matriz que varia mais rapidamente e cada laço sucessivo acede à
dimensão seguinte de variação mais rápida. A organização dos loops desta forma crescente
melhorará frequentemente a eficiência da cache e melhorará a vetorização na maioria das
arquiteturas.  

OpenACC's 3 Níveis de Paralelismo
---------------------------------
O OpenACC define três níveis de paralelismo: *gang*, *worker* e *vector*.
Adicionalmente, a execução pode ser marcada como sendo sequencial (*seq*). O paralelismo vectorial
tem a granularidade mais fina, com uma instrução individual
operando em vários pedaços de dados (muito parecido com o paralelismo *SIMD* em uma
CPU moderna ou o paralelismo *SIMT* numa GPU moderna). As operações vectoriais são executadas
com um determinado *comprimento de vector*, indicando quantos elementos de dados podem ser
operados com a mesma instrução. O paralelismo de gang é um paralelismo
de granulação grossa, em que as gangs trabalham independentemente uns dos outros e podem não
sincronizar. O paralelismo de worker situa-se entre os níveis de vector e de uma gang. Uma gang
consiste em 1 ou mais workers, cada um dos quais opera num vector de algum
comprimento.  Dentro de uma gang, o modelo OpenACC expõe uma memória *cache*, que pode ser
utilizada por todos os workers e vectores dentro do gang, e é legal sincronizar
dentro de uma gang, embora o OpenACC não exponha a sincronização ao utilizador.
Usando esses três níveis de paralelismo, mais o seqüencial, um programador pode mapear
o paralelismo no código para qualquer dispositivo. O OpenACC não requer que o
programador faça esse mapeamento explicitamente, no entanto. Se o programador optar por
não mapear explicitamente os loops para o dispositivo de interesse, o compilador
implicitamente fará este mapeamento usando o que ele sabe sobre o dispositivo de destino.
Isto torna o OpenACC altamente portátil, uma vez que o mesmo código pode ser mapeado para qualquer
número de dispositivos de destino. Quanto mais explícito for o mapeamento do paralelismo adicionado pelo
programador ao código, menos portátil ele torna o código para
outras arquiteturas.

![OpenACC's Three Levels of Parallelism](images/levels_of_parallelism.png)

### Understanding OpenACC's Three Levels of Parallelism

Os termos _gang_, _worker_ e _vector_ são estranhos à maioria dos programadores, pelo que
o significado destes três níveis de paralelismo é frequentemente perdido por novos programadores
programadores OpenACC. Aqui está um exemplo prático para ajudar a entender esses
três níveis. Imagine que precisa de pintar o seu apartamento. Uma pessoa com
um rolo e um balde de tinta pode pintar um pequeno
pequeno apartamento em poucas horas, talvez num dia. Para um apartamento pequeno, um pintor é
provavelmente suficiente para completar o trabalho, mas e se eu precisar pintar
todos os apartamentos de um grande edifício de vários andares. Nesse caso, é uma tarefa bastante
tarefa bastante assustadora para uma só pessoa. Há alguns truques que este pintor
pode tentar para trabalhar mais rapidamente. Uma opção é trabalhar mais depressa, movendo
o rolo pela parede o mais rápido que o seu braço conseguir. Há um limite prático, no entanto, para a velocidade a que um ser humano pode efetivamente usar um
rolo de pintura. Outra opção é utilizar um rolo de pintura maior. Talvez o nosso pintor
tenha começado com um rolo de pintura de 4 polegadas, por isso, se o atualizar para um rolo de 8 polegadas,
pode cobrir o dobro do espaço da parede no mesmo período de tempo. Porquê parar
aí? Vamos comprar um rolo de pintura de 32 polegadas para cobrir ainda mais paredes
por pincelada! Agora vamos começar a deparar-nos com problemas diferentes. Por exemplo, o braço do pintor provavelmente não se pode mover tão depressa com um rolo de 32 polegadas como um de 8 polegadas, pelo que não há garantia de que este seja efetivamente mais rápido.
Além disso, os rolos mais largos podem resultar em momentos embaraçosos quando o pintor tem de
pintar por cima de um lugar que já pintou para que o rolo caiba ou
o rolo mais largo pode demorar mais tempo a encher-se de tinta. Em qualquer dos casos,
há um limite claro para a rapidez com que um único pintor pode fazer o trabalho, por isso
vamos convidar mais pintores.

Agora, suponhamos que tenho 4 pintores a trabalhar na obra. Se forem dadas áreas independentes para pintar, o trabalho deve ser feito quase 4 vezes mais rápido, mas ao custo de
de obter 4 vezes mais rolos, tinteiros e latas de tinta. Este é
provavelmente um pequeno preço a pagar para que o trabalho seja feito quase 4 vezes mais rápido.
No entanto, grandes trabalhos requerem grandes equipes, por isso vamos aumentar o número de
pintores novamente para 16. Se cada pintor puder trabalhar de forma independente, o tempo
para completar a pintura provavelmente diminuirá mais 4 vezes,
mas agora pode haver outras ineficiências. Por exemplo, é provavelmente
mais barato comprar baldes grandes de tinta, em vez de latas de tinta pequenas, por isso
vamos guardar esses baldes num local central onde todos podem aceder
a eles. Agora, se um pintor precisar encher o seu balde, tem de andar para ir buscar a tinta, o que diminui o tempo que estão a pintar. Aqui está uma ideia,
vamos organizar os nossos 16 pintores em 4 grupos de 4 pintores, cada um dos quais tem
o seu próprio balde para partilhar. Agora, desde que os pintores de cada grupo estejam
a trabalhar em trabalhos perto do resto da equipe, a caminhada para obter mais tinta é muito
mais curta, mas as equipes continuam a ser livres para trabalhar de forma completamente independente uns dos outros.

Nesta analogia, existem 3 níveis de paralelismo, tal como no OpenACC. O nível
nível mais refinado pode não ser completamente óbvio, mas é o tamanho do
rolo. A largura do rolo dita a quantidade de parede que o pintor pode pintar
com cada pincelada. Os rolos mais largos significam mais paredes por pincelada, até um certo limite.
Em seguida, há pintores paralelos dentro de cada equipe. Estes pintores podem trabalhar independentemente uns dos outros, mas ocasionalmente precisam de aceder ao seu
balde de tinta partilhado ou coordenar o próximo trabalho a fazer.
Finalmente, temos as nossas equipes, que podem trabalhar de forma completamente independente umas das outras e podem até trabalhar em horários diferentes (por exemplo, turno do dia e o turno da noite), representando o paralelismo mais grosseiro da nossa hierarquia.

No OpenACC, os _gangs_ são como as equipes de trabalho, são completamente independentes uns dos outros e podem funcionar em paralelo ou mesmo em alturas diferentes.
Os _Workers_ são os pintores individuais, podem funcionar sozinhos mas podem
podem também partilhar recursos com outros _workers_ do mesmo _gang_. Finalmente, o
rolo de pintura representa o _vector_ em que a largura do rolo representa
o comprimento do _vector_. Os _Workers_ executam a mesma instrução em vários
elementos de dados utilizando operações de _vector_. Assim, os _gangs_ são compostos por pelo menos
um _worker_, que opera sobre um _vector_ de dados.

Mapeando Paralelismo para o Hardware
-----------------------------------
Com alguma compreensão de como o hardware do acelerador subjacente funciona, é possível
informar ao compilador como ele deve mapear as iterações do loop em
paralelismo no hardware. Vale a pena reafirmar que quanto mais detalhes o
compilador recebe sobre como mapear o paralelismo num determinado
acelerador específico, menor será a portabilidade de desempenho do código. Por exemplo, 
definir um comprimento de vetor fixo pode melhorar o desempenho em um processador e prejudicar o desempenho noutro ou fixar o número de grupos usados para executar
um ciclo pode resultar na limitação do desempenho em processadores com um maior 
grau de paralelismo.

Como discutido anteriormente neste guia, a diretiva `loop` tem a intenção de dar ao compilador
informações adicionais sobre o próximo loop no código. Além das cláusulas mostradas anteriormente, que tinham a intenção de garantir a correção, as cláusulas abaixo informam ao compilador qual nível de paralelismo deve ser usado
para o laço dado.

* Gang clause - dividir o loop entre as gangs
* Worker clause - dividir o loop entre os workers
* Vector clause - vetorizar o loop
* Seq clause - não particiona o loop, mas executá-o sequencialmente

Estas diretivas podem também ser combinadas num determinado loop. Por exemplo, um loop `gang vector` seria particionado em gangs, cada uma com 1
trabalhador implicitamente, e então vetorizado. A especificação OpenACC impõe que
o laço mais externo deve ser um laço gang, o laço paralelo mais interno deve ser
um loop vectorial, e um loop worker pode aparecer no meio. Um loop seqüencial pode
aparecer em qualquer nível.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang
    for ( i=0; i<N; i++)
      #pragma acc loop vector
      for ( j=0; j<M; j++)
        ;
~~~~

--

~~~~ {.fortran .numberLines}
    !$acc parallel loop gang
    do j=1,M
      !$acc loop vector
      do i=1,N
~~~~

Informar ao compilador onde particionar os loops é apenas uma parte da
optimização dos loops. O programador pode adicionalmente informar ao compilador o
número específico de gangs, workers, ou o comprimento do vetor a ser usado para os loops.
Este mapeamento específico é obtido de forma ligeiramente diferente quando se utiliza a diretiva `kernels`
ou a diretiva `parallel`. No caso da diretiva `kernels`,
as cláusulas `gang`, `worker` e `vector` aceitam um parâmetro inteiro que
opcionalmente informará ao compilador como particionar aquele nível de paralelismo.
Por exemplo, `vector(128)` informa ao compilador para utilizar um vetor com comprimento de 128
para o loop.  

~~~~ {.c .numberLines}
    #pragma acc kernels
    {
    #pragma acc loop gang
    for ( i=0; i<N; i++)
      #pragma acc loop vector(128)
      for ( j=0; j<M; j++)
        ;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc kernels
    !$acc loop gang
    do j=1,M
      !$acc loop vector(128)
      do i=1,N

    !$acc end kernels
~~~~

Ao utilizar a diretiva `parallel`, a informação é apresentada
na própria diretiva `parallel`, ao invés de em cada loop individual, na
forma das cláusulas `num_gangs`, `num_workers`, e `vector_length` da diretiva `parallel`.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang vector_length(128)
    for ( i=0; i<N; i++)
      #pragma acc loop vector
      for ( j=0; j<M; j++)
        ;
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop gang vector_length(128)
    do j=1,M
      !$acc loop vector(128)
      do i=1,N
~~~~

Uma vez que estes mapeamentos irão variar entre diferentes aceleradores, a directiva `loop`
aceita uma cláusula `device_type`, que informará ao compilador que
estas cláusulas só se aplicam a um determinado tipo de dispositivo. Cláusulas após uma cláusula
`device_type` até o próximo `device_type` ou até o final da diretiva
aplicar-se-ão apenas ao dispositivo especificado. Cláusulas que aparecem antes de
todas as cláusulas `device_type` são consideradas valores padrão, que serão usados se
não forem substituídas por uma cláusula posterior. Por exemplo, o código abaixo
especifica que um comprimento de vetor de 128 deve ser usado em dispositivos do tipo
`acc_device_nvidia` ou um comprimento de vetor de 256 deve ser usado em dispositivos do
tipo `acc_device_radeon`. O compilador escolherá um comprimento de vetor padrão para
todos os outros tipos de dispositivos.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang vector \
                device_type(acc_device_nvidia) vector_length(128) \
                device_type(acc_device_radeon) vector_length(256)
    for (i=0; i<N; i++)
    {
      y[i] = 2.0f * x[i] + y[i];
    }
~~~~

Cláusula Collapse
---------------
Quando um código contém loops fortemente aninhados, é frequentemente benéfico
*colapsar* estes loops num único loop. O colapso de loops significa que dois loops
de contagem de iterações N e M, respectivamente, serão automaticamente transformados num único
com uma contagem de viagens de N vezes M. Ao colapsar dois ou mais loops paralelos num
num único ciclo, o compilador tem uma maior quantidade de paralelismo para usar quando
mapear o código para o dispositivo. Em arquiteturas altamente paralelas, como as GPUs,
isso pode resultar em melhor desempenho. Além disso, se um loop não tiver paralelismo
suficiente para o hardware por si só, colapsá-lo com outro loop
multiplica o paralelismo disponível. Isso é especialmente benéfico em
loops vetoriais, já que alguns tipos de hardware exigirão comprimentos de vetor mais longos para
vetoriais mais longos do que outros para obter alto desempenho. O colapso de loops de grupo também pode ser benéfico
se permitir a geração de um número maior de grupos para processadores altamente paralelos.
O código abaixo demonstra como usar a diretiva de collapse.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang collapse(2)
    for(ie = 0; ie < nelemd; ie++) {
      for(q = 0; q < qsize; q++) {
        #pragma acc loop vector collapse(3)
        for(k = 0; k < nlev; k++) {
          for(j = 0; j < np; j++) {
            for(i = 0; i < np; i++) {
              qtmp = elem[ie].state.qdp[i][j][k][q][n0_qdp];
              vs1tmp = vstar[i][j][k][0][ie] * elem[ie].metdet[i][j] * qtmp;
              vs2tmp = vstar[i][j][k][1][ie] * elem[ie].metdet[i]]j] * qtmp;
              gv[i][j][k][0] = (dinv[i][j][0][0][ie] * vs1tmp + dinv[i][j][0][1][ie] * vs2tmp);
              gv[i][j][k][1] = (dinv[i][j][1][0][ie] * vs1tmp + dinv[i][j][1][1][ie] * vs2tmp);
            }
          }
        }
      }
    }
~~~~

---

~~~~ {.fortran .numberLines}    
    ! $acc parallel loop gang collapse (2)
    do ie = 1 , nelemd
      do q = 1 , qsize
        ! $acc loop vector collapse (3)
        do k = 1 , nlev
          do j = 1 , np
            do i = 1 , np
              qtmp = elem (ie )% state % qdp (i,j,k,q, n0_qdp )
              vs1tmp = vstar (i,j,k ,1, ie) * elem (ie )% metdet (i,j) * qtmp
              vs2tmp = vstar (i,j,k ,2, ie) * elem (ie )% metdet (i,j) * qtmp
              gv(i,j,k ,1) = ( dinv (i,j ,1 ,1 , ie )* vs1tmp + dinv (i,j ,1 ,2, ie )* vs2tmp )
              gv(i,j,k ,2) = ( dinv (i,j ,2 ,1 , ie )* vs1tmp + dinv (i,j ,2 ,2, ie )* vs2tmp )
            enddo
          enddo
        enddo
      enddo
    enddo
~~~~

O código acima é um excerto de uma aplicação real em que o colapso de loops
ampliou o paralelismo disponível para ser explorado. Na linha 1, os dois
loops mais externos são colapsados juntos para possibilitar a geração de *gangs*
através das iterações de ambos os loops, tornando assim o número possível de gangs
`nelemd` x `qsize` em vez de apenas `nelemd`. O colapso na linha 4 colapsa
3 pequenos loops para aumentar o possível *comprimento do vector*, uma vez que nenhum dos loops
itera por iterações suficientes para criar um comprimento de vetor razoável no
acelerador alvo. O quanto esta optimização irá acelerar o código irá variar
de acordo com a aplicação e o acelerador de destino, mas não é incomum
ver grandes aumentos de velocidade ao usar o colapso em ninhos de loops.

Paralelismo de Routine
-------------------
Um capítulo anterior introduziu a directiva `routine` para chamar funções e
subrotinas a partir de regiões paralelas OpenACC. Naquele capítulo foi assumido que
a rotina seria chamada a partir de cada iteração do laço, portanto exigindo uma diretiva `routine seq`. Em alguns casos, a própria rotina pode conter
paralelismo que deve ser mapeado para o dispositivo. Nestes casos, a diretiva `routine`
pode ter uma cláusula `gang`, `worker`, ou `vector` ao invés de `seq` para
informar ao compilador que a rotina conterá o nível especificado de
paralelismo. Isto pode ser pensado como _reserving_ um nível particular de 
paralelismo para os loops naquela rotina. Isto é para que quando o compilador
encontrar o local de chamada da rotina afetada, ele saberá como
pode paralelizar o código para usar a rotina. É importante notar que 
se uma `rotina acc` chama outra rotina, essa rotina também deve ter uma diretiva `acc routine`. No momento, a especificação do OpenACC não permite especificar vários níveis possíveis de paralelismo em uma única rotina.

Case Study - Optimize Loops
---------------------------
Este estudo de caso centrar-se-á num algoritmo diferente dos capítulos anteriores.
Quando um compilador tem informações suficientes sobre loops para tomar decisões informadas, é frequentemente difícil melhorar o desempenho de um determinado laço paralelo em mais do que alguns por cento. Em alguns casos, o código não possui as
informações necessárias para que o compilador tome decisões de optimização informadas.
Nesses casos, muitas vezes é possível para um desenvolvedor otimizar significativamente os loops paralelos, informando ao compilador como decompor e distribuir
os loops para o hardware.

O código utilizado nesta secção implementa uma operação de produto esparso de matriz e vector (SpMV)
esparsa. Isto significa que uma matriz e um vector serão multiplicados em conjunto,
mas a matriz tem muito poucos elementos que não são zero (é *esparsa*),
o que significa que o cálculo destes valores é desnecessário. A matriz é armazenada numa
formato Compress Sparse Row (CSR). No CSR, a matriz esparsa, que pode conter um
número significativo de células cujo valor é zero, desperdiçando assim uma
quantidade significativa de memória, é armazenada utilizando três matrizes mais pequenas: uma contendo os valores
valores não nulos da matriz, uma segunda que descreve onde numa dada linha
e uma terceira que descreve as colunas em que os dados residiriam. O código para este exercício está abaixo.

~~~~ {.c .numberLines}
    #pragma acc parallel loop
    for(int i=0;i<num_rows;i++) {
      double sum=0;
      int row_start=row_offsets[i];
      int row_end=row_offsets[i+1];
      #pragma acc loop reduction(+:sum)
      for(int j=row_start;j<row_end;j++) {
        unsigned int Acol=cols[j];
        double Acoef=Acoefs[j];
        double xcoef=xcoefs[Acol];
        sum+=Acoef*xcoef;
      }
      ycoefs[i]=sum;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop reduction(+:tmpsum)
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo
~~~~

Uma coisa importante a notar sobre este código é que o compilador não é capaz de
determinar quantos não-zeros cada linha conterá e usar essa informação para
para programar os loops. O programador sabe, no entanto, que o número de
elementos não nulos por linha é muito pequeno e este pormenor será fundamental para
para obter um elevado desempenho. 

***NOTA: Como este estudo de caso apresenta técnicas de optimização, será
necessário realizar optimizações que podem ser benéficas num hardware, mas
não em outros. Este estudo de caso foi realizado usando o compilador NVHPC 20.11 em uma GPU
GPU NVIDIA Volta V100. Estas mesmas técnicas podem ser aplicadas em outras arquiteturas,
particularmente aquelas semelhantes às GPUs NVIDIA, mas será necessário tomar
certas decisões de optimização baseadas no acelerador específico em uso.***

Ao examinar o feedback do compilador a partir do código mostrado abaixo, sei que o compilador
optou por usar um comprimento de vetor de 256 no loop mais interno. Eu
poderia também ter obtido essa informação de um perfil de tempo de execução da
aplicação. 

~~~~
    matvec(const matrix &, const vector &, const vector &):
          3, Generating Tesla code
              4, #pragma acc loop gang /* blockIdx.x */
              9, #pragma acc loop vector(128) /* threadIdx.x */
                 Generating reduction(+:sum)
          3, Generating present(ycoefs[:],xcoefs[:],row_offsets[:],Acoefs[:],cols[:])
          9, Loop is parallelizable
~~~~

Com base no meu conhecimento da matriz, sei que isto é
significativamente maior do que o número típico de não-zeros por linha, pelo que muitas
das *vector lanes* no acelerador serão desperdiçadas porque não há
trabalho suficiente para elas. A primeira coisa a tentar para melhorar o
desempenho é ajustar o comprimento do vetor usado no loop mais interno. Acontece que
sei que o compilador que estou a usar vai restringir-me a usar múltiplos do
*(o tamanho mínimo de execução do SIMT em GPUs NVIDIA) deste processador,
que é 32. Esse detalhe vai variar de acordo com o acelerador escolhido.
Abaixo está o código modificado usando um comprimento de vetor de 32.


~~~~ {.c .numberLines}
    #pragma acc parallel loop vector_length(32)
    for(int i=0;i<num_rows;i++) {
      double sum=0;
      int row_start=row_offsets[i];
      int row_end=row_offsets[i+1];
      #pragma acc loop vector reduction(+:sum)
      for(int j=row_start;j<row_end;j++) {
        unsigned int Acol=cols[j];
        double Acoef=Acoefs[j];
        double xcoef=xcoefs[Acol];
        sum+=Acoef*xcoef;
      }
      ycoefs[i]=sum;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop vector_length(32)
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop vector reduction(+:tmpsum)
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo
~~~~

Observe que agora informei explicitamente ao compilador que o loop mais interno
deve ser um laço vetorial, para garantir que o compilador mapeará o paralelismo
exactamente como eu quero. Eu posso tentar diferentes comprimentos de vetor para encontrar o valor ótimo
ideal para o meu acelerador, modificando a cláusula `vector_length`. Abaixo está um gráfico
mostrando a velocidade relativa da variação do comprimento do vetor
comparado com o valor selecionado pelo compilador.

![Relative speed-up from varying vector_length from the default value of
128](images/spmv_speedup_vector_length.png)

Repare-se que o melhor desempenho é obtido com o vector de menor comprimento. Mais uma vez,
isto deve-se ao facto de o número de não-zeros por linha ser muito pequeno, pelo que um
comprimento do vector resulta em menos recursos computacionais desperdiçados. No chip específico
estou a usar, o menor comprimento de vector possível, 32, atinge o melhor
desempenho possível. Neste acelerador em particular, eu também sei que o hardware
não terá um desempenho eficiente com esse comprimento de vetor, a menos que possamos identificar mais
paralelismo de outra forma. Neste caso, podemos usar o nível *worker* para preencher cada *gang* com mais desses vetores curtos. Abaixo está o
código modificado.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang worker num_workers(4) vector_length(32)
    for(int i=0;i<num_rows;i++) {
      double sum=0;
      int row_start=row_offsets[i];
      int row_end=row_offsets[i+1];
      #pragma acc loop vector
      for(int j=row_start;j<row_end;j++) {
        unsigned int Acol=cols[j];
        double Acoef=Acoefs[j];
        double xcoef=xcoefs[Acol];
        sum+=Acoef*xcoef;
      }
      ycoefs[i]=sum;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop gang worker num_workers(32) vector_length(32)
    do i=1,a%num_rows
      tmpsum = 0.0d0
      row_start = arow_offsets(i)
      row_end   = arow_offsets(i+1)-1
      !$acc loop vector reduction(+:tmpsum)
      do j=row_start,row_end
        acol = acols(j)
        acoef = acoefs(j)
        xcoef = x(acol)
        tmpsum = tmpsum + acoef*xcoef
      enddo
      y(i) = tmpsum
    enddo
~~~~

Nesta versão do código, mapeei explicitamente o loop mais externo para ambos
e para o paralelismo de workers e vou variar o número de workers usando a cláusula
cláusula `num_workers`. Os resultados são os seguintes.

![Speed-up from varying number of workers for a vector length of
32.](images/spmv_speedup_num_workers.png)

Neste hardware em particular, o melhor desempenho vem de um comprimento de vector de
32 e 4 workers, o que é semelhante ao loop mais simples com um comprimento de vector predefinido de 128.
Neste caso, nós
observámos um aumento de velocidade de 2,5X ao diminuir o comprimento do vector e outro aumento de 1,26X
ao variar o número de workers em cada grupo, resultando numa melhoria geral de 3,15X no desempenho em relação ao código OpenACC não ajustado.

***Boas Práticas:*** Embora não seja mostrado para economizar espaço, é geralmente
melhor usar a cláusula `device_type` sempre que especificar os tipos de
otimizações demonstradas nesta seção, porque estas cláusulas provavelmente
diferem de acelerador para acelerador. Ao utilizar a cláusula `device_type` é possível fornecer esta informação apenas em aceleradores onde as
otimizações se aplicam e permitir que o compilador tome suas próprias decisões em outras
arquiteturas. A especificação OpenACC sugere especificamente `nvidia`,
`radeon`, e `host` como três strings comuns de tipo de dispositivo.
