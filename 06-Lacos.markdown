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

As discussed earlier in this guide, the `loop` directive is intended to give the
compiler additional information about the next loop in the code. In addition to
the clauses shown before, which were intended to ensure correctness, the
clauses below inform the compiler which level of parallelism should be used to
for the given loop.

* Gang clause - partition the loop across gangs
* Worker clause - partition the loop across workers
* Vector clause - vectorize the loop
* Seq clause - do not partition this loop, run it sequentially instead

These directives may also be combined on a particular loop. For example, a
`gang vector` loop would be partitioned across gangs, each of which with 1
worker implicitly, and then vectorized. The OpenACC specification enforces that
the outermost loop must be a gang loop, the innermost parallel loop must be
a vector loop, and a worker loop may appear in between. A sequential loop may
appear at any level.

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

Informing the compiler where to partition the loops is just one part of
optimizing the loops. The programmer may additionally tell the compiler the
specific number of gangs, workers, or the vector length to use for the loops.
This specific mapping is achieved slightly differently when using the `kernels`
directive or the `parallel` directive. In the case of the `kernels` directive,
the `gang`, `worker`, and `vector` clauses accept an integer parameter that
will optionally inform the compiler how to partition that level of parallelism.
For example, `vector(128)` informs the compiler to use a vector length of 128
for the loop. 

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

When using the `parallel` directive, the information is presented
on the `parallel` directive itself, rather than on each individual loop, in the
form of the `num_gangs`, `num_workers`, and `vector_length` clauses to the
`parallel` directive.

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

Since these mappings will vary between different accelerator, the `loop`
directive accepts a `device_type` clause, which will inform the compiler that
these clauses only apply to a particular device type. Clauses after a
`device_type` clause up until either the next `device_type` or the end of the
directive will apply only to the specified device. Clauses that appear before
all `device_type` clauses are considered default values, which will be used if
they are not overridden by a later clause. For example, the code below
specifies that a vector length of 128 should be used on devices of type
`acc_device_nvidia` or a vector length of 256 should be used on devices of
type `acc_device_radeon`. The compiler will choose a default vector length for
all other device types.

~~~~ {.c .numberLines}
    #pragma acc parallel loop gang vector \
                device_type(acc_device_nvidia) vector_length(128) \
                device_type(acc_device_radeon) vector_length(256)
    for (i=0; i<N; i++)
    {
      y[i] = 2.0f * x[i] + y[i];
    }
~~~~

Collapse Clause
---------------
When a code contains tightly nested loops it is frequently beneficial to
*collapse* these loops into a single loop. Collapsing loops means that two loops
of trip counts N and M respectively will be automatically turned into a single
loop with a trip count of N times M. By collapsing two or more parallel loops into a
single loop the compiler has an increased amount of parallelism to use when
mapping the code to the device. On highly parallel architectures, such as GPUs,
this can result in improved performance. Additionally, if a loop lacked
sufficient parallelism for the hardware by itself, collapsing it with another
loop multiplies the available parallelism. This is especially beneficial on
vector loops, since some hardware types will require longer vector lengths to
achieve high performance than others. Collapsing gang loops may also be beneficial
if it allows for generating a greater number of gangs for highly-parallel processors.
The code below demonstrates how to use the collapse directive.

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

The above code is an excerpt from a real application where collapsing loops
extended the parallelism available to be exploited. On line 1, the two
outermost loops are collapsed together to make it possible to generate *gangs*
across the iterations of both loops, thus making the possible number of gangs
`nelemd` x `qsize` rather than just `nelemd`. The collapse at line 4 collapses
together 3 small loops to increase the possible *vector length*, as none of the
loops iterate for enough trips to create a reasonable vector length on the
target accelerator. How much this optimization will speed-up the code will vary
according to the application and the target accelerator, but it's not uncommon
to see large speed-ups by using collapse on loop nests.

Routine Parallelism
-------------------
A previous chapter introduced the `routine` directive for calling functions and
subroutines from OpenACC parallel regions. In that chapter it was assumed that
the routine would be called from each loop iteration, therefore requiring a
`routine seq` directive. In some cases, the routine itself may contain
parallelism that must be mapped to the device. In these cases, the `routine`
directive may have a `gang`, `worker`, or `vector` clause instead of `seq` to
inform the compiler that the routine will contain the specified level of
parallelism. This can be thought of as _reserving_ a particular level of 
parallelism for the loops in that routine. This is so that when the compiler
then encounters the call site of the affected routine, it will then know how
it can parallelize the code to use the routine. It's important to note that 
if an `acc routine` calls another routine, that routine must also have an
`acc routine` directive. At this time the OpenACC specification does not
allow for specifying multiple possible levels of parallelism on a single
routine.

Case Study - Optimize Loops
---------------------------
This case study will focus on a different algorithm than the previous chapters.
When a compiler has sufficient information about loops to make informed
decisions, it's frequently difficult to improve the performance of a given
parallel loop by more than a few percent. In some cases, the code lacks the
information necessary for the compiler to make informed optimization decisions.
In these cases, it's often possible for a developer to optimize the parallel
loops significantly by informing the compiler how to decompose and distribute
the loops to the hardware.

The code used in this section implements a sparse, matrix-vector product (SpMV)
operation. This means that a matrix and a vector will be multiplied together,
but the matrix has very few elements that are not zero (it is *sparse*),
meaning that calculating these values is unnecessary. The matrix is stored in a
Compress Sparse Row (CSR) format. In CSR the sparse array, which may contain a
significant number of cells whose value is zero, thus wasting a significant
amount of memory, is stored using three, smaller arrays: one containing the
non-zero values from the matrix, a second that describes where in a given row
these non-zero elements would reside, and a third describing the columns in
which the data would reside. The code for this exercise is below.

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

One important thing to note about this code is that the compiler is unable to
determine how many non-zeros each row will contain and use that information in
order to schedule the loops. The developer knows, however, that the number of
non-zero elements per row is very small and this detail will be key to
achieving high performance. 

***NOTE: Because this case study features optimization techniques, it is
necessary to perform optimizations that may be beneficial on one hardware, but
not on others. This case study was performed using the NVHPC 20.11 compiler on an
NVIDIA Volta V100 GPU. These same techniques may apply on other architectures,
particularly those similar to NVIDIA GPUs, but it will be necessary to make
certain optimization decisions based on the particular accelerator in use.***

In examining the compiler feedback from the code shown below, I know that the
compiler has chosen to use a vector length of 256 on the innermost loop. I
could have also obtained this information from a runtime profile of the
application. 

~~~~
    matvec(const matrix &, const vector &, const vector &):
          3, Generating Tesla code
              4, #pragma acc loop gang /* blockIdx.x */
              9, #pragma acc loop vector(128) /* threadIdx.x */
                 Generating reduction(+:sum)
          3, Generating present(ycoefs[:],xcoefs[:],row_offsets[:],Acoefs[:],cols[:])
          9, Loop is parallelizable
~~~~

Based on my knowledge of the matrix, I know that this is
significantly larger than the typical number of non-zeros per row, so many of
the *vector lanes* on the accelerator will be wasted because there's not
sufficient work for them. The first thing to try in order to improve
performance is to adjust the vector length used on the innermost loop. I happen
to know that the compiler I'm using will restrict me to using multiples of the
*warp size* (the minimum SIMT execution size on NVIDIA GPUs) of this processor,
which is 32. This detail will vary according to the accelerator of choice.
Below is the modified code using a vector length of 32.


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

Notice that I have now explicitly informed the compiler that the innermost loop
should be a vector loop, to ensure that the compiler will map the parallelism
exactly how I wish. I can try different vector lengths to find the optimal
value for my accelerator by modifying the `vector_length` clause. Below is a graph
showing the relative speed-up of varying the vector length
compared to the compiler-selected value.

![Relative speed-up from varying vector_length from the default value of
128](images/spmv_speedup_vector_length.png)

Notice that the best performance comes from the smallest vector length. Again,
this is because the number of non-zeros per row is very small, so a small
vector length results in fewer wasted compute resources. On the particular chip
I'm using, the smallest possible vector length, 32, achieves the best possible
performance. On this particular accelerator, I also know that the hardware will
not perform efficiently at this vector length unless we can identify further
parallelism another way. In this case, we can use the *worker* level of
parallelism to fill each *gang* with more of these short vectors. Below is the
modified code.

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

In this version of the code, I've explicitly mapped the outermost loop to both
gang and worker parallelism and will vary the number of workers using the
`num_workers` clause. The results follow.

![Speed-up from varying number of workers for a vector length of
32.](images/spmv_speedup_num_workers.png)

On this particular hardware, the best performance comes from a vector length of
32 and 4 workers, which is similar to the simpler loop with a default vector length of 128.
In this case, we
observed a 2.5X speed-up from decreasing the vector length and another 1.26X
speed-up from varying the number of workers within each gang, resulting in an
overall 3.15X performance improvement from the untuned OpenACC code.

***Best Practice:*** Although not shown in order to save space, it's generally
best to use the `device_type` clause whenever specifying the sorts of
optimizations demonstrated in this section, because these clauses will likely
differ from accelerator to accelerator. By using the `device_type` clause it's
possible to provide this information only on accelerators where the
optimizations apply and allow the compiler to make its own decisions on other
architectures. The OpenACC specification specifically suggests `nvidia`,
`radeon`, and `host` as three common device type strings.
