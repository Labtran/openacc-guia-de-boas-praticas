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
próximo laço no código fonte. A directiva `loop` foi mostrada acima em
ligação com a directiva `parallel`, embora também seja válida com
`kernels`. As cláusulas de laço vêm em duas formas: cláusulas de correção e cláusulas
para optimização. Este capítulo irá discutir apenas as duas cláusulas de correção
e um capítulo posterior irá discutir cláusulas de optimização.

### private ###
The private clause specifies that each loop iteration requires its own copy of
the listed variables. For example, if each loop contains a small, temporary
array named `tmp` that it uses during its calculation, then this variable must
be made private to each loop iteration in order to ensure correct results. If
`tmp` is not declared private, then threads executing different loop iterations
may access this shared `tmp` variable in unpredictable ways, resulting in a
race condition and potentially incorrect results. Below is the synax for the
`private` clause.

    private(var1, var2, var3, ...)

There are a few special cases that must be understood about scalar
variables within loops. First, loop iterators will be privatized by default, so
they do not need to be listed as private. Second, unless otherwise specified,
any scalar accessed within a parallel loop will be made *first private* by
default, meaning a private copy will be made of the variable for each loop
iteration and it will be initialized with the value of that scalar upon
entering the region. Finally, any variables (scalar or not) that are declared
within a loop in C or C++ will be made private to the iterations of that loop
by default.

Note: The `parallel` construct also has a `private` clause which will privatize
the listed variables for each gang in the parallel region. 

### reduction ###
The `reduction` clause works similarly to the `private` clause in that a
private copy of the affected variable is generated for each loop iteration, but
`reduction` goes a step further to reduce all of those private copies into one
final result, which is returned from the region. For example, the maximum of
all private copies of the variable may be required. A
reduction may only be specified on a scalar variable and only common, specified
operations can be performed, such as `+`, `*`, `min`, `max`, and various
bitwise operations (see the OpenACC specification for a complete list). The
format of the reduction clause is as follows, where *operator* should be
replaced with the operation of interest and *variable* should be replaced with
the variable being reduced:

    reduction(operator:variable)

An example of using the `reduction` clause will come in the case study below.

Routine Directive
-----------------
Function or subroutine calls within parallel loops can be problematic for
compilers, since it's not always possible for the compiler to see all of the
loops at one time. OpenACC 1.0 compilers were forced to either inline all
routines called within parallel regions or not parallelize loops containing
routine calls at all. OpenACC 2.0 introduced the `routine` directive to address
this shortcoming. The `routine` directive gives the compiler the necessary
information about the function or subroutine and the loops it contains in order
to parallelize the calling parallel region. The routine directive must be added
to a function definition informing the compiler of the level of parallelism
used within the routine. OpenACC's *levels of parallelism* will be discussed in a
later section.

### C++ Class Functions ###
When operating on C++ classes, it's frequently necessary to call class
functions from within parallel regions. The example below shows a C++ class
`float3` that contains 3 floating point values and has a `set` function that is
used to set the values of its `x`, `y`, and `z` members to that of another
instance of `float3`. In order for this to work from within a parallel region,
the `set` function is declared as an OpenACC routine using the `routine`
directive. Since we know that it will be called by each iteration of a parallel
loop, it's declared a `seq` (or *sequential*) routine. 

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
When one or more loop iterations need to access an element in memory at the
same time data races can occur. For instance, if one loop iteration is
modifying the value contained in a variable and another is trying to read from
the same variable in parallel, different results may occur depending on which
iteration occurs first. In serial programs, the sequential loops ensure that
the variable will be modified and read in a predictable order, but parallel
programs don't make guarantees that a particular loop iteration will happen
before another. In simple cases, such as finding a sum, maximum, or minimum
value, a reduction operation will ensure correctness. For more complex
operations, the `atomic` directive will ensure that no two threads can attempt
to perfom the contained operation simultaneously. Use of atomics is sometimes a
necessary part of parallelization to ensure correctness.

The `atomic` directive accepts one of four clauses to declare the type of
operation contained within the region. The `read` operation ensures that no two
loop iterations will read from the region at the same time. The `write`
operation will ensure that no two iterations with write to the region at the
same time. An `update` operation is a combined read and write. Finally a
`capture` operation performs an update, but saves the value calculated in that
region to use in the code that follows. If no clause is given, then an update
operation will occur.

### Atomic Example ###

<!-- ![A histogram of number distribution.](images/histogram.png) -->

A histogram is a common technique for counting up how many times values occur
from an input set according to their value. The example
code below loops through a series of integer numbers of a known range and
counts the occurances of each number in that range. Since each number in the
range can occur multiple times, we need to ensure that each element in the
histogram array is updated atomically. The code below demonstrates using the
`atomic` directive to generate a histogram.

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

Notice that updates to the histogram array `h` are performed atomically.
Because we are incrementing the value of the array element, an update operation
is used to read the value, modify it, and then write it back.

Case Study - Parallelize
------------------------
In the last chapter we identified the two loop nests within the convergence
loop as the most time consuming parts of our application.  Additionally we
looked at the loops and were able to determine that the outer convergence loop
is not parallel, but the two loops nested within are safe to parallelize. In
this chapter we will accelerate those loop nests with OpenACC using the
directives discussed earlier in this chapter. To further emphasize the
similarities and differences between `parallel` and `kernels` directives, we
will accelerate the loops using both and discuss the differences.

### Parallel Loop ###
We previously identified the available parallelism in our code, now we will use
the `parallel loop` directive to accelerate the loops that we identified. Since
we know that the two doubly-nested sets of loops are parallel, simply add a
`parallel loop` directive above each of them. This will inform the compiler
that the outer of the two loops is safely parallel. Some compilers will
additionally analyze the inner loop and determine that it is also parallel, but
to be certain we will also add a `loop` directive around the inner loops. 

There is one more subtlety to accelerating the loops in this example: we are
attempting to calculate the maximum value for the variable `error`. As
discussed above, this is considered a *reduction* since we are reducing from
all possible values for `error` down to just the single maximum. This means
that it is necessary to indicate a reduction on the first loop nest (the one
that calculates `error`). 

***Best Practice:*** Some compilers will detect the reduction on `error` and
implicitly insert the `reduction` clause, but for maximum portability the
programmer should always indicate reductions in the code.

At this point the code looks like the examples below.

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

***Best Practice:*** Most OpenACC compilers will accept only the `parallel
loop` directive on the `j` loops and detect for themselves that the `i` loop
can also be parallelized without needing the `loop` directives on the `i`
loops. By placing a `loop` directive on each loop that can be
parallelized, the programmer ensures that the compiler will understand that the
loop is safe the parallelize. When used within a `parallel` region, the `loop`
directive asserts that the loop iterations are independent of each other and
are safe the parallelize and should be used to provide the compiler as much
information about the loops as possible.

Building the above code using the NVHPC compiler produces the
following compiler feedback (shown for C, but the Fortran output is similar).

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


Analyzing the compiler feedback gives the programmer the ability to ensure that
the compiler is producing the expected results or fix any problems.
In the output above we see that accelerator kernels were generated for the two
loops that were identified (at lines 58 and 71, in the compiled source file)
and that the compiler automatically generated data movement, which will be
discussed in more detail in the next chapter.

Other clauses to the `loop` directive that may further benefit the performance
of the resulting code will be discussed in a later chapter.  

<!---(***TODO: Link to later chapter when done.***)--->

### Kernels ###
Using the `kernels` construct to accelerate the loops we've identified requires
inserting just one directive in the code and allowing the compiler to perform
the parallel analysis. Adding a `kernels` construct around the two
computational loop nests results in the following code.

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
    
The above code demonstrates some of the power that the `kernels` construct
provides, since the compiler will analyze the code and identify both loop nests
as parallel and it will automatically discover the reduction on the `error`
variable without programmer intervention. An OpenACC compiler will likely
discover not only that the outer loops are parallel, but also the inner loops,
resulting in more available parallelism with fewer directives than the
`parallel loop` approach. Had the programmer put the `kernels` construct around
the convergence loop, which we have already determined is not parallel, the
compiler likely would not have found any available parallelism. Even with the
`kernels` directive it is necessary for the programmer to do some amount of
analysis to determine where parallelism may be found.

Taking a look at the compiler output points to some more subtle differences
between the two approaches.

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

The first thing to notice from the above output is that the compiler correctly
identified all four loops as being parallelizable and generated kernels from
those loops. Also notice that the compiler only generated implicit data
movement directives at line 54 (the beginning of the `kernels` region), rather
than at the beginning of each `parallel loop`. This means that the resulting
code should perform fewer copies between host and device memory in this version
than the version from the previous section. A more subtle difference between
the output is that the compiler chose a different loop decomposition scheme (as
is evident by the implicit `acc loop` directives in the compiler output) than
the parallel loop because `kernels` allowed it to do so. More details on how to
interpret this decomposition feedback and how to change the behavior will be
discussed in a later chapter.

---

At this point we have expressed all of the parallelism in the example code and
the compiler has parallelized it for an accelerator device. Analyzing the
performance of this code may yield surprising results on some accelerators,
however. The results below demonstrate the performance of this code on 1 - 16
CPU threads on an AMD Threadripper CPU and an NVIDIA Volta V100
GPU using both implementations above. The *y axis* for figure 3.1 is execution
time in seconds, so smaller is better. For the two OpenACC versions, the bar is
divided by time transferring data between the host and device and time executing
on the device.

![Jacobi Iteration Performance - Step 1](images/jacobi_step1_graph.png)

The performance of this improves as more CPU threads are added to the calculation,
however, since the code is memory-bound the performance benefit of adding
additional threads quickly diminishes. Also, the OpenACC versions perform poorly
compared to the CPU
baseline. The both the OpenACC `kernels` and `parallel loop` versions perform
worse than the serial CPU baseline. It is also clear that the `parallel loop` version
spends significantly more time in data transfer than the `kernels` version.
Further performance analysis is necessary to
identify the source of this slowdown. This analysis has already been applied to
the graph above, which breaks down time spent
computing the solution and copying data to and from the accelerator.

A variety of tools are available for performing this analysis, but since this
case study was compiled for an NVIDIA GPU, NVIDIA Nsight Systems will be
used to understand the application peformance. The screenshot in figure 3.2
shows Nsight Systems profile for ***2*** iterations of the convergence loop in
the `parallel loop` version of the code.

![Screenshot of NVIDIA Nsight Systems Profile on 2 steps of the Jacobi Iteration
showing a high amount of data transfer compared to
computation.](images/ch3_profile.png) 

Since the test machine has two distinct memory spaces, one for the CPU and one
for the GPU, it's necessary to copy data between the two memories. In this
screenshot, the tool represents data transfers using the tan colored boxes in the
two *MemCpy* rows and the computation time in the green and purple boxes in the
rows below *Compute*. It should be obvious from the timeline displayed that
significantly more time is being spent copying data to and from the
accelerator before and after each compute kernel than actually computing on the
device. In fact, the majority of the time is spent either in memory copies or
in overhead incurred by the runtime scheduling memory copeis. In the next
chapter we will fix this inefficiency, but first, why does the `kernels`
version outperform the `parallel loop` version? 

When an OpenACC compiler parallelizes a region of code it must analyze the data
that is needed within that region and copy it to and from the accelerator if
necessary. This analysis is done at a per-region level and will typically
default to copying arrays used on the accelerator both to and from the device
at the beginning and end of the region respectively. Since the `parallel loop`
version has two compute regions, as opposed to only one in the `kernels`
version, data is copied back and forth between the two regions. As a result,
the copy and overhead times are roughly twice that of the `kernels` region,
although the compute kernel times are roughly the same.
