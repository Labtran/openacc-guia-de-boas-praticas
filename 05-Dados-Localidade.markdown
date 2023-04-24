Optimizar a localidade dos dados
======================
No final do capítulo anterior, vimos que, embora tenhamos movido as partes mais intensivas de computação da aplicação para o acelerador, às vezes o
processo de copiar dados do host para o acelerador e vice-versa será mais caro do que a própria computação. Isso ocorre porque é difícil para um compilador determinar quando (ou se) os dados serão necessários no
futuro, então ele deve ser cauteloso e garantir que os dados serão copiados caso seja
necessário. Para melhorar isso, vamos explorar a *localidade de dados* da aplicação.
A localidade dos dados significa que os dados usados na memória do dispositivo ou do host devem permanecer na
memória enquanto forem necessários. Esta ideia é por vezes referida como
optimização da reutilização de dados ou optimização da eliminação de cópias de dados desnecessárias entre as memórias do host e do dispositivo. De qualquer forma, fornecer ao compilador
as informações necessárias para realocar os dados apenas quando for necessário, é
frequentemente a chave para o sucesso com o OpenACC.

----

Depois de expressar o paralelismo das regiões importantes de um programa, é
frequentemente necessário fornecer ao compilador informações adicionais sobre
a localidade dos dados usados pelas regiões paralelas. Como observado na secção anterior, um compilador terá uma abordagem cautelosa à movimentação de dados, sempre
copiando os dados que podem ser necessários, para que o programa ainda produza
resultados corretos. Um programador terá conhecimento de quais dados são realmente necessários
e quando eles serão necessários. O programador também terá conhecimento de como os dados
podem ser partilhados entre duas funções, algo que é difícil para um compilador
determinar. As ferramentas de criação de perfil podem ajudar o programador a identificar 
quando ocorre excesso de movimentação de dados, como será mostrado no estudo de caso no final 
deste capítulo.

O próximo passo no processo de aceleração é fornecer ao compilador
informações adicionais sobre a localidade dos dados para maximizar a reutilização dos dados no
dispositivo e minimizar as transferências de dados. É após essa etapa que a maioria das
aplicações observará o benefício da aceleração OpenACC. Esta etapa será benéfica principalmente em máquinas onde o host e o dispositivo têm
memórias separadas.

Regiões de dados
------------
A construção `data` facilita a partilha de dados entre múltiplas
regiões paralelas. Uma região de dados pode ser adicionada em torno de uma ou mais regiões paralelas na mesma função ou pode ser colocada em um nível mais alto na árvore de chamadas do programa
para permitir que os dados sejam compartilhados entre regiões em múltiplas funções.
A construção `data` é uma construção estruturada, o que significa que ela deve começar e terminar no mesmo escopo (como a mesma função ou sub-rotina). Uma seção posterior discutirá como lidar com casos onde uma construção estruturada não é
útil. Uma região `data` pode ser adicionada ao exemplo anterior do `parallel loop` para
permitir que os dados sejam compartilhados entre os dois ninhos de laço, como segue.

~~~~ {.c .numberLines}
    #pragma acc data
    {
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
    }
~~~~

----

~~~~ {.fortran .numberLines}
    !$acc data
    !$acc parallel loop
    do i=1,N
      y(i) = 0
      x(i) = i
    enddo
  
    !$acc parallel loop
    do i=1,N
      y(i) = 2.0 * x(i) + y(i)
    enddo
    !$acc end data
~~~~

A região `data` nos exemplos acima permite que as matrizes `x` e `y` sejam
reutilizados entre as duas regiões `parallel`. Isso removerá qualquer cópia de dados que acontecem entre as duas regiões, mas ainda não garante uma
movimentação de dados. Para fornecer as informações necessárias para realizar uma movimentação de dados, o programador pode adicionar cláusulas de dados à região `data`.

*Nota:* Uma região de dados implícita é criada por cada região `parallel` e `kernels`.

Cláusulas de dados 
------------
As cláusulas de dados dão ao programador controle adicional sobre como e quando os dados são
criados no dispositivo e copiados de ou para o mesmo. Estas cláusulas podem ser adicionadas a qualquer construção
`data`, `parallel`, ou `kernels` para informar o compilador sobre as necessidades de dados daquela região do código. As diretivas de dados, juntamente com uma breve
descrição de seus significados, seguem.

* `copy` - Cria espaço para as variáveis listadas no dispositivo, inicializa a variável copiando dados para o dispositivo no início da região, copia os resultados de volta para o anfitrião no final da região e, finalmente, libera
  o espaço no dispositivo quando terminar.
* `copyin` - Cria espaço para as variáveis listadas no dispositivo, inicializa a variável copiando dados para o dispositivo no início da região e libera o espaço no dispositivo quando termina sem copiar os dados de volta para o host.
* `copyout` - Crie espaço para as variáveis listadas no dispositivo, mas não as inicializa. No final da região, copia os resultados de volta para o anfitrião e libera o espaço no dispositivo.
* `create` - Cria espaço para as variáveis listadas e libera no final da região, mas não copia para ou do dispositivo.
* `present` - As variáveis listadas já estão presentes no dispositivo, pelo que não é necessário realizar qualquer acção adicional. Isto é mais frequentemente usado quando uma região de dados existe numa rotina de nível superior.
* `deviceptr` - As variáveis listadas utilizam a memória do dispositivo que foi gerida fora do OpenACC, pelo que as variáveis devem ser utilizadas no dispositivo sem qualquer tradução de endereços. Esta cláusula é geralmente utilizada quando o OpenACC é misturado com outro modelo de programação, como será discutido no capítulo de interoperabilidade.

No caso das cláusulas `copy`, `copyin`, `copyout` e `create`, a funcionalidade pretendida não 
ocorrerá se a variável referenciada já existir na memória do dispositivo. Pode ser útil pensar 
nessas cláusulas como tendo uma cláusula `present` implícita anexada a elas, onde se a variável 
estiver presente no dispositivo, a outra cláusula será ignorada. Um exemplo importante deste 
comportamento é a utilização da cláusula `copy` quando a variável já existe na memória do dispositivo 
não copiará nenhum dados entre o host e o dispositivo. Existe uma diretiva diferente para copiar dados 
entre o host e o dispositivo de dentro de uma região de dados, e será discutida em breve.

### Moldar matrizes ###
Por vezes, um compilador precisará de alguma ajuda extra para determinar o tamanho e a forma
das matrizes usadas em regiões paralelas ou de dados. Na maior parte das vezes, os programadores Fortran
podem confiar na natureza auto-descritora das matrizes Fortran, mas os programadores C/C++
frequentemente precisam fornecer informações adicionais ao compilador
para que ele saiba o tamanho da matriz a ser alocada no dispositivo e a quantidade de dados
que precisam ser copiados. Para dar esta informação, o programador acrescenta uma especificação *shape*
às cláusulas de dados. 

Em C/C++, a forma de uma matriz é descrita
como `x[start:count]` onde *x* é o nome da variável, *start* é o primeiro elemento a ser copiado e
*count* é o número de elementos a copiar. Se o primeiro elemento for 0, então ele
pode ser deixado de fora, tomando a forma de `x[:count]`. 

Em Fortran, a forma de uma matriz é descrita como `x(start:end)` onde *x* é o nome da variável  
nome da variável, *start* é o primeiro elemento a ser copiado e *end* é o último elemento
a ser copiado. Se *início* for o início da matriz ou *fim* for o fim da matriz,
podem ser deixados de fora, assumindo a forma de `x(:end)`, `x(start:)` ou `x(:)`. 

A modelagem de matrizes é frequentemente necessária em códigos C/C++ quando o OpenACC aparece
dentro de chamadas de função ou as matrizes são alocadas dinamicamente, uma vez que a
forma da matriz não será conhecida em tempo de compilação. A modelagem também é útil
quando apenas uma parte da matriz precisa ser armazenada no dispositivo. 

Como um exemplo de modelagem de matriz, o código abaixo modifica o exemplo anterior
adicionando informações de forma a cada uma das matrizes.

~~~~ {.c .numberLines}
    #pragma acc data create(x[0:N]) copyout(y[0:N])
    {
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
    }
~~~~

----

~~~~ {.fortran .numberLines}
    !$acc data create(x(1:N)) copyout(y(1:N))
    !$acc parallel loop
    do i=1,N
      y(i) = 0
      x(i) = i
    enddo
  
    !$acc parallel loop
    do i=1,N
      y(i) = 2.0 * x(i) + y(i)
    enddo
    !$acc end data
~~~~

----

Neste exemplo, o programador sabe que tanto `x` quanto `y` serão preenchidos com dados no dispositivo, então nenhum deles precisa ter dados copiados
do host. Entretanto, como `y` é utilizado dentro de uma cláusula `copyout`,
os dados contidos em `y` serão copiados do dispositivo para o host
quando o fim da região de dados for alcançado. Isso é útil em uma situação
onde você precisa dos resultados armazenados em `y` mais tarde no código do host.

Tempos de vida dos dados não estruturados
---------------------------
Embora as regiões de dados estruturados sejam suficientes para optimizar a
localidade de dados em muitos programas, não são suficientes em alguns casos, particularmente
aqueles que utilizam práticas de codificação orientadas para objetos, ou quando desejam gerenciar dados de dispositivos em diferentes arquivos de código. Por exemplo, numa classe C++, os dados
são frequentemente atribuídos num construtor de classe, desalocados no destruidor,
e não podem ser acedidos fora da classe. Isso impossibilita o uso de regiões de dados estruturados impossível porque não há um escopo único e estruturado onde o construtor
pode ser colocado.  Para essas situações, podemos usar
tempos de vida de dados não estruturados. As diretivas `enter data` e `exit data` podem ser
usadas para identificar precisamente quando os dados devem ser alocados e desalocados no
dispositivo. 

A diretiva `enter data` aceita as cláusulas de dados `create` e `copyin` e
pode ser usada para especificar quando os dados devem ser criados no dispositivo.

A directiva `exit data` aceita a cláusula `copyout` e uma cláusula especial `delete` para especificar quando os dados devem ser removidos do dispositivo.  

Se uma variável aparecer em várias diretivas `enter data`, ela só será
eliminada do dispositivo se for utilizado um número equivalente de diretivas `exit data`
forem utilizadas. Para garantir que os dados sejam apagados, você pode adicionar a cláusula `finalize`
à diretiva `exit data`. Além disso, se uma variável aparecer
em várias diretivas `enter data`, apenas a instância fará qualquer
movimentação de dados de host para dispositivo. Se você precisar mover dados entre o host
e o dispositivo a qualquer momento após os dados serem alocados com `enter data`, você deve
utilizar a diretiva `update`, que é discutida mais adiante neste capítulo.

### Dados de classe C++ ###
Os dados de classe C++ são uma das principais razões pelas quais os tempos de vida dos dados não estruturados
foram adicionados ao OpenACC. Como descrito acima, o encapsulamento fornecido pelas classes torna impossível o uso de uma região `data` estruturada para controlar a
localidade dos dados da classe. Os programadores podem optar por usar as diretivas de tempo de vida de dados não estruturados
ou a API OpenACC para controlar a localidade dos dados dentro de uma classe C++. O uso das diretivas é preferível, uma vez que serão ignoradas com segurança
por compiladores não-OpenACC, mas a API também está disponível para momentos em que as diretivas
não são expressivas o suficiente para atender às necessidades do programador. A API não será discutida neste guia, mas está bem documentada no site do OpenACC website.

O exemplo abaixo mostra uma classe C++ simples que tem um construtor, um
destrutor e um construtor de cópia. O gerenciamento de dados dessas rotinas foi
foi tratada usando as diretivas OpenACC.

~~~~ {.cpp .numberLines}
    template <class ctype> class Data
    {
      private:
        /// Length of the data array
        int len;
        /// Data array
        ctype *arr;
    
      public:
        /// Class constructor
        Data(int length)
        {
          len = length;
          arr = new ctype[len];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(arr[0:len])
        }

        /// Copy constructor
        Data(const Data<ctype> &d)
        {
          len = d.len;
          arr = new ctype[len];
    #pragma acc enter data copyin(this)
    #pragma acc enter data create(arr[0:len])
    #pragma acc parallel loop present(arr[0:len],d)
          for(int i = 0; i < len; i++)
            arr[i] = d.arr[i];
        }

        /// Class destructor
        ~Data()
        {
    #pragma acc exit data delete(arr)
    #pragma acc exit data delete(this)
          delete arr;
          len = 0;
        }
    };
~~~~

Observe que uma diretiva `enter data` é adicionada ao construtor da classe para lidar com a criação de espaço para os dados da classe no dispositivo. Além do array de dados o ponteiro `this` é copiado para o dispositivo. A cópia do ponteiro `this`
garante que o membro escalar `len`, que denota o comprimento do
do array de dados `arr`, e o ponteiro `arr` estejam disponíveis no acelerador
bem como no host. É importante colocar a diretiva `enter data` depois que os dados da classe tenham sido inicializados. Da mesma forma, as diretivas `exit data` são adicionadas
ao destrutor para lidar com a limpeza da memória do dispositivo. É importante colocar esta diretiva antes que os membros do array sejam liberados, porque uma vez que as cópias
do host forem liberadas, o ponteiro subjacente pode se tornar inválido, tornando impossível a liberação da memória do dispositivo. Pela mesma razão, o ponteiro `this`
não deve ser removido do dispositivo até que toda a outra memória tenha sido liberada.

O construtor de cópia é um caso especial que vale a pena analisar por si só. O construtor de cópia será responsável por alocar espaço no dispositivo para a
que ele está criando, mas ele também dependerá de dados que são gerenciados pela classe que está sendo copiada. Como o OpenACC não fornece atualmente uma maneira portátil de copiar de um array para outro, como um `memcpy` no host, um laço
é usado para copiar cada elemento individual de um array para o outro.
Como sabemos que o objeto `Data` passado também terá seus membros no dispositivo, nós usamos uma cláusula `present` no `parallel loop` para informar o
compilador que nenhuma movimentação de dados é necessária.

----

A mesma técnica usada no construtor e no destrutor da classe acima pode ser
usada em outras linguagens de programação também. Por exemplo, é prática comum
em códigos Fortran ter uma sub-rotina que aloca e inicializa todos os arrays
contidas em um módulo. Tal rotina é um lugar natural para usar uma região `enter
data`, pois a alocação da memória do host e do dispositivo aparecerá
dentro da mesma rotina no código. Colocando as directivas `enter data` e `exit data`
próximas à alocação e desalocação usual de dados no código simplifica a manutenção do código.

Diretiva Update
----------------
Manter os dados residentes no acelerador é muitas vezes a chave para obter um alto
desempenho, mas às vezes é necessário copiar dados entre as memórias do host e do dispositivo. A diretiva `update` fornece uma maneira de explicitamente
atualizar os valores da memória do host ou do dispositivo com os valores do outro. Isto
pode ser considerado como a sincronização do conteúdo das duas memórias. A diretiva
`update` aceita uma cláusula `device` para copiar dados do host para o
para o dispositivo e uma cláusula `self` para atualizar do dispositivo para a memória local,
que é a memória do host.

Como um exemplo da diretiva `update`, abaixo estão duas rotinas que podem ser
adicionadas à classe `Data` acima para forçar uma cópia do host para o dispositivo e do dispositivo
para o host.

~~~~ {.c .numberLines}
    void update_host()
    {
    #pragma acc update self(arr[0:len])
      ;
    }
    void update_device()
    {
    #pragma acc update device(arr[0:len])
      ;
    }
~~~~

As cláusulas de atualização aceitam uma forma de matriz, como já foi discutido na secção de cláusulas data. Embora o exemplo acima copie toda a matriz `arr` para ou do dispositivo, uma matriz parcial também pode ser fornecida para reduzir o custo de transferência de dados quando apenas parte de uma matriz precisa ser atualizada, como por exemplo quando troca de condições de contorno.

***Boas Práticas:*** Tal como referido anteriormente no documento, as variáveis num código OpenACC
devem ser sempre consideradas como um objecto singular, em vez de uma cópia *host* e
uma cópia *device*. Mesmo ao desenvolver em uma máquina com um host unificado e
memória do dispositivo é importante incluir uma diretiva `update` sempre que
acessar dados do host ou dispositivo que foram previamente escritos pelo
outro, pois isso garante a correção em
todos os dispositivos.  Para sistemas com memórias distintas, o `update` sincronizará
os valores da variável afetada no host e no dispositivo. Em dispositivos com
uma memória unificada, a atualização será ignorada, não incorrendo em nenhuma penalidade de desempenho.
No exemplo abaixo, a omissão do `update` na linha 17 produzirá resultados diferentes em uma máquina com memória unificada e não unificada, tornando o código não portátil.


~~~~ {.c .numberLines}
    for(int i=0; i<N; i++)
    {
      a[i] = 0;
      b[i] = 0;
    }
    
    #pragma acc enter data copyin(a[0:N])
    
    #pragma acc parallel loop
    {
      for(int i=0; i<N; i++)
      {
        a[i] = 1; 
      }
    }
    
    #pragma acc update self(a[0:N])
    
    for(int i=0; i<N; i++)
    {
      b[i] = a[i];  
    }
    
    #pragma acc exit data
~~~~

<!---
Cache Directive
---------------
***Delaying slightly because the cache directive is still being actively
improved in the PGI compiler.***

Some parallel accelerators, GPUs in particular, have a high-speed memory that
can serve as a user-managed cache. OpenACC provides a mechanism for declaring
arrays and parts of arrays that would benefit from utilizing a fast memory if
it's available within each gang. The `cache` directive may be placed within a
loop and specify the array or array section should be placed in a fast memory
for the extent of that loop.

Global Data
-----------
***Discuss `declare` directive.***

When dealing with global data, such as variables that are declared globally,
static to the file, or extern in C and C++ or common blocks and their contained
data in Fortran, data regions and unstructured data directives are not
sufficient. In these cases it is necessary to use the `declare` directive to
declare that these variables should be available on the device. The `declare`
directive has many complexities, which will be discussed as needed, so this
section will only discuss it in the context of global variables in C anc C++
and common blocks in Fortran.
--->

Best Practice: Offload Inefficient Operations to Maintain Data Locality
-----------------------------------------------------------------------
Devido ao elevado custo das transferências de dados PCIe em sistemas com memórias distintas de host
e do dispositivo, é frequentemente benéfico mover secções da aplicação para o dispositivo acelerador, mesmo quando o código não tem paralelismo suficiente para ver
benefício direto. A perda de desempenho da execução de código em série ou com um
baixo grau de paralelismo num acelerador paralelo é frequentemente menor do que o custo de
transferência de matrizes para trás e para a frente entre as duas memórias. Um programador pode
utilizar uma região `paralela` com apenas 1 gang como uma forma de descarregar uma secção
do código para o acelerador. Por exemplo, no código abaixo, o primeiro e os últimos elementos do array são elementos do host que precisam ser definidos como zero. A
região `parallel` (sem um `loop`) é usada para executar as partes que são
em série.

~~~~ {.c .numberLines}
    #pragma acc parallel loop
    for(i=1; i<(N-1); i++)
    {
      // calculate internal values
      A[i] = 1;
    }
    #pragma acc parallel
    {
      A[0]   = 0;
      A[N-1] = 0;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc parallel loop
    do i=2,N-1
      ! calculate internal values
      A(i) = 1
    end do
    !$acc parallel
      A(1) = 0;
      A(N) = 0;
    !$acc end parallel
~~~~

No exemplo acima, a segunda região `parallel` irá gerar e lançar um
pequeno kernel para definir o primeiro e o último elemento. Os kernels pequenos geralmente
não são executados por tempo suficiente para superar o custo de um lançamento de kernel em alguns
dispositivos de descarregamento, como as GPUs. É importante que a transferência de dados economizada
ao empregar esta técnica seja grande o suficiente para superar o alto custo de uma
lançamento do kernel em alguns dispositivos. Tanto o `parallel loop` quanto a segunda região
região `parallel` podem ser feitas assíncronas (discutido num capítulo posterior) para
reduzir o custo do segundo lançamento do kernel.

*Nota: Como a diretiva `kernels` instrui o compilador a procurar por
paralelismo, não existe uma técnica similar para `kernels`, mas a abordagem `parallel`
acima pode ser facilmente colocada entre as regiões `kernels`.*

Estudo de caso - Optimizar a localidade dos dados
-----------------------------------
No final do último capítulo, tínhamos deslocado os principais laços computacionais do nosso código de exemplo e, ao fazê-lo, introduzimos uma quantidade significativa de
transferências de dados implícitas. O perfil de desempenho do nosso código mostra que para cada
iteração os arrays `A` e `Anew` estão sendo copiados para frente e para trás entre o
*host* e *device*, quatro vezes para a versão `parallel loop` e duas vezes para a versão
a versão `kernels`. Dado que os valores para esses vetores não são necessários
até que a resposta tenha convergido, vamos adicionar uma região de dados ao redor do
laço de convergência. Além disso, precisaremos especificar como os vetores devem ser
geridos por esta região de dados. Tanto o valor inicial quanto o valor final do vetor `A`
são necessários, então esse array precisará de uma cláusula de dados `copy`. Os
resultados no array `Anew`, no entanto, são utilizados apenas dentro desta secção de
código, então uma cláusula `create` será usada para ele. O código resultante é mostrado
abaixo.

*Nota: As alterações necessárias durante este passo são as mesmas para ambas as versões do código, então apenas a versão `parallel loop` será mostrada.*

~~~~ {.c .numberLines startFrom="51"}
    #pragma acc data copy(A[:n][:m]) create(Anew[:n][:m])
        while ( error > tol && iter < iter_max )
        {
            error = 0.0;
    
            #pragma acc parallel loop reduction(max:error)
            for( int j = 1; j < n-1; j++)
            {
                #pragma acc loop reduction(max:error)
                for( int i = 1; i < m-1; i++ )
                {
                    Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                        + A[j-1][i] + A[j+1][i]);
                    error = fmax( error, fabs(Anew[j][i] - A[j][i]));
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

~~~~ {.fortran .numberLines startFrom="51"}
    !$acc data copy(A) create(Anew)
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
    !$acc end data
~~~~    

Com esta alteração, apenas o valor calculado para o erro máximo, que é
exigido pelo laço de convergência, é copiado do dispositivo a cada iteração.
Os vetores `A` e `Anew` permanecerão locais ao dispositivo através da extensão
deste cálculo. Usando o NVIDIA NSight Systems novamente, vemos que cada
transferências de dados agora só ocorre no início e no final da região de dados e
que o tempo entre cada iteração é muito menor. 

![NVIDIA Nsight Systems showing a single iteration of the Jacobi solver after adding
the OpenACC data region.](images/ch4_profile.png)

Olhando para o desempenho final deste código, vemos que o tempo para o código OpenACC numa GPU é agora muito mais rápido do que até mesmo o melhor código de CPU com threads.
Embora apenas a versão `parallel loop` seja mostrada no gráfico de desempenho,
a versão `kernels` tem um desempenho igualmente bom, uma vez que a região `data` foi
adicionada.

![Runtime of Jacobi Iteration after adding OpenACC data
region](images/jacobi_step2_graph.png)

Termina assim o estudo de caso da Iteração Jacobi. A simplicidade desta implementação mostra geralmente bons aumentos de velocidade com o OpenACC, deixando frequentemente
pouco potencial para mais melhorias. O leitor deve sentir-se encorajado,
no entanto, a revisitar este código para ver se são possíveis mais melhorias
no dispositivo que lhe interessa.
