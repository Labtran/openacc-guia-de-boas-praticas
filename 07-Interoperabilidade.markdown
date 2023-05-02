OpenACC Interoperability
========================
Os autores do OpenACC reconheceram que, por vezes, pode ser benéfico misturar OpenACC com código acelerado usando outras linguagens de programação paralela,
como CUDA ou OpenCL, ou bibliotecas matemáticas aceleradas. Essa interoperabilidade
significa que um desenvolvedor pode escolher o paradigma de programação que faz mais sentido numa situação específica e aproveitar o código e as bibliotecas que
já estejam disponíveis. Os programadores não precisam decidir no início de um projeto
entre OpenACC *ou* outra coisa, podem optar por usar OpenACC *e*
outras tecnologias. 

***NOTE:*** The examples used in this chapter can be found online at
https://github.com/jefflarkin/openacc-interoperability

A Região de Dados do Host
--------------------
O primeiro método para interoperar entre o OpenACC e algum outro código é
gerindo todos os dados utilizando o OpenACC, mas chamando uma função que requer
dados do dispositivo. Para o propósito do exemplo, a rotina `cublasSaxpy` será usada
em vez de escrever uma rotina *saxpy*, como foi mostrado num capítulo anterior. Esta rotina é fornecida gratuitamente pela NVIDIA para o seu hardware na biblioteca CUBLAS.
A maioria dos outros fabricantes fornecem sua própria biblioteca ajustada.

A diretiva `host_data` dá ao programador uma maneira de expor o endereço do dispositivo
de um determinado array para o host para passar para uma função. Estes dados devem ter
já ter sido movidos para o dispositivo anteriormente. O nome desta construção muitas vezes 
confunde os novos utilizadores, mas pode ser considerado como uma região `data` inversa, uma vez que 
ela pega os dados no `dispositivo` e os expõe para o `host`. A região `host_data`
aceita apenas a cláusula `use_device`, que especifica quais variáveis do dispositivo devem ser expostas ao host. No exemplo abaixo, os arrays `x`
e `y` são colocados no dispositivo através de uma região `data` e então inicializados em
um loop OpenACC. Estas matrizes são então passadas para a função `cublasSaxpy
como ponteiros de dispositivo usando a região `host_data`.

~~~~ {.c .numberLines}
    #pragma acc data create(x[0:n]) copyout(y[0:n])
    {
      #pragma acc kernels
      {
        for( i = 0; i < n; i++)
        {
          x[i] = 1.0f;
          y[i] = 0.0f;
        }
      }
  
      #pragma acc host_data use_device(x,y)
      {
        cublasSaxpy(n, 2.0, x, 1, y, 1);
      }
    }
~~~~

---

~~~~ {.fortran .numberLines}
    !$acc data create(x,y)
    !$acc kernels
    X(:) = 1.0
    Y(:) = 0.0
    !$acc end kernels

    !$acc host_data use_device(x,y)
    call cublassaxpy(N, 2.0, x, 1, y, 1)
    !$acc end host_data
    !$acc update self(y)
    !$acc end data
~~~~

A chamada para `cublasSaxpy` pode ser alterada para qualquer função que espere ponteiros de dispositivo
como parâmetros.

### Bibliotecas de Dispositivos Assíncronos
***NOTA:*** Ao utilizar a região `host_data` para passar dados para chamadas assíncronas de bibliotecas ou kernels, deve-se tomar cuidado com o tempo de vida dos dados
no dispositivo. Um exemplo comum deste padrão é passar dados do dispositivo para uma biblioteca
biblioteca MPI com reconhecimento de dispositivo, como ilustrado abaixo.   

Um uso comum da região `host_data` é passar ponteiros de dispositivos para uma
implementação MPI sensível a dispositivos. Tais bibliotecas MPI podem ter otimizações específicas
quando passam dados de dispositivo, como Remote Direct Memory Access
(RDMA) ou pipelining. Para rotinas MPI síncronas, a diretiva `host_data`
pode ser usada como mostrado acima, mas deve-se tomar cuidado ao misturar esta diretiva
com funções MPI assíncronas (por exemplo, MPI_ISend, MPI_IRecv, etc.).
Veja, por exemplo, o código a seguir:

~~~~ {.c .numberLines}
    #pragma acc data copyin(buf)
    { // Data in `buf` put on device
    #pragma acc host_data use_device(buf)
    { // Device pointer to `buf` passed to MPI
       MPI_Isend(buf, ...);
       // MPI_Isend immediatly returns to main thread
    }
    // MPI_Isend may not have completed sending data
    } // Data in `buf` potentially removed from device
~~~~

~~~~ {.fortran .numberLines}
    !$acc data copyin(buf)
    ! Data in `buf` put on device
    !$acc host_data use_device(buf)
    ! Device pointer to `buf` passed to MPI
       call MPI_Isend(buf, ...);
       ! MPI_Isend immediatly returns to main thread
    !$acc end host_data
    ! MPI_Isend may not have completed sending data
    !$acc end data
    ! Data in `buf` potentially removed from device
~~~~

No exemplo acima, o ponteiro de dispositivo para os dados em `buf` é fornecido para `MPI_ISend`, 
que devolverá imediatamente o controlo à thread de execução, mesmo que os dados 
ainda não tenham sido enviados. Assim, quando o fim da região de dados é atingido, a cópia do dispositivo
de `buf` pode ser desalocada antes que a biblioteca MPI tenha terminado de enviar os dados. Isso pode 
resultar em um travamento do aplicativo ou no prosseguimento do aplicativo, mas enviando valores de lixo. 
Para corrigir este problema, os desenvolvedores devem emitir um `MPI_Wait` antes do final da região de dados para garantir
que é seguro alterar ou desalocar o `buf`. Os exemplos abaixo demonstram como utilizar corretamente
`host_data` com chamadas MPI assíncronas.

~~~~ {.c .numberLines}
    #pragma acc data copyin(buf)
    { // Data in `buf` put on device
    #pragma acc host_data use_device(buf)
    { // Device pointer to `buf` passed to MPI
       MPI_Isend(buf, ..., request);
       // MPI_Isend immediatly returns to main thread
    }
    // Wait to ensure `buf` is safe to deallocate
    MPI_Wait(request, ...);
    } // Data in `buf` potentially removed from device
~~~~

~~~~ {.fortran .numberLines}
    !$acc data copyin(buf)
    ! Data in `buf` put on device
    !$acc host_data use_device(buf)
    ! Device pointer to `buf` passed to MPI
       call MPI_Isend(buf, ...)
       ! MPI_Isend immediatly returns to main thread
    !$acc end host_data
    ! Wait to ensure `buf` is safe to deallocate
    call MPI_Wait(request, ...)
    !$acc end data
    ! Data in `buf` potentially removed from device
~~~~

Utilização de Ponteiros de Dispositivo
---------------------
Como já existe um grande ecossistema de aplicações aceleradas que utilizam
linguagens como CUDA ou OpenCL, também pode ser necessário adicionar uma região OpenACC
a uma aplicação acelerada existente. Nesse caso, as matrizes podem ser
gerenciadas fora do OpenACC e já existirem no dispositivo. Para este caso o 
OpenACC fornece a cláusula de dados `deviceptr`, que pode ser usada onde qualquer cláusula de dados
pode aparecer. Esta cláusula informa ao compilador que as variáveis
especificadas já são disponíveis no dispositivo e nenhuma outra ação precisa ser
tomada sobre elas. O exemplo abaixo utiliza a função `acc_malloc`, que
aloca a memória do dispositivo e retorna um ponteiro, para alocar um array apenas no
dispositivo e então usa esse array dentro de uma região OpenACC.

~~~~ {.c .numberLines}
    void saxpy(int n, float a, float * restrict x, float * restrict y)
    {
      #pragma acc kernels deviceptr(x,y)
      {
        for(int i=0; i<n; i++)
        {
          y[i] += a*x[i];
        }
      }
    }
    void set(int n, float val, float * restrict arr)
    {
    #pragma acc kernels deviceptr(arr)
      {
        for(int i=0; i<n; i++)
        {
          arr[i] = val;
        }
      }
    }
    int main(int argc, char **argv)
    {
      float *x, *y, tmp;
      int n = 1<<20;
    
      x = acc_malloc((size_t)n*sizeof(float));
      y = acc_malloc((size_t)n*sizeof(float));
    
      set(n,1.0f,x);
      set(n,0.0f,y);
    
      saxpy(n, 2.0, x, y);
      acc_memcpy_from_device(&tmp,y,(size_t)sizeof(float));
      printf("%f\n",tmp);
      acc_free(x);
      acc_free(y);
      return 0;
    }
~~~~

---

~~~~ {.fortran .numberLines}
    module saxpy_mod
      contains
      subroutine saxpy(n, a, x, y)
        integer :: n
        real    :: a, x(:), y(:)
        !$acc parallel deviceptr(x,y)
        y(:) = y(:) + a * x(:)
        !$acc end parallel
      end subroutine
    end module
~~~~

Observe que nas rotinas `set` e `saxpy`, onde as regiões de computação OpenACC
são encontradas, cada região de computação é informada que os ponteiros
que estão sendo passados já são ponteiros de dispositivo usando a cláusula `deviceptr`. Este
exemplo também utiliza as rotinas `acc_malloc`, `acc_free`, e `acc_memcpy_from_device`
para gerenciamento de memória. Embora o exemplo acima utilize `acc_malloc`
e `acc_memcpy_from_device`, que são fornecidas pela especificação OpenACC
para gerenciamento de memória portátil, uma API específica do dispositivo também pode ter sido usada,
como `cudaMalloc` e `cudaMemcpy`.

Obtenção de Endereços de Ponteiro de Dispositivo e de Anfitrião
-------------------------------------------
O OpenACC fornece as chamadas de função `acc_deviceptr` e `acc_hostptr` para
obter os endereços de dispositivo e host de ponteiros baseados nos endereços de host e
endereços de dispositivo, respectivamente. Estas rotinas requerem que os endereços
realmente tenham endereços correspondentes, caso contrário, retornarão NULL.

~~~~ {.c .numberLines}
    double * x = (double*) malloc(N*sizeof(double));
    #pragma acc data create(x[:N])
    {
        double * device_x = (double*) acc_deviceptr(x);
        foo(device_x);
    }
~~~~

<!---
Mapping Arrays
--------------
***This is a pretty complicated thing to explain. Would anyone object to it
being left out?***
--->

Características Adicionais de Interoperabilidade Específicas do Fornecedor
----------------------------------------------------
A especificação OpenACC sugere várias características que são específicas de
fornecedores individuais. Embora as implementações não sejam obrigadas a fornecer a funcionalidade, é útil saber que esses recursos existem em algumas
implementações. O objetivo desses recursos é fornecer interoperabilidade
com o tempo de execução nativo de cada plataforma. Os desenvolvedores devem consultar a especificação
OpenACC e a documentação do seu compilador para obter uma lista completa de
recursos suportados.

### Filas Assíncronas e Fluxos CUDA (NVIDIA)
Como demonstrado no próximo capítulo, as filas de trabalho assíncronas são frequentemente uma
importante forma para lidar com o custo das transferências de dados PCIe em dispositivos com
memória distinta do host e do dispositivo. No modelo de programação NVIDIA CUDA
as operações assíncronas são programadas usando fluxos CUDA. Como os desenvolvedores podem
precisar interoperar entre fluxos CUDA e filas OpenACC, a especificação
sugere duas rotinas para mapear fluxos CUDA e filas assíncronas OpenACC.

A função `acc_get_cuda_stream` aceita um ID assíncrono inteiro e retorna um
objeto de fluxo CUDA (como um void\*) para uso como um fluxo CUDA.

A função `acc_set_cuda_stream` aceita um identificador assíncrono inteiro e um objeto de fluxo CUDA
(como um void\*) e mapeia o fluxo CUDA usado pelo identificador assíncrono
para o fluxo fornecido.

Com essas duas funções, é possível colocar operações OpenACC e operações
CUDA no mesmo fluxo CUDA subjacente para que elas sejam executadas
na ordem apropriada.

### Memória gerenciada CUDA (NVIDIA)
A NVIDIA adicionou suporte para *CUDA Managed Memory*, que fornece um único ponteiro
para a memória, independentemente de ser acessado do host ou do dispositivo, em CUDA
6.0. De muitas maneiras, a memória gerenciada é semelhante ao gerenciamento de memória OpenACC, em
que apenas uma única referência à memória é necessária e o tempo de execução lidará com as complexidades da movimentação de dados. A vantagem que a memória gerenciada
tem é que ela é mais capaz de lidar com estruturas de dados complexas, como classes C++ ou estruturas que contêm ponteiros, uma vez que as referências a ponteiros
são válidas tanto no host quanto no dispositivo. Mais informações sobre a memória gerenciada
CUDA podem ser obtidas na NVIDIA. Para usar a memória gerenciada em um programa OpenACC
o desenvolvedor pode simplesmente declarar ponteiros para a memória gerenciada como ponteiros de dispositivo
usando a cláusula `deviceptr` para que o tempo de execução do OpenACC não
tente criar uma alocação de dispositivo separada para os ponteiros. 

Também vale a pena observar que o compilador NVIDIA HPC (antigo compilador PGI)
tem suporte direto para usar a memória gerenciada CUDA por meio de uma opção de compilador.
Consulte a documentação do compilador para obter mais detalhes.

### Usando kernels de dispositivo CUDA (NVIDIA)
A diretiva `host_data` é útil para passar a memória do dispositivo para kernels CUDA invocáveis pelo host. Nos casos em que é necessário chamar um kernel de dispositivo (função CUDA
`__device__`) de dentro de uma região paralela OpenACC, é possível
usar a diretiva `acc routine` para informar ao compilador que a função que está sendo
chamada está disponível no dispositivo. A declaração da função deve ser decorada
com a diretiva `acc routine` e o nível de paralelismo no qual a
função pode ser chamada. No exemplo abaixo, a função `f1dev` é uma função seqüencial que será chamada de cada thread CUDA, então ela é declarada  `acc
routine seq`.

~~~~ {.cpp .numberLines}
    // Function implementation
    extern "C" __device__ void
    f1dev( float* a, float* b, int i ){
      a[i] = 2.0 * b[i];
    }
    
    // Function declaration
    #pragma acc routine seq
    extern "C" void f1dev( float*, float* int );
    
    // Function call-site
    #pragma acc parallel loop present( a[0:n], b[0:n] )
    for( int i = 0; i < n; ++i )
    {
      // f1dev is a __device__ function build with CUDA
      f1dev( a, b, i );
    }
~~~~
