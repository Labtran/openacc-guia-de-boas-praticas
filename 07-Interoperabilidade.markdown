OpenACC Interoperabilidade 
Os autores do OpenACC reconheceram que, por vezes, pode ser benéfico misturar código OpenACC com código acelerado utilizando outras linguagens de programação paralela, como CUDA ou OpenCL, ou bibliotecas matemáticas aceleradas. Essa interoperabilidade significa que um desenvolvedor pode escolher o paradigma de programação que faz mais sentido em uma situação específica e aproveitar o código e as bibliotecas que já podem estar disponíveis. Os programadores não precisam decidir no início de um projeto entre o OpenACC e outra coisa qualquer, podem optar por utilizar o OpenACC e outras tecnologias.

NOTE: The examples used in this chapter can be found online at https://github.com/jefflarkin/openacc-interoperability

A Região de Dados do Host
O primeiro método para interoperar entre o OpenACC e algum outro código é gerenciar todos os dados usando o OpenACC, mas chamando uma função que requer dados do dispositivo. Para o propósito do exemplo, a rotina cublasSaxpy será usada no lugar de escrever uma rotina saxpy, como foi mostrado num capítulo anterior. Esta rotina é fornecida gratuitamente pela NVIDIA para o seu hardware na biblioteca CUBLAS. A maior parte dos outros fabricantes fornecem a sua própria biblioteca, ajustada.

A directiva host_data dá ao programador uma forma de expor o endereço do dispositivo de uma determinada matriz ao host para passar para uma função. Esses dados já devem ter sido movidos para o dispositivo anteriormente. O nome desta construção confunde frequentemente os novos utilizadores, mas pode ser considerada como uma região de dados inversa, uma vez que pega nos dados do dispositivo e os expõe ao host. A região host_data aceita apenas a cláusula use_device, que especifica quais variáveis do dispositivo devem ser expostas ao host. No exemplo abaixo, os vetores x e y são colocados no dispositivo por meio de uma região de dados e, em seguida, inicializados em um loop OpenACC. Essas matrizes são então passadas para a função cublasSaxpy como ponteiros de dispositivo usando a região host_data.

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
A chamada para cublasSaxpy pode ser alterada para qualquer função que espere ponteiros de dispositivos como parâmetros.

Bibliotecas de Dispositivos Assíncronos
NOTA: Ao utilizar a região host_data para passar dados para chamadas de bibliotecas assíncronas ou kernels, é necessário ter cuidado com o tempo de vida dos dados no dispositivo. Um exemplo comum deste padrão é a passagem de dados do dispositivo para uma biblioteca MPI sensível ao dispositivo, como ilustrado abaixo.

Um uso comum da região host_data é passar ponteiros de dispositivo para uma implementação MPI com reconhecimento de dispositivo. Essas bibliotecas MPI podem ter optimizações específicas quando passam dados de dispositivo, tais como Remote Direct Memory Access (RDMA) ou pipelining. Para rotinas MPI síncronas, a diretiva host_data pode ser usada como mostrado acima, mas é preciso ter cuidado ao misturar essa diretiva com funções MPI assíncronas (por exemplo, MPI_ISend, MPI_IRecv, etc.). Veja, por exemplo, o código a seguir:

    #pragma acc data copyin(buf)
    { // Data in `buf` put on device
    #pragma acc host_data use_device(buf)
    { // Device pointer to `buf` passed to MPI
       MPI_Isend(buf, ...);
       // MPI_Isend immediatly returns to main thread
    }
    // MPI_Isend may not have completed sending data
    } // Data in `buf` potentially removed from device
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
No exemplo acima, o ponteiro de dispositivo para os dados em buf é fornecido a MPI_ISend, que devolverá imediatamente o controle à linha de execução, mesmo que os dados ainda não tenham sido enviados. Assim, quando o fim da região de dados é alcançado, a cópia de dispositivo de buf pode ser desalocada antes que a biblioteca MPI tenha terminado de enviar os dados. Isso pode resultar em um travamento do aplicativo ou no prosseguimento do aplicativo, mas com o envio de valores de lixo. Para corrigir esse problema, os desenvolvedores devem emitir um MPI_Wait antes do final da região de dados para garantir que é seguro alterar ou desalocar buf. Os exemplos abaixo demonstram como usar corretamente host_data com chamadas MPI assíncronas.

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
Utilização de Ponteiros de Dispositivo
Como já existe um grande ecossistema de aplicações aceleradas que utilizam linguagens como CUDA ou OpenCL, também pode ser necessário adicionar uma região OpenACC a uma aplicação acelerada existente. Neste caso, as matrizes podem ser gerenciadas fora do OpenACC e já existem no dispositivo. Para este caso, o OpenACC fornece a cláusula deviceptr data, que pode ser usada onde qualquer cláusula data pode aparecer. Esta cláusula informa o compilador de que as variáveis especificadas já estão no dispositivo e que não é necessário realizar qualquer outra ação sobre elas. O exemplo abaixo usa a função acc_malloc, que aloca memória de dispositivo e retorna um ponteiro, para alocar uma matriz somente no dispositivo e então usa essa matriz dentro de uma região OpenACC.

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
Observe que nas rotinas set e saxpy, onde as regiões de computação OpenACC são encontradas, cada região de computação é informada de que os ponteiros que estão sendo passados já são ponteiros de dispositivo usando a cláusula deviceptr. Este exemplo também usa as rotinas acc_malloc, acc_free e acc_memcpy_from_device para gerenciamento de memória. Embora o exemplo acima use acc_malloc e acc_memcpy_from_device, que são fornecidos pela especificação OpenACC para gerenciamento de memória portátil, uma API específica de dispositivo também pode ter sido usada, como cudaMalloc e cudaMemcpy.

Obtenção de endereços de ponteiro do dispositivo e do host
OpenACC provides the acc_deviceptr and acc_hostptr function calls for obtaining the device and host addresses of pointers based on the host and device addresses, respectively. These routines require that the addresses actually have corresponding addresses, otherwise they will return NULL.

    double * x = (double*) malloc(N*sizeof(double));
    #pragma acc data create(x[:N])
    {
        double * device_x = (double*) acc_deviceptr(x);
        foo(device_x);
    }
Características Adicionais de Interoperabilidade Específicas do fornecedor
A especificação OpenACC sugere vários recursos que são específicos de fornecedores individuais. Embora as implementações não sejam obrigadas a fornecer a funcionalidade, é útil saber que esses recursos existem em algumas implementações. O objectivo destas características é proporcionar interoperabilidade com o tempo de execução nativo de cada plataforma. Os programadores devem consultar a especificação OpenACC e a documentação do seu compilador para obter uma lista completa das funcionalidades suportadas.

Filas assíncronas e fluxos CUDA (NVIDIA)
Conforme demonstrado no próximo capítulo, as filas de trabalho assíncronas são frequentemente uma maneira importante de lidar com o custo das transferências de dados PCIe em dispositivos com memória de host e de dispositivo distintas. No modelo de programação NVIDIA CUDA, as operações assíncronas são programadas usando fluxos CUDA. Como os desenvolvedores podem precisar interoperar entre fluxos CUDA e filas OpenACC, a especificação sugere duas rotinas para mapear fluxos CUDA e filas assíncronas OpenACC.

A função acc_get_cuda_stream aceita um ID assíncrono inteiro e devolve um objecto de fluxo CUDA (como um void*) para utilização como um fluxo CUDA.

A função acc_set_cuda_stream aceita um identificador assíncrono inteiro e um objecto de fluxo CUDA (como um void*) e mapeia o fluxo CUDA utilizado pelo identificador assíncrono para o fluxo fornecido.

Com estas duas funções, é possível colocar operações OpenACC e operações CUDA no mesmo fluxo CUDA subjacente para que sejam executadas na ordem correcta.

Memória gerenciada pelo CUDA (NVIDIA)
A NVIDIA adicionou suporte à memória gerenciada CUDA, que fornece um único ponteiro para a memória, independentemente de ela ser acessada do host ou do dispositivo, na CUDA 6.0. Em muitos aspectos, a memória gerenciada é semelhante ao gerenciamento de memória OpenACC, já que apenas uma única referência à memória é necessária e o tempo de execução lidará com as complexidades da movimentação de dados. A vantagem que a memória gerenciada às vezes tem é que ela é mais capaz de lidar com estruturas de dados complexas, como classes C++ ou estruturas contendo ponteiros, já que as referências de ponteiro são válidas tanto no host quanto no dispositivo. Mais informações sobre a memória gerenciada CUDA podem ser obtidas na NVIDIA. Para usar a memória gerenciada em um programa OpenACC, o desenvolvedor pode simplesmente declarar ponteiros para a memória gerenciada como ponteiros de dispositivo usando a cláusula deviceptr para que o tempo de execução do OpenACC não tente criar uma alocação de dispositivo separada para os ponteiros.

Também vale a pena observar que o compilador NVIDIA HPC (antigo compilador PGI) tem suporte direto para o uso da memória gerenciada CUDA por meio de uma opção de compilador. Consulte a documentação do compilador para obter mais detalhes.

Usando kernels de dispositivo CUDA (NVIDIA)
A diretiva host_data é útil para passar a memória do dispositivo para kernels CUDA que podem ser chamados pelo host. Nos casos em que é necessário chamar um kernel de dispositivo (função CUDA __device__) de dentro de uma região paralela OpenACC, é possível usar a diretiva de rotina acc para informar ao compilador que a função que está sendo chamada está disponível no dispositivo. A declaração da função deve ser decorada com a directiva de rotina acc e o nível de paralelismo em que a função pode ser chamada. No exemplo abaixo, a função f1dev é uma função sequencial que será chamada a partir de cada thread CUDA, portanto é declarada como acc routine seq.

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
