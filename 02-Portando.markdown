Acelerando uma aplicação com OpenACC
----------------------------------------
Esta secção irá detalhar uma abordagem incremental para acelerar uma aplicação
utilizando o OpenACC. Ao adoptar esta abordagem, é benéfico revisitar cada
passo várias vezes, verificando os resultados de cada passo para verificar se estão corretos. O trabalho
limitará progressivamente o âmbito de cada mudança para melhorar a produtividade e
depuração.

### Sintaxe da Directiva OpenACC ###
Este guia irá introduzir as directivas OpenACC de forma crescente, à medida que se tornam
útil para o processo de portabilidade. Todas as directivas OpenACC têm uma sintaxe comum,
no entanto, com o `ac` sentinela, designando para o compilador que o texto
que se segue será OpenACC, uma directiva, e cláusulas a essa directiva, muitas
dos quais são opcionais, mas fornecem ao compilador informações adicionais. 

Em C e C++, estas directivas assumem a forma de um pragmatismo. O código de exemplo
abaixo mostra a directiva OpenACC `kernels` sem quaisquer cláusulas adicionais

~~~~ {.c .numberLines}
    #pragma acc kernels
~~~~

Em Fortran, as diretivas assumem a forma de um comentário especial, como demonstrado
abaixo.

~~~~ {.fortran .numberLines}
    !$acc kernels
~~~~

Algumas directivas OpenACC aplicam-se a blocos estruturados de código, enquanto outras são
declarações executáveis. Em C e C++ um bloco de código pode ser representado por chaves (`{` e `}`). Em Fortran, um bloco de código começará com um
Directiva OpenACC (`!$acc kernels`) e termina com uma directiva final correspondente 
(`!$acc end kernels`).


### Ciclo de portabilidade ###
Os programadores devem adoptar uma abordagem incremental para acelerar as aplicações
utilizando o OpenACC para garantir a correção. Este guia seguirá a abordagem de
primeiro avaliando o desempenho das aplicações, depois utilizando o OpenACC para fazer o paralelismo
loops importantes no código, otimizando a próxima localidade de dados removendo
migrações de dados desnecessárias entre o hospedeiro e o acelerador, e finalmente
optimização de loops dentro do código para maximizar o desempenho de uma determinada
arquitetura. Esta abordagem tem sido bem sucedida em muitas aplicações, porque
prioriza as mudanças que são susceptíveis de proporcionar os maiores retornos, de modo que o programador pode atingir a aceleração de forma rápida e produtiva.

Há duas coisas importantes a registar antes de detalhar cada passo. Primeiro, em
vezes durante este processo o desempenho da aplicação pode realmente abrandar.
Os programadores não devem ficar frustrados se os seus esforços iniciais resultarem numa
perda de desempenho. Como será explicado mais tarde, este é geralmente o resultado
do movimento de dados implícito entre o hospedeiro e o acelerador, que será
optimizado como parte do ciclo de portabilidade. Em segundo lugar, é fundamental que
Os programadores verifiquem os resultados do programa para verificar se estão corretos após cada alteração.
As verificações de correção frequentes pouparão muito esforço de depuração, uma vez que os erros
podem ser encontrados e corrigidos imediatamente, antes de terem a oportunidade de se agravarem.
Alguns programadores podem achar benéfica a utilização de uma ferramenta de controle da versão para
fotografar o código após cada alteração bem sucedida para que qualquer alteração de ruptura possa
ser rapidamente removida e o código devolvido a um bom estado.

#### Avaliar o desempenho da aplicação ####
Antes de se poder começar a acelerar uma aplicação, é importante compreender
em que rotinas e loops uma aplicação está gastando a maior parte do seu tempo e
porquê. É fundamental compreender as partes mais demoradas da
aplicação para maximizar o benefício da aceleração. A Lei de Amdahl
informa-nos que o speed-up atingível com a execução de uma aplicação num
acelerador paralelo será limitado pelo código serial restante. Em outras
palavras, a aplicação verá o maior benefício ao acelerar a maior parte do
código o mais possível e dando prioridade às partes que consomem mais tempo. Uma variedade
de ferramentas pode ser utilizado para identificar partes importantes do código, incluindo simples
temporizadores de aplicação.

#### Paralelizando Laços ####
Uma vez identificadas as regiões importantes do código, as directivas OpenACC
devem ser utilizadas para acelerar estas regiões no dispositivo alvo. Laços paralelos
dentro do código devem ser decorados com directivas OpenACC para fornecer ao compilador OpenACC a informação necessária para paralelizar o código para a arquitetura alvo.

#### Optimizar a localização de dados ####
Porque muitas arquiteturas aceleradas, tais como CPU + arquiteturas GPU, utilizam
espaços de memória distintos para o *host* e o *dispositivo*, é necessário para o
compilador gerir os dados em ambas as memórias e mover os dados entre as duas
memórias para garantir resultados corretos. Os compiladores raramente têm pleno conhecimento da
aplicação, pelo que devem ser cautelosos, a fim de garantir a correcção, que
envolve frequentemente a cópia de dados de e para o acelerador mais frequentemente do que é
realmente necessário. O programador pode dar ao compilador informações adicionais
sobre como gerir a memória para que se mantenha local ao acelerador como
enquanto for possível e só se move entre as duas memórias quando absolutamente
necessário. Os programadores realizarão frequentemente os maiores ganhos de desempenho após
optimizar a movimentação de dados durante esta etapa.

#### Optimize Laços ####
Os compiladores tomarão decisões sobre a forma de mapear o paralelismo no código para
o acelerador alvo baseado na heurística interna e no conhecimento limitado
que tem sobre a aplicação. Por vezes, é possível obter um desempenho adicional fornecendo ao compilador mais informações para que este possa tomar melhores
decisões sobre como mapear o paralelismo com o acelerador. Quando se vem de um
arquitetura tradicional de CPU para uma arquitetura mais paralela, como uma GPU, ela
também pode ser necessário reestruturar os laços para expor o paralelismo adicional para
o acelerador ou para reduzir a frequência do movimento de dados. Codificar frequentemente
refatorando o que foi motivado pela melhoria do desempenho em paralelo
é benéfico também para as CPUs tradicionais.

---

Este processo não é de forma alguma a única forma de acelerar utilizando o OpenACC, mas
tem sido comprovadamente bem sucedida em numerosas aplicações. Fazendo os mesmos passos em
diferentes ordens podem causar tanto frustração como dificuldade de depuração, por isso é
aconselhável executar cada etapa do processo na ordem indicada acima.

### Melhores Práticas de Computação Heterogénea ###
Muitas aplicações têm sido escritas com pouco ou mesmo nenhum paralelismo exposto
no código. As aplicações que expõem o paralelismo fazem-no frequentemente de uma forma grosseira, onde um pequeno número de threads ou processos são executados por
muito tempo e calculando uma quantidade significativa de trabalho cada um. GPUs modernos e processadores de muitos núcleos
no entanto, são concebidos para executar threads de granulometria fina, que são
de curta duração e executam uma quantidade mínima de trabalho cada um. Essas arquiteturas paralelas alcançam alto rendimento negociando desempenho de thread único em favor de várias ordens de magnitude de mais paralelismo. Isto significa que quando
acelerando uma aplicação com OpenACC, que foi concebida à luz de 
maior paralelismo de hardware, pode ser necessário refatorar o código para
favorecer os laços estreitamente aninhados com uma quantidade significativa de reutilização de dados. Em muitos
casos, estas mesmas alterações de código beneficiam também arquiteturas de CPU mais tradicionais que
bem, melhorando a utilização da cache e a vetorização.

O OpenACC pode ser utilizado para acelerar aplicações em dispositivos que tenham uma discreta
memória ou que têm um espaço de memória que é partilhado com o host. Mesmo em dispositivos
que utilizam uma memória partilhada, ainda existe frequentemente uma hierarquia da rapidez,
memória estreita para o acelerador e uma memória maior e mais lenta utilizada pelo host.
Por este motivo, é importante estruturar o código da aplicação para maximizar a
reutilização de matrizes, independentemente de a arquitetura subjacente utilizar ou não arquitetura discreta
ou memórias unificadas. Ao refatorar o código para utilização com OpenACC, é
frequentemente benéfico assumir uma memória discreta, mesmo que o dispositivo onde esteja
desenvolvendo tenha uma memória unificada. Isto força a localidade de dados a ser um local primário
na refatoração e assegurará que o código resultante
explora memórias hierárquicas e é portátil para uma vasta gama de dispositivos.

Estudo de caso - Jacobi Iteration
---------------------------------
Ao longo deste guia iremos utilizar aplicações simples para demonstrar cada passo
do processo de aceleração. A primeira aplicação deste tipo resolverá a
equação 2D-Laplace com o solucionador iterativo Jacobi. Os métodos iterativos são uma
técnica comum para aproximar a solução dos PDE elípticos, como a
equação 2D-Laplace, dentro de alguma tolerância permitida. No caso do nosso
exemplo, faremos um cálculo simples onde cada ponto
calcula o seu valor como a média dos valores dos seus vizinhos. O cálculo irá
continuar a iterar até que a alteração máxima no valor entre duas
iterações desça abaixo de algum nível de tolerância ou o número máximo de iterações
é alcançado. Para uma comparação consistente através do documento,
os exemplos irão sempre iterar 1000 vezes. O laço de iteração principal para ambos C/C++
e Fortran aparece abaixo.

~~~~ {.c .numberLines startFrom="52"}
    while ( error > tol && iter < iter_max )
    {
        error = 0.0;

        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                Anew[j][i] = 0.25 * ( A[j][i+1] + A[j][i-1]
                                    + A[j-1][i] + A[j+1][i]);
                error = fmax( error, fabs(Anew[j][i] - A[j][i]));
            }
        }

        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }
~~~~    

---

~~~~ {.fortran .numberLines startFrom="52"}
    do while ( error .gt. tol .and. iter .lt. iter_max )
      error=0.0_fp_kind
  
      do j=1,m-2
        do i=1,n-2
          Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
                                       A(i  ,j-1) + A(i  ,j+1) )
          error = max( error, abs(Anew(i,j)-A(i,j)) )
        end do
      end do
  
      do j=1,m-2
        do i=1,n-2
          A(i,j) = Anew(i,j)
        end do
      end do
  
      if(mod(iter,100).eq.0 ) write(*,'(i5,f10.6)'), iter, error
      iter = iter + 1
  
    end do
~~~~

O laço mais externo em cada exemplo será referido como a *o laço de convergência*, uma vez que faz loops até que a resposta tenha convergido, atingindo um máximo de
tolerância ao erro ou número de iterações. Note-se que quer se trate ou não de um laço
a iteração ocorre dependendo do valor de erro da iteração anterior. Também,
os valores para cada elemento de `A` são calculados com base nos valores da
iteração anterior, conhecida como dependência de dados. Estes dois factos significam que este
laço não pode ser executados em paralelo.

O primeiro laço aninhado dentro do laço de convergência calcula o novo valor para
cada elemento com base nos valores atuais dos seus vizinhos. Note-se que é
necessário para armazenar este novo valor num conjunto diferente. Se cada iteração
armazena o novo valor de volta a si mesmo, então existiria uma dependência de dados entre
os elementos de dados, uma vez que a ordem em que cada elemento é calculado afectaria o
resposta final. Ao armazenarmos numa matriz temporária, asseguramos que todos os valores são
calculado utilizando o estado actual de `A` antes de `A` ser actualizado. Como resultado,
cada iteração de laço é completamente independente uma da outra iteração. Estas iterações de laço podem ser executadas em segurança em qualquer ordem ou em paralelo e ao final
o resultado seria o mesmo. Este laço também calcula um valor máximo de erro. O valor de erro é a diferença entre o novo valor e o antigo. Se o valor máximo de
a quantidade de mudança entre duas iterações está dentro de alguma tolerância, o problema
é considerado convergente e o laço exterior sairá.

O segundo laço aninhado simplesmente atualiza o valor de `A` com os valores calculados
em `Anew`. Se esta for a última iteração do ciclo de convergência, `A` será
o valor final, convergente. Se o problema ainda não tiver convergido, então `A` irá
servir como entrada para a próxima iteração. Tal como com o laço aninhado a acima, cada
iteração deste laço aninhado é independente um do outro e é seguro para
paralelizar.

Nas próximas secções iremos acelerar esta simples aplicação utilizando o
método descrito no presente documento.
