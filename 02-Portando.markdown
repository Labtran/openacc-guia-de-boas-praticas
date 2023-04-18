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
