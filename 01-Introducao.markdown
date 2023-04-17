Introdução
============
Este guia apresenta métodos e melhores práticas para acelerar as aplicações
de uma forma incremental e portátil de desempenho. Embora alguns dos exemplos possam
mostrar resultados utilizando um determinado compilador ou acelerador, a informação apresentada
neste documento destina-se a abordar todas as arquitecturas, ambas disponíveis em
tempo de publicação e bem como para o futuro. Os leitores devem estar à vontade com
C, C++, ou Fortran, mas não necessitam de experiência com programação paralela ou
computação acelerada, embora tal experiência seja útil.

Note: This guide is a community effort. To contribute, please visit the project
[on Github](https://github.com/OpenACC/openacc-best-practices-guide).

Escrevendo um Código Portátil
---------------------
O atual panorama é visto com uma variedade de arquitecturas de computação: CPUs multi-core, GPUs, dispositivos multi-core, DSPs, processadores ARM, e FPGAs, para citar alguns. É agora comum encontrar não apenas uma, mas várias destas
arquiteturas diferentes dentro da mesma máquina. Os programadores devem fazer
a portabilidade do seu código, caso contrário arriscam-se a bloquear a sua
aplicação a uma única arquitetura, o que pode limitar a capacidade de rodar em
arquiteturas futuras. Embora a variedade de arquiteturas possa parecer assustadora
para o programador, uma análise mais atenta revela tendências que mostram muito em comum
entre elas. A primeira coisa a notar é que todas estas arquitecturas estão
movendo-se no sentido de um maior paralelismo. As CPUs não estão apenas adicionando núcleos de CPU
mas também expandindo a duração das suas operações SIMD. As GPUs têm crescido para um elevado grau de paralelismo de blocos e SIMT. É evidente que ir através de
todas as arquiteturas necessita de um grau significativo de paralelismo
a fim de alcançar um alto desempenho. Os processadores modernos não precisam apenas de uma grande
quantidade de paralelismo, mas expõe frequentemente múltiplos níveis de paralelismo
com diferentes graus de dificuldade. A próxima coisa a notar é que todas
estas arquiteturas têm exposto hierarquias de memória. As CPUs têm a principal
memória do sistema, tipicamente DDR, e múltiplas camadas de memória cache. As GPUs têm a memória principal da CPU, a memória principal da GPU, e vários graus de memória cache ou scratchpad. Adicionalmente em arquiteturas híbridas, que incluem duas ou mais arquiteturas diferentes, existem máquinas onde as duas arquiteturas têm memórias completamente separadas, algumas com separação física mas logicamente a mesma memória, e outras com memória totalmente partilhada.

Devido a estas complexidades, é importante que os criadores escolham um
modelo de programação que equilibra a necessidade de portabilidade com a necessidade de
desempenho. Abaixo estão quatro modelos de programação variando em diferentes graus de
portabilidade e desempenho. Numa aplicação real, o melhor é utilizar frequentemente
uma mistura de abordagens para assegurar um bom equilíbrio entre a elevada portabilidade e
desempenho.

### Bibliotecas ###

As bibliotecas standard (e *de facto* standard) fornecem o mais alto grau de
portabilidade porque o programador pode frequentemente substituir apenas a biblioteca
utilizada sem sequer alterar o próprio código fonte quando altera a arquitetura. Uma vez que muitos vendedores de hardware fornecem versões altamente customizadas de
bibliotecas comuns, o uso de bibliotecas também pode resultar num desempenho muito elevado.
Embora as bibliotecas possam fornecer tanto alta portabilidade como alto desempenho, poucas
são as aplicações só podem utilizar bibliotecas devido ao seu âmbito limitado.
    
Alguns vendedores fornecem bibliotecas adicionais como um valor adicional para sua plataforma
mas implementam APIs não-padronizados. Estas bibliotecas fornecem
alto desempenho, mas pouca portabilidade. Felizmente, porque as bibliotecas fornecem
APIs modulares, o impacto da utilização de bibliotecas não-portáteis pode ser isolado para
limitar o impacto na portabilidade global da aplicação.

### Linguagens de Programação Standard ###

Muitas linguagens de programação padrão têm ou estão  a adotar
características para programação paralela. Por exemplo, a Fortran 2008 acrescentou apoio
para `do concurrent`, o que expõe o potencial paralelismo com esse laço,
e C++17 acrescentou apoio para `std::execution`, que permite aos utilizadores exprimir
paralelismo com certas estruturas de laço.
A adoção destas características de linguagem é frequentemente lenta, e muitas linguagens padrão
só agora começam a discutir as características de programação paralela para futuros
lançamentos. Quando estas características se tornarem comuns, fornecerão
portabilidade, uma vez que fazem parte de uma linguagem padrão, e se forem bem concebidas
também pode proporcionar alto desempenho.

### Diretivas do Compilador ###

Quando as linguagens de programação padrão não têm suporte para as características necessárias as
diretivas de compilação podem fornecer funcionalidades adicionais. As diretivas, na
forma de pragmas em C/C++ e comentários em Fortran, fornecem mais
informação aos compiladores sobre como construir e/ou optimizar o código. A maioria dos
compiladores apoiam as suas próprias diretivas, bem como diretivas como a OpenACC e
OpenMP, que são apoiados por grupos industriais e implementados por uma gama de
compiladores. Ao utilizar diretivas de compilação apoiadas pela indústria, o programador pode
escrever código com um elevado grau de portabilidade entre compiladores e
arquiteturas. No entanto, frequentemente, estas diretivas de compilação são escritas para
permanecerem em um nível muito elevado, tanto pela simplicidade como pela portabilidade, o que significa que
o desempenho pode atrasar paradigmas de programação de nível inferior. Muitos programadores estão
dispostos a abdicar de 10-20% do desempenho para obter um alto
grau de portabilidade para outras arquiteturas e para melhorar a produtividade. A tolerância para este compromisso de portabilidade/desempenho irá
variar de acordo com as necessidades do programador e da aplicação.

### Extensôes para Programação Paralela ###

CUDA e OpenCL são exemplos de extensões a linguagens de programação existentes
para dar capacidades adicionais de programação paralela. O código escrito nestas
linguagens está frequentemente a um nível inferior ao de outras opções, mas o resultado
pode frequentemente alcançar um desempenho superior. No nível mais baixo
os detalhes são expostos e a forma como um problema é decomposto no hardware
deve ser explicitamente gerido com essas linguagens. Esta é a melhor opção quando
os objetivos de desempenho superam a portabilidade, uma vez que a natureza de baixo nível das
linguagens de programação tornam frequentemente o código resultante menos portátil. Boas práticas de engenharia de software podem reduzir o impacto que estas linguagens têm sobre
portabilidade.

----

Não há um modelo de programação que se adapte a todas as necessidades. Um programador de aplicações
precisa avaliar as prioridades do projeto e tomar decisões em conformidade.
Uma boa prática é começar com a programação mais portátil e produtiva e passar para modelos de programação de nível inferior apenas quando necessário e 
de uma forma modular. Ao fazê-lo, o programador pode acelerar grande parte da
aplicação muito rapidamente, o que muitas vezes é mais benéfico do que tentar obter
o mais alto desempenho de uma determinada rotina antes de passar para
o próximo. Quando o tempo de desenvolvimento é limitado, concentrar-se em acelerar o máximo 
da aplicação da forma mais produtiva possível é geralmente mais produtiva do que concentrar apenas
na rotina que consome mais tempo.

O que é OpenACC?
----------------
Com o surgimento da GPU e de arquiteturas de muitos núcleos de alta performance
computacional, os programadores desejam a capacidade de programar usando um
modelo de programação de alto nível que proporcione tanto alto desempenho como portabilidade para
uma vasta gama de arquiteturas computacionais. O OpenACC surgiu em 2011 como um
modelo de programação que utiliza diretivas de compilação de alto nível para expor
paralelismo no código e paralelização de compiladores para construir o código para um
variedade de aceleradores paralelos. Este documento pretende ser uma boa prática e um
guia para acelerar uma aplicação usando OpenACC para dar a ambos um bom
desempenho e portabilidade a outros dispositivos.

### O Modelo de Aceleração OpenACC ###
A fim de assegurar que o OpenACC seria portátil para todos os computadores
e arquitecturas disponíveis no momento do seu início e no futuro,
O OpenACC define um modelo abstracto para computação acelerada. Este modelo expõe
múltiplos níveis de paralelismo que podem aparecer num processador, bem como numa
hierarquia de memórias com diferentes graus de velocidade e endereçamento. O
objectivo deste modelo é assegurar que o OpenACC seja aplicável a mais do que apenas uma
arquitetura em particular ou mesmo apenas as arquiteturas em ampla disponibilidade no momento, mas para assegurar que o OpenACC também possa ser utilizado em futuros dispositivos. 

No seu núcleo, o OpenACC suporta o descarregamento tanto de dados de cálculo como de dados de um
*host* dispositivo a um *accelerator* dispositivo. Na verdade, estes dispositivos podem ser os
mesmos ou podem ser arquiteturas completamente diferentes, tais como o caso de uma CPU
hospedeiro e acelerador GPU. Os dois dispositivos podem também ter espaços de memória separados
ou um único espaço de memória. No caso de os dois dispositivos terem diferentes
memórias o compilador OpenACC e o runtime analisarão o código e manipularão qualquer
gestão da memória do acelerador e a transferência de dados entre o hospedeiro e a
memória do dispositivo. A figura 1.1 mostra um diagrama de alto nível do resumo do acelerador OpenACC, mas lembre-se que os dispositivos e memórias podem ser fisicamente o
o mesmo em algumas arquitecturas.

![OpenACC's Abstract Accelerator Model](images/execution_model2.png)

Mais detalhes do modelo de acelerador abstracto do OpenACC serão apresentados
ao longo deste guia, quando forem pertinentes. 

----

***Boas Práticas:*** Para programadores que chegam ao OpenACC a partir de outros
modelos de programação acelerada, tais como CUDA ou OpenCL, onde a memória do anfitrião e do acelerador
é frequentemente representada por duas variáveis distintas (`host_A[]` e
`device_A[]`, por exemplo), é importante lembrar que ao utilizar o OpenACC
uma variável deve ser pensada como um único objecto, independentemente de como
é suportado pela memória em um ou mais espaços de memória. 

Se se assumir que representa duas memórias separadas, dependendo de onde é utilizada no programa,
então é possível escrever programas que acessem a variável de
formas inseguras, resultando em códigos que não seriam portáteis para dispositivos que partilham
uma única memória entre o hospedeiro e o dispositivo. Como em qualquer programa paralelo ou
paradigma de programação assíncrona, acedendo à mesma variável a partir de duas
secções de código podem resultar simultaneamente numa race condition que produza
resultados inconsistentes. 

Assumindo que se está sempre acessando a uma única variável ,independentemente de como é armazenada na memória, o programador irá evitar
cometer erros que poderiam custar um esforço significativo para depurar.

### Benefícios e Limitações do OpenACC ###
O OpenACC foi concebido para ser uma linguagem de alto nível e independente de plataforma para
aceleradores de programação. Como tal, é possível desenvolver um único código fonte que
pode ser executado numa gama de dispositivos e alcançar um bom desempenho. A simplicidade
e a portabilidade que o modelo de programação do OpenACC proporciona, por vezes
custo para o desempenho. O modelo de acelerador abstracto OpenACC define um mínimo
denominador comum para dispositivos aceleradores, mas não pode representar arquitetura
específicas destes dispositivos sem tornar a linguagem menos portátil. Aí
serão sempre algumas optimizações que são possíveis num nível inferior de modelo de programação, como a CUDA ou OpenCL, que não pode ser representada a um nível
alto. Por exemplo, embora o OpenACC tenha a directiva "cache", algumas utilizações de
*memória partilhada* em GPUs NVIDIA são mais facilmente representadas usando CUDA. O mesmo
é verdade para qualquer anfitrião ou dispositivo: certas optimizações são de nível demasiado baixo para um
abordagem de alto nível como o OpenACC. Cabe aos programadores determinar a
custo e benefício da utilização selectiva de uma linguagem de programação de nível inferior para
secções críticas do código de desempenho. Nos casos em que o desempenho é demasiado
crítico para adoptar uma abordagem de alto nível, ainda é possível utilizar o OpenACC para
grande parte da aplicação, enquanto utiliza outra abordagem em certos lugares, como
será discutido num capítulo posterior sobre interoperabilidade.
