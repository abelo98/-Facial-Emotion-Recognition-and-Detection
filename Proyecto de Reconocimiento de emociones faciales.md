## Proyecto de Reconocimiento de emociones faciales

### Resumen 

Hoy en día el uso de la inteligencia artificial para tareas de clasificación se ha vuelto algo viral. Esto se debe a que una máquina con todo su poder de cómputo es más perspicaz a la hora de detectar patrones casi irreconocibles para los humanos.

 La tarea de este proyecto trata sobre reconocer diferentes emociones a partir de un dataset formado por imágenes faciales.  Este trabajo explora el área del aprendizaje supervisado dentro del área de machine learning. Con el se pondrá en práctica el uso de redes neuronales y de visión por computadora para resolver el problema en cuestión, siendo capaz de detectar y aprender ciertos patrones regulares de las caras para posteriormente dar una estimación precisa de que emoción se está detectando.

### Redes Neuronales

Como hemos tratado en el curso sobre machine learning las redes neuronales representan una herramienta muy eficaz ante un problema de clasificación. La presentación de su idea a priori parece algo sencilla. Una capa encargada de recibir una entrada en forma de tensor, una serie de capas que ejecutan ciertas una operación algebraica sobre una entrada, que no es más que la salida de una capa anterior y finalmente un capa encargada del proceso de clasificación (en este caso por el problema que se trata, no necesariamente es para clasificar). Este proceso que hemos descrito es conocido como **forward**, donde el resultado de las activaciones de las **neuronas** es propagado hacia adelante, en la capa final con una función de error determinada se determina cuán distante está el resultado de nuestra red de la respuesta original y mediante la técnica de **back-propagation** se ajustan los pesos de la red neuronal. Esta última fase se realiza mediante el algoritmo de optimización llamado gradiente descendiente, donde un peso $w_i$ se actualiza de la siguiente manera $w_i =w_i -\alpha\frac{\partial E}{w_i}$. La idea detrás del gradiente descendiente es que el cálculo de las derivadas otorga un vector indicando la dirección de máximo de la función y si avanzamos una cantidad $\alpha$ en dirección contraria convergemos al mínimo de la función, lo cuál es nuestro objetivo, ya que se quiere minimizar la función $E$. Este proceso **forward & Back-propagation** se puede repetir innumerables ocasiones para alcanzar un resultado conveniente. Por supuesto aquí entran en cuestión otros factores como el **overfitting** o **underfitting** que provocan que entrenar mucho sobre los mismos datos o tener pocos datos se convierta en un problema.

### Dataset

El dataset en cuestión es **fer2013** 

### Modelo basado en redes convolucionales recurrentes

Para nuestro problema 
