# Calculate Sobel with CUDA

- number of thread per block: 256
- dims of grid: (width/ 256 + 1, height)
- every thread is responsible for calculating a pixel

Before | After
:-------------------------:|:-------------------------:
![](/lab3/candy.png)  |  ![](/lab3/candy.out.png)