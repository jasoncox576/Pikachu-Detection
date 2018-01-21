# Pikachu-Detection
A use of two dimensional cross-correlation in order to find a point of maximum similarity and detect one or multiple pikachu amongst several different pokemon

Another demonstration of the handiness of cross-correlation in finding similarities between images. This seemed like an easy enough project provided that one could properly handle the image processing and figure out how to detect multiple instances of the template (Pikachu) within the image. The actual cross-correlation algorithm is deceptively simple to implement given the spooky-looking equation (https://en.wikipedia.org/wiki/Cross-correlation). Using the numpy function correlate2d proved to be rather unfruitful probably because the output size ended up being different, in all cases, than the size of the image. Another setback in implementing a correlation algorithm independently was the unusual occurence that, in multiplying the values of the array together, the parts of the image that contained the most white space ended up with the highest correlation, erroniously. Discovered through the visual aid of 2D heatmaps, it only made sense that this was because multiplying two numbers such as 150 * 255 will of course return higher than multiplying 150 * 150. Thus, a Z-Normalization was used to normalize the pixels to a proper range of [-1, 1]. This displayed immediate success, the reason being that if a positive number is multiplied by a (not so similar) pixel with a negative value, it will detract from the correlation.  
  

To start, it made sense to try this out with a simple image that shows a pikachu in the middle of a white background, and nothing else. Here are the 2D and 3D plots of the correlations returned:
![simple2d](https://user-images.githubusercontent.com/14042582/35189917-cee1afb6-fe1a-11e7-81fb-0987da29a4f7.png)
![simple3d](https://user-images.githubusercontent.com/14042582/35189918-d23d2dfc-fe1a-11e7-8050-ec3c8313a629.png)
In my humble opinion it looks like something out of Mordor.  


Next, it made sense to try something like moving pikachu to the top-left corner of the image just for a sanity-check. Here are the results.
![leftcorner2d](https://user-images.githubusercontent.com/14042582/35190220-17d4127a-fe22-11e7-82ff-93ddda944f30.png)
![leftcorner3d](https://user-images.githubusercontent.com/14042582/35190221-17e65412-fe22-11e7-8ad9-dc7f54a7cbf6.png)


  
And here is the mapping of the correlation values of pikachu.bmp over image.bmp, which has three pikachus scattered amongst a bunch of other pokemon.
![image2d](https://user-images.githubusercontent.com/14042582/35189853-56154e7c-fe19-11e7-9419-f276d19407bc.png)
Of course, the 3D topographical map was a sight to behold...
![image3d](https://user-images.githubusercontent.com/14042582/35189854-5b2af2c2-fe19-11e7-9e51-22997df64e98.png)
One can clearly see three distinct maxes, representing the three pikachus shown in the image,
![out](https://user-images.githubusercontent.com/14042582/35190302-8f964570-fe23-11e7-90e9-e3806f297768.png)
As well as the bounding boxes which wrap around in the correct areas as shown in the output image.  
  
<br />   
<br />
<br />
    
Finally, the results of correlation over nopikachu.png, which contained a charmander, squirtle and bulbasaur but no pikachu. Unsurprisingly no major peaks can be seen and the point of max correlation is somewhere in between charmander and bulbasaur.
![nopikachu2d](https://user-images.githubusercontent.com/14042582/35190324-114e3262-fe24-11e7-88a1-9692aec3b5a2.png)
![nopikachu3d](https://user-images.githubusercontent.com/14042582/35190325-11608aac-fe24-11e7-8889-8a0449fddfec.png)
![out](https://user-images.githubusercontent.com/14042582/35190341-6fd703c2-fe24-11e7-803a-8ce20d896261.png)

<br />
Out of curiosity it seemed a good idea to test the efficacy of the same algorithm with a blurred image. A mean blur was used, in which the values of every pixel were averaged with all of the pixel values [1-X] units away, X being the kernel size. For this particular test, a blur kernel size of 3x3 was used.

![imageblur2d](https://user-images.githubusercontent.com/14042582/35190383-b4d92f62-fe25-11e7-84c6-9d4c0f7ad304.png)
![imageblur3d](https://user-images.githubusercontent.com/14042582/35190384-b4eab494-fe25-11e7-857b-976ab72752e2.png)
It looks like it did a fair bit of damage. In the output file, it is apparent that neither of the other two pikachus passed the correlation threshold required to be considered another max (99%). Upon turning it down to ~91%, all three are caught. Not a major problem but one big enough to warrant taking note of.
<br />
![out](https://user-images.githubusercontent.com/14042582/35190381-aef50c88-fe25-11e7-9461-12d47d7d355d.png)
<br />
As the size of the blur kernel increases, the blur becomes more intense and cross correlation begins to fall apart altogether. At that point, it is likely that the problem has shifted from the domain of cross correlation- a simple yet effective technique applicable only in certain domains - to something like convolutional neural networks, which prove more effective for extreme image distortions.

<br />
I hope you enjoyed reading this writeup, now you can clone this repository and find pikachu for yourself!
