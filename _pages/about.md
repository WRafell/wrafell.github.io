---
permalink: /
layout: single
title: "About"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html

---

<button id="lang-btn" onclick="toggleLang()" style="
  background: transparent;
  border: 1px solid #3a3a3a;
  color: #B8B3AA;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7em;
  letter-spacing: 0.12em;
  padding: 3px 10px;
  cursor: pointer;
  text-transform: uppercase;
  transition: border-color 0.2s, color 0.2s;
  margin-bottom: 1.4em;
" onmouseover="this.style.borderColor='#9B3D2E';this.style.color='#9B3D2E'" onmouseout="this.style.borderColor='#3a3a3a';this.style.color='#B8B3AA'">ES</button>

<div id="content-en">

### About

Hi! I am a Postdoctoral Research Fellow at [UNIST](https://www.unist.ac.kr/), working with [Jaejun Yoo](https://sites.google.com/view/jaejunyoo/home) in the [Lab of Advanced Imaging Technology (LAIT)](https://sites.google.com/view/jaejunyoo/home). I received my Ph.D. from the [Graduate School of Data Science](https://gsds.kaist.ac.kr/) at [KAIST](https://www.kaist.ac.kr/en/), where I was advised by [Mun Yong Yi](http://kirc.kaist.ac.kr/people_director.html) as part of [KIRC](http://kirc.kaist.ac.kr/) (*Knowledge Innovation Research Center*). My research interests are in developing and applying deep learning-based systems to assist and improve the fields of healthcare, visual recognition, and computational imaging.

### Education

- Ph.D., Graduate School of Data Science, KAIST, South Korea
*Feb 2020 - Feb 2025*
- M.S., Graduate School of Knowledge Service Engineering, KAIST, South Korea
*Feb 2018 - Feb 2020*
- B.S., Electronic and Communication Engineering, INTEC, Dominican Republic
*Feb 2011 - Aug 2014*

### Experience

- Postdoctoral Research Fellow, [LAIT](https://sites.google.com/view/jaejunyoo/home), UNIST, South Korea
*Mar 2025 - Present*

</div>

<div id="content-es" style="display:none">

### Acerca de mí

¡Hola! Soy Investigador Postdoctoral en [UNIST](https://www.unist.ac.kr/), trabajando con [Jaejun Yoo](https://sites.google.com/view/jaejunyoo/home) en el [Laboratorio de Tecnología Avanzada de Imagen (LAIT)](https://sites.google.com/view/jaejunyoo/home). Obtuve mi doctorado en la [Escuela de Postgrado de Ciencia de Datos](https://gsds.kaist.ac.kr/) del [KAIST](https://www.kaist.ac.kr/en/), bajo la tutoría de [Mun Yong Yi](http://kirc.kaist.ac.kr/people_director.html) como parte del [KIRC](http://kirc.kaist.ac.kr/) (*Centro de Investigación en Innovación del Conocimiento*). Mis intereses de investigación se centran en el desarrollo y aplicación de sistemas basados en aprendizaje profundo para asistir y mejorar los campos de la salud, el reconocimiento visual y la imagen computacional.

### Educación

- Doctorado, Escuela de Postgrado de Ciencia de Datos, KAIST, Corea del Sur
*Feb 2020 - Feb 2025*
- Maestría, Escuela de Postgrado de Ingeniería de Servicios del Conocimiento, KAIST, Corea del Sur
*Feb 2018 - Feb 2020*
- Licenciatura, Ingeniería Electrónica y de Comunicaciones, INTEC, República Dominicana
*Feb 2011 - Ago 2014*

### Experiencia

- Investigador Postdoctoral, [LAIT](https://sites.google.com/view/jaejunyoo/home), UNIST, Corea del Sur
*Mar 2025 - Presente*

</div>

### Publications

- **Quinones, W. R.**, Ashraf, M., & Yi, M. Y. (2021, December). Impact of Patch Extraction Variables on Histopathological Imagery Classification Using Convolution Neural Networks. In 2021 International Conference on Computational Science and Computational Intelligence (CSCI) (pp. 1176-1181). IEEE.

- Ashraf, M., **Quinones, W. R.**, Kim, M., Ko, Y. S., & Yi, M. Y. (2022). A loss-based patch label denoising method for improving whole-slide image analysis using a convolutional neural network. Scientific reports, 12(1), 1392.

- Ko, Y. S., Choi, Y. M., Kim, M., Park, Y., Ashraf, M., **Quiñones Robles, W. R.**, ... & Yi, M. Y. (2022). Improving quality control in the routine practice for histopathological interpretation of gastrointestinal endoscopic biopsies using artificial intelligence. Plos one, 17(12), e0278542.

- Kim, M., **Quiñones Robles, W. R.**, Ko, Y. S., Wong, B., Lee, S., & Yi, M. Y. (2024). A predicted-loss based active learning approach for robust cancer pathology image analysis in the workplace. BMC Medical Imaging, 24(1), 5.

- Noree, S., **Quiñones Robles, W. R.**, Ko, Y. S., & Yi, M. Y. (2025). Leveraging commonality across multiple tissue slices for enhanced whole slide image classification using graph convolutional networks. BMC Medical Imaging, 25(1), 11.

- Kim, J. W., Wong, B., Fu, H., **Quiñones Robles, W. R.**, Ko, Y. S., & Yi, M. Y. (2025). MicroMIL: Graph-Based Multiple Instance Learning for Context-Aware Diagnosis with Microscopic Images. In International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI).

### Preprints / Under Review

- **Quiñones Robles, W. R.**, Noree, S., Ko, Y. S., & Yi, M. Y. (2024). Artificial Feature Maps using Fractals: A New Data Augmentation Strategy for Deep Learning-based Whole-Slide Image Analysis. Under Review.

- **Quiñones Robles, W. R.**, Noree, S., Ko, Y. S., Wong, B., Kim, J., & Yi, M. Y. (2025). Leveraging Spatial Context for Positive Pair Sampling in Histopathology Image Representation Learning. arXiv preprint.

<script>
function toggleLang() {
  var en = document.getElementById('content-en');
  var es = document.getElementById('content-es');
  var btn = document.getElementById('lang-btn');
  if (en.style.display !== 'none') {
    en.style.display = 'none';
    es.style.display = 'block';
    btn.textContent = 'EN';
  } else {
    en.style.display = 'block';
    es.style.display = 'none';
    btn.textContent = 'ES';
  }
}
</script>
