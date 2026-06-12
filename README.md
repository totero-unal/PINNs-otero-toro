# 🩸 OneDBioPINN: Physics-Informed Neural Networks for 1D Arterial Hemodynamics

Este repositorio contiene la implementación de una **Red Neuronal Informada por la Física (PINN)** diseñada para simular y predecir el flujo sanguíneo (hemodinámica) en redes arteriales unidimensionales (1D). El modelo es capaz de resolver las ecuaciones de Navier-Stokes 1D y estimar variables clínicas críticas, como la **presión arterial**, utilizando únicamente mediciones limitadas de área y velocidad.

La arquitectura está basada en el trabajo pionero de *Kissas et al.* y aborda uno de los desafíos más complejos de las PINNs biológicas: el colapso por diferencia de escalas matemáticas.

---

## 🚀 Características Clave

* **Resolución Multivaso:** Capacidad para modelar desde una bifurcación simple de 3 vasos hasta estructuras complejas de 4 vasos (Arco Aórtico y Arteria Carótida).
* **Física Integrada (Ecuaciones de Navier-Stokes 1D):** La red no solo aprende de los datos, sino que sus predicciones están restringidas matemáticamente por las leyes de conservación de masa y momento.
* **Adimensionalización y Normalización Estricta:** Implementa un sistema de escalado que evita el desvanecimiento de gradientes provocado por la disparidad de magnitudes entre presiones (escalas grandes) y áreas (escalas microscópicas).
* **Transfer Learning incorporado:** Permite restaurar modelos pre-entrenados mediante *checkpoints* (`.ckpt`) para realizar ajustes finos (*fine-tuning*) rápidos en pocos minutos en lugar de horas.

---

## 🧠 Fundamento Matemático

A diferencia de la IA tradicional, la función de pérdida ($Loss$) de este modelo actúa como un juez físico a través de tres componentes:

1. **Pérdida de Medición:** Minimiza el error entre los datos reales del paciente y las predicciones de la red.
2. **Pérdida Residual:** Evalúa si el Área ($A$), la Velocidad ($u$) y la Presión ($p$) cumplen con las ecuaciones diferenciales de flujo en el interior de los vasos.
3. **Pérdida de Interfaz (Bifurcaciones):** Garantiza la conservación de la masa y la continuidad del momento dinámico en los puntos de división arterial:

$$\Delta Q = Q_{entrada} - \sum Q_{salidas} = 0$$

---

## 📁 Estructura del Repositorio

| Archivo / Directorio | Descripción |
| :--- | :--- |
| `OneDBioPINN.py` | Clase principal que construye el grafo en TensorFlow, maneja la física y el entrenamiento. |
| `main_bifurcation.py` | Script de ejecución para una bifurcación simple de 3 vasos arteriales. |
| `main_aortic_carotid.py` | Script avanzado para la red de 4 vasos del arco aórtico. |
| `/data` | Archivos `.npy` con las geometrías, tiempos y mediciones de velocidad/área. |
| `/SavedModels` | Puntos de control (`.ckpt`) con pesos pre-entrenados para *Transfer Learning*. |

---

## 🛠️ Requisitos e Instalación

Debido a la gestión nativa de grafos computacionales y optimizaciones de gradientes requeridas por las PINNs originales, este proyecto utiliza la API de **TensorFlow 1.x**.

Instala las dependencias necesarias en tu entorno de Python:

```bash
pip install tensorflow==2.15.0  # O cualquier versión moderna de TF2
pip install numpy matplotlib