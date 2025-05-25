# 🖐️ Reconocimiento de Gestos con la Mano Derecha

Este proyecto usa **MediaPipe**, **OpenCV** y **Pygame** para detectar gestos con la mano derecha en tiempo real a través de la webcam. Cuando se reconoce un gesto, se muestra una imagen correspondiente superpuesta en la pantalla.

## 📦 Requisitos

Asegúrate de tener Python 3.7 o superior instalado.

### Dependencias:

Instálalas con pip:

```bash
pip install opencv-python mediapipe pygame numpy
```

## 📂 Estructura del Proyecto

```
📁 handGesture/
│
├── main.py                # Código principal de reconocimiento de gestos
├── imgs/                  # Carpeta que contiene las imágenes PNG de cada gesto
│   ├── okey.png
│   ├── peace.png
│   ├── rock.png
│   └── ... (otros gestos)
```

## 🧠 Gestos Reconocidos

El proyecto puede detectar y mostrar imágenes para los siguientes gestos:

- ✋ `givemefive`  
- 👌 `okey`  
- ✊ `blm`  
- 🤘 `rock`  
- 🤟 `yey`  
- 🖕 `fucku`  
- 👌 `perfect`  
- ✌️ `peace`  
- 🤙 `surf`  
- 👎 `bad`  
- 👊 `punch`  
- 🤞 `promise`  
- 🤌 `italian`  
- 👉 `tiny`  
- 👈 `pistol`  
- 🫰 `uwu`  
- 👆 `looser`  
- ☝️ `friki`  
- 🖖 `vulcano`  
- ☹️ `no_gesture` (cuando no se detecta ningún gesto)

## ▶️ Cómo Ejecutar

1. Coloca las imágenes de cada gesto (con fondo transparente preferiblemente) en la carpeta `imgs` y nómbralas exactamente como los nombres indicados arriba (`okey.png`, `rock.png`, etc.).
2. Ejecuta el script:

```bash
python main.py
```

3. Usa tu **mano derecha** delante de la cámara para mostrar gestos. El programa detectará el gesto y mostrará una imagen correspondiente en la pantalla.

## 📝 Notas

- Asegúrate de tener buena iluminación para mejorar la detección.
- Este sistema está calibrado para una **sola mano (derecha)**.
- Puedes cerrar el programa presionando la tecla **ESC**.

## 🎨 Créditos

Desarrollado utilizando:

- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [Pygame](https://www.pygame.org/)
